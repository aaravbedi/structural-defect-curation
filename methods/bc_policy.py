"""
Minimal behavior-cloning policy: MLP trained via MSE on (obs, action) pairs.

Obs augmentation: adds a 6-dim one-hot phase label computed deterministically from obs.
Phase label removes BC mode-averaging across phases (PREGRASP, DESCEND, GRASP, etc.)
without using any hidden state — the phase is fully Markovian.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# Base observation keys (eef_quat excluded: near-zero variance → normalization explosions).
OBS_KEYS = [
    'robot0_eef_pos',
    'robot0_gripper_qpos',
    'akita_black_bowl_1_pos',
    'akita_black_bowl_1_to_robot0_eef_pos',
    'plate_1_pos',
    'plate_1_to_robot0_eef_pos',
    'init_bowl_pos',
]

# Phase constants — must match collect_demos.py geometry.
_SAFE_Z   = 1.20
_GRASP_DZ = 0.022
_PLACE_DZ = 0.06
_XY_TOL   = 0.015
_Z_TOL    = 0.010
_GRIP_Q   = 0.020
_HOLD     = 0.060

PHASE_PREGRASP  = 0
PHASE_DESCEND   = 1
PHASE_GRASP     = 2
PHASE_TRANSPORT = 3
PHASE_LOWER     = 4
PHASE_RELEASE   = 5
N_PHASES        = 6


def get_phase(obs_dict, init_bowl_pos, init_plate_pos):
    """
    Compute the deterministic Markovian phase [0-5] from current obs.
    Mirrors the logic in collect_demos.markovian_policy.
    """
    eef        = np.asarray(obs_dict['robot0_eef_pos'])
    b2e        = np.asarray(obs_dict['akita_black_bowl_1_to_robot0_eef_pos'])
    gripper_q  = np.asarray(obs_dict['robot0_gripper_qpos'])
    ib         = np.asarray(init_bowl_pos)
    ip         = np.asarray(init_plate_pos)

    grasp_z = ib[2] + _GRASP_DZ
    place_z = ip[2] + _PLACE_DZ

    grip_closing = gripper_q[0] > _GRIP_Q
    bowl_dist    = np.linalg.norm(b2e)
    bowl_held    = grip_closing and bowl_dist < _HOLD

    def xy_err(a, b):
        return np.linalg.norm(np.asarray(a[:2]) - np.asarray(b[:2]))

    if not bowl_held:
        if xy_err(eef, ib) > _XY_TOL:
            return PHASE_PREGRASP
        elif eef[2] > grasp_z + _Z_TOL:
            return PHASE_DESCEND
        else:
            return PHASE_GRASP
    else:
        if xy_err(eef, ip) > _XY_TOL:
            return PHASE_TRANSPORT
        elif eef[2] > place_z + _Z_TOL:
            return PHASE_LOWER
        else:
            return PHASE_RELEASE


def obs_to_vec(obs_dict):
    """Concatenate base obs keys into a flat numpy vector."""
    return np.concatenate([np.asarray(obs_dict[k]).flatten() for k in OBS_KEYS])


def build_obs_vec(obs_dict, init_bowl_pos, init_plate_pos):
    """Base obs + 6-dim phase one-hot. This is the full input to the BC network."""
    base = obs_to_vec(obs_dict)
    phase = get_phase(obs_dict, init_bowl_pos, init_plate_pos)
    onehot = np.zeros(N_PHASES, dtype=np.float32)
    onehot[phase] = 1.0
    return np.concatenate([base, onehot])


class BCPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dims=(256, 256)):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def predict(self, obs_vec, device='cpu'):
        with torch.no_grad():
            x = torch.FloatTensor(obs_vec).unsqueeze(0).to(device)
            return self.net(x).squeeze(0).cpu().numpy()


def load_dataset(hdf5_path, success_only=True):
    """
    Load obs+actions from HDF5.
    Augments obs with deterministic phase one-hot.
    Optionally filters to successful episodes only.
    """
    import h5py
    observations, actions = [], []
    with h5py.File(hdf5_path, 'r') as f:
        n_total = len(f.keys())
        n_used  = 0
        for demo_key in sorted(f.keys()):
            demo = f[demo_key]
            if success_only and not demo.attrs.get('success', False):
                continue
            n_used += 1
            obs_grp      = demo['obs']
            T            = demo['actions'].shape[0]
            act          = demo['actions'][:]
            init_bowl    = obs_grp['init_bowl_pos'][0]
            init_plate   = obs_grp['plate_1_pos'][0]
            for t in range(T):
                obs_t  = {k: obs_grp[k][t] for k in OBS_KEYS}
                obs_vec = build_obs_vec(obs_t, init_bowl, init_plate)
                observations.append(obs_vec)
                actions.append(act[t])
        print(f"  Using {n_used}/{n_total} demos (success_only={success_only})")
    return np.array(observations, dtype=np.float32), np.array(actions, dtype=np.float32)


def train(hdf5_path, save_path, cfg, device='cpu', success_only=True):
    obs_data, act_data = load_dataset(hdf5_path, success_only=success_only)
    print(f"Dataset: {len(obs_data)} transitions from {hdf5_path}")

    obs_mean = obs_data.mean(0)
    obs_std  = obs_data.std(0) + 1e-6

    obs_norm = (obs_data - obs_mean) / obs_std
    dataset  = TensorDataset(torch.FloatTensor(obs_norm), torch.FloatTensor(act_data))
    loader   = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True)

    in_dim  = obs_data.shape[1]
    act_dim = act_data.shape[1]
    model   = BCPolicy(in_dim, act_dim, hidden_dims=cfg['hidden_dims']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])

    best_loss = float('inf')
    n_epochs  = cfg.get('n_epochs', 200)
    for epoch in range(n_epochs):
        total_loss = 0.0
        for obs_b, act_b in loader:
            obs_b, act_b = obs_b.to(device), act_b.to(device)
            pred = model(obs_b)
            loss = nn.functional.mse_loss(pred, act_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(obs_b)
        avg_loss = total_loss / len(obs_data)
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}  loss={avg_loss:.5f}")
        best_loss = min(best_loss, avg_loss)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'model_state': model.state_dict(),
        'obs_mean':    obs_mean,
        'obs_std':     obs_std,
        'obs_dim':     in_dim,
        'act_dim':     act_dim,
        'hidden_dims': cfg['hidden_dims'],
    }, save_path)
    print(f"Model saved to {save_path}  (best loss={best_loss:.5f})")
    return model, obs_mean, obs_std


def load_policy(ckpt_path, device='cpu'):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = BCPolicy(ckpt['obs_dim'], ckpt['act_dim'], hidden_dims=ckpt['hidden_dims'])
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model, ckpt['obs_mean'], ckpt['obs_std']
