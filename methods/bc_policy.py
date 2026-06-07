"""
Minimal behavior-cloning policy: MLP trained via MSE on (obs, action) pairs.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


OBS_KEYS = [
    'robot0_eef_pos',
    'robot0_eef_quat',
    'robot0_gripper_qpos',
    'akita_black_bowl_1_pos',
    'akita_black_bowl_1_to_robot0_eef_pos',
    'plate_1_pos',
    'plate_1_to_robot0_eef_pos',
]

N_PHASE_GROUPS = 8
# Maps 9 scripted phases → 8 behavioral groups.
# Each group must have consistent z-action direction:
#   TRANSPORT: act_z ≈ +0.4 (upward correction toward SAFE_Z)
#   LOWER:     act_z ≈ −0.5 (downward to plate)
# Sharing these two groups causes BC to average opposing z-actions → arm stalls/crashes.
PHASE_TO_GROUP = {
    0: 0,  # RISE
    1: 1,  # PREGRASP
    2: 2,  # DESCEND (go down, gripper open)
    3: 3,  # GRASP   (hold position, close gripper)
    4: 4,  # LIFT    (go up with bowl, gripper closed)
    5: 5,  # TRANSPORT (move toward plate at SAFE_Z, gripper closed)
    6: 6,  # LOWER     (descend to plate, gripper closed — SEPARATE from TRANSPORT)
    7: 7,  # RELEASE (open gripper)
    8: 7,  # DONE
}


def phase_to_onehot(p):
    oh = np.zeros(N_PHASE_GROUPS, dtype=np.float32)
    oh[PHASE_TO_GROUP.get(int(p), 7)] = 1.0
    return oh


def obs_to_vec(obs_dict):
    """Concatenate selected keys from an obs dict into a flat numpy vector."""
    return np.concatenate([obs_dict[k].flatten() for k in OBS_KEYS])


def obs_dim():
    dims = {'robot0_eef_pos': 3, 'robot0_eef_quat': 4, 'robot0_gripper_qpos': 2,
            'akita_black_bowl_1_pos': 3, 'akita_black_bowl_1_to_robot0_eef_pos': 3,
            'plate_1_pos': 3, 'plate_1_to_robot0_eef_pos': 3}
    return sum(dims[k] for k in OBS_KEYS)


class BCPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dims=(256, 256)):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        # tanh output keeps predictions in (-1, 1) — prevents unbounded
        # extrapolation when rollout obs drifts out of training distribution.
        # All training actions are in [-1, 1] so tanh is a lossless bound.
        layers += [nn.Linear(in_dim, action_dim), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def predict(self, obs_vec, device='cpu'):
        """Single-step inference from a numpy obs vector."""
        with torch.no_grad():
            x = torch.FloatTensor(obs_vec).unsqueeze(0).to(device)
            return self.net(x).squeeze(0).cpu().numpy()


SKIP_RISE_STEPS = 0


def load_dataset(hdf5_path, n_history=1):
    """Build (obs_input, action) pairs with phase one-hot conditioning.

    Reads '_phase' from obs group (recorded during collection). Falls back to
    zeros if missing (legacy datasets without phase recording).
    n_history>1 concatenates previous frames for velocity context.
    """
    import h5py
    observations, actions = [], []
    with h5py.File(hdf5_path, 'r') as f:
        for demo_key in sorted(f.keys()):
            demo = f[demo_key]
            obs_grp = demo['obs']
            T = demo['actions'].shape[0]

            phase_arr = obs_grp['_phase'][:] if '_phase' in obs_grp else np.zeros(T, dtype=np.int32)

            obs_frames = []
            for t in range(T):
                obs_t = {k: obs_grp[k][t] for k in OBS_KEYS}
                obs_frames.append(np.concatenate([obs_to_vec(obs_t), phase_to_onehot(phase_arr[t])]))

            start = SKIP_RISE_STEPS
            for t in range(start, T):
                hist = [obs_frames[max(start, t - h)] for h in range(n_history - 1, -1, -1)]
                observations.append(np.concatenate(hist))
                actions.append(demo['actions'][t])

    return np.array(observations, dtype=np.float32), np.array(actions, dtype=np.float32)


def train(hdf5_path, save_path, cfg, device='cpu'):
    seed = cfg.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_history = cfg.get('n_history', 1)
    obs_data, act_data = load_dataset(hdf5_path, n_history=n_history)
    print(f"Dataset: {len(obs_data)} transitions from {hdf5_path}")

    # Normalize observations
    obs_mean = obs_data.mean(0)
    obs_std  = obs_data.std(0) + 1e-6

    obs_norm = (obs_data - obs_mean) / obs_std
    dataset = TensorDataset(
        torch.FloatTensor(obs_norm),
        torch.FloatTensor(act_data),
    )
    loader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True)

    in_dim = obs_data.shape[1]
    act_dim = act_data.shape[1]
    model = BCPolicy(in_dim, act_dim, hidden_dims=cfg['hidden_dims']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg['lr']),
                                 weight_decay=float(cfg.get('weight_decay', 1e-4)))

    best_loss = float('inf')
    for epoch in range(cfg['n_epochs']):
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
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{cfg['n_epochs']}  loss={avg_loss:.5f}")
        best_loss = min(best_loss, avg_loss)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'model_state': model.state_dict(),
        'obs_mean': obs_mean,
        'obs_std': obs_std,
        'obs_dim': in_dim,
        'act_dim': act_dim,
        'hidden_dims': cfg['hidden_dims'],
        'n_history': n_history,
    }, save_path)
    print(f"Model saved to {save_path}  (best loss={best_loss:.5f})")
    return model, obs_mean, obs_std


def load_policy(ckpt_path, device='cpu'):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = BCPolicy(ckpt['obs_dim'], ckpt['act_dim'], hidden_dims=ckpt['hidden_dims'])
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    n_history = ckpt.get('n_history', 1)
    return model, ckpt['obs_mean'], ckpt['obs_std'], n_history
