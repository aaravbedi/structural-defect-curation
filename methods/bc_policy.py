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
        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def predict(self, obs_vec, device='cpu'):
        """Single-step inference from a numpy obs vector."""
        with torch.no_grad():
            x = torch.FloatTensor(obs_vec).unsqueeze(0).to(device)
            return self.net(x).squeeze(0).cpu().numpy()


def load_dataset(hdf5_path):
    import h5py
    observations, actions = [], []
    with h5py.File(hdf5_path, 'r') as f:
        for demo_key in f.keys():
            demo = f[demo_key]
            obs_grp = demo['obs']
            T = demo['actions'].shape[0]
            for t in range(T):
                obs_t = {k: obs_grp[k][t] for k in OBS_KEYS}
                observations.append(obs_to_vec(obs_t))
                actions.append(demo['actions'][t])
    return np.array(observations, dtype=np.float32), np.array(actions, dtype=np.float32)


def train(hdf5_path, save_path, cfg, device='cpu'):
    obs_data, act_data = load_dataset(hdf5_path)
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
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])

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
    }, save_path)
    print(f"Model saved to {save_path}  (best loss={best_loss:.5f})")
    return model, obs_mean, obs_std


def load_policy(ckpt_path, device='cpu'):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = BCPolicy(ckpt['obs_dim'], ckpt['act_dim'], hidden_dims=ckpt['hidden_dims'])
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model, ckpt['obs_mean'], ckpt['obs_std']
