"""
Seven demonstration quality metrics for curation of robot learning datasets.
Each metric takes a demo (obs_seq dict, action_seq array) and returns a scalar
where higher = better quality.
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


# ── helpers ──────────────────────────────────────────────────────────────────

OBS_KEYS = [
    'robot0_eef_pos',
    'robot0_eef_quat',
    'robot0_gripper_qpos',
    'akita_black_bowl_1_pos',
    'akita_black_bowl_1_to_robot0_eef_pos',
    'plate_1_pos',
    'plate_1_to_robot0_eef_pos',
]


def _obs_to_vec(obs_seq_dict):
    """Convert obs dict of arrays (T, D_k) → (T, D_total)."""
    parts = []
    for k in OBS_KEYS:
        if k in obs_seq_dict:
            v = obs_seq_dict[k]
            if v.ndim == 1:
                v = v[:, None]
            parts.append(v)
    return np.concatenate(parts, axis=1).astype(np.float32)


def _action_magnitudes(action_seq):
    """Per-timestep L2 norm of the action vector. Shape: (T,)"""
    return np.linalg.norm(action_seq, axis=1)


def _action_summary_features(action_seq):
    """Per-demo summary: [mean, std, max, min, rms] of per-dim magnitudes.
    Returns a flat vector of length 5 * action_dim."""
    mags = np.abs(action_seq)  # (T, A)
    features = np.concatenate([
        mags.mean(axis=0),
        mags.std(axis=0),
        mags.max(axis=0),
        mags.min(axis=0),
        np.sqrt((mags ** 2).mean(axis=0)),
    ])
    return features.astype(np.float32)


def _state_action_summary(obs_seq_dict, action_seq):
    """Concatenate obs mean/std and action summary for trajectory-level features."""
    obs = _obs_to_vec(obs_seq_dict)
    obs_feats = np.concatenate([obs.mean(axis=0), obs.std(axis=0)])
    act_feats = _action_summary_features(action_seq)
    return np.concatenate([obs_feats, act_feats]).astype(np.float32)


# ── Metric 1: Smoothness (SPARC) ─────────────────────────────────────────────

def smoothness(obs_seq, action_seq, fs=20.0, fc=10.0, amp_th=0.05, padding_zeros=4):
    """
    Spectral arc length (SPARC) of the action speed profile.
    Higher (less negative) = smoother.

    obs_seq: dict of obs arrays (unused by this metric)
    action_seq: (T, A) numpy array
    """
    speed = _action_magnitudes(action_seq)
    T = len(speed)

    # FFT of the speed profile
    N = T + padding_zeros * T
    Mhat = np.abs(np.fft.rfft(speed, n=N)) / T
    # Normalise by max
    if Mhat.max() < 1e-10:
        return 0.0
    Mhat = Mhat / Mhat.max()

    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    idx = freqs <= fc
    Mhat_fc = Mhat[idx]
    freqs_fc = freqs[idx]

    # Crop to indices where amplitude is above threshold
    above = np.where(Mhat_fc >= amp_th)[0]
    if len(above) == 0:
        return 0.0
    Mhat_crop = Mhat_fc[: above[-1] + 1]
    freqs_crop = freqs_fc[: above[-1] + 1]

    # Arc length in the normalised frequency-amplitude space
    dM = np.diff(Mhat_crop)
    df = np.diff(freqs_crop / fc)
    arc = -np.sqrt((dM ** 2 + df ** 2)).sum()
    return float(arc)


# ── Metric 2: Entropy ────────────────────────────────────────────────────────

def entropy(obs_seq, action_seq):
    """
    Negative std of the action sequence (averaged over dims).
    Higher = less variable = more consistent (better quality for scripted data).
    """
    return float(-action_seq.std(axis=0).mean())


# ── Metric 3: Length ─────────────────────────────────────────────────────────

def length(obs_seq, action_seq):
    """
    Negative total trajectory length in action space (shorter = better).
    """
    diffs = np.diff(action_seq, axis=0)
    traj_len = np.linalg.norm(diffs, axis=1).sum()
    return float(-traj_len)


# ── Metric 4: Isolation Forest ───────────────────────────────────────────────

class IsolationForestScorer:
    """
    Fit an IsolationForest on per-demo action summary features from clean demos,
    then score new demos. Higher score = less anomalous = better quality.
    """

    def __init__(self, contamination=0.1, n_estimators=100, random_state=42):
        self.iforest = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
        )
        self.scaler = StandardScaler()
        self._fitted = False

    def fit(self, clean_demos):
        """
        clean_demos: list of (obs_seq_dict, action_seq) tuples
        """
        feats = np.array([_action_summary_features(a) for _, a in clean_demos])
        feats_scaled = self.scaler.fit_transform(feats)
        self.iforest.fit(feats_scaled)
        self._fitted = True

    def score(self, obs_seq, action_seq):
        assert self._fitted, "Call fit() first"
        feat = _action_summary_features(action_seq).reshape(1, -1)
        feat_scaled = self.scaler.transform(feat)
        # decision_function: negative anomaly score, higher = more normal
        return float(self.iforest.decision_function(feat_scaled)[0])


# ── Metric 5: Ensemble ───────────────────────────────────────────────────────

def ensemble(obs_seq, action_seq, w_smooth=0.5, w_len=0.3, w_ent=0.2):
    """
    Weighted combination of smoothness + length + entropy.
    Weights chosen to emphasise trajectory quality.
    """
    s = smoothness(obs_seq, action_seq)
    l = length(obs_seq, action_seq)
    e = entropy(obs_seq, action_seq)
    return float(w_smooth * s + w_len * l + w_ent * e)


# ── Metric 6: kNN ────────────────────────────────────────────────────────────

class KNNScorer:
    """
    Score demos by negative mean distance to k nearest clean-demo neighbors
    in a trajectory-level feature space (state + action summary).
    Higher = closer to clean demos = better quality.
    """

    def __init__(self, k=5):
        self.k = k
        self.nn = NearestNeighbors(n_neighbors=k, metric='euclidean')
        self.scaler = StandardScaler()
        self._fitted = False

    def fit(self, clean_demos):
        feats = np.array([_state_action_summary(o, a) for o, a in clean_demos])
        feats_scaled = self.scaler.fit_transform(feats)
        self.nn.fit(feats_scaled)
        self._fitted = True

    def score(self, obs_seq, action_seq):
        assert self._fitted, "Call fit() first"
        feat = _state_action_summary(obs_seq, action_seq).reshape(1, -1)
        feat_scaled = self.scaler.transform(feat)
        dists, _ = self.nn.kneighbors(feat_scaled)
        return float(-dists.mean())


# ── Metric 7: Trajectory Alignment ───────────────────────────────────────────

class TrajectoryAlignmentScorer:
    """
    Cosine similarity between a demo's mean state trajectory and the
    dataset mean state trajectory. Higher = more aligned with the norm.
    """

    def __init__(self):
        self.dataset_mean_state = None
        self._fitted = False

    def fit(self, clean_demos):
        all_means = []
        for obs_seq, _ in clean_demos:
            obs = _obs_to_vec(obs_seq)
            all_means.append(obs.mean(axis=0))
        self.dataset_mean_state = np.mean(all_means, axis=0)
        self._fitted = True

    def score(self, obs_seq, action_seq):
        assert self._fitted, "Call fit() first"
        obs = _obs_to_vec(obs_seq)
        demo_mean = obs.mean(axis=0)
        ref = self.dataset_mean_state
        norm_d = np.linalg.norm(demo_mean)
        norm_r = np.linalg.norm(ref)
        if norm_d < 1e-10 or norm_r < 1e-10:
            return 0.0
        return float(np.dot(demo_mean, ref) / (norm_d * norm_r))


# ── Public API ────────────────────────────────────────────────────────────────

STANDALONE_METRICS = {
    'smoothness': smoothness,
    'entropy': entropy,
    'length': length,
    'ensemble': ensemble,
}

FITTABLE_METRIC_CLASSES = {
    'isolation_forest': IsolationForestScorer,
    'kNN': KNNScorer,
    'trajectory_alignment': TrajectoryAlignmentScorer,
}
