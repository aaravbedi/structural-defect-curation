"""
Seven demonstration quality metrics for curation of robot learning datasets.
Each metric takes a demo (obs_seq dict, action_seq array) and returns a scalar
where higher = better quality.

All metrics truncate demos to TRUNC_T=324 steps before feature extraction to
remove episode length as a trivial proxy for the defect label.  The early-release
defect causes contaminated demos to run the full 500-step horizon while clean
demos finish in ~325 steps; without truncation any length-sensitive feature
achieves AUROC≈1.0 for free.

The honest AUROC ceiling after truncation is ~0.91: 43/47 defective demos
release the gripper before step 324 (mean t_release=199), while 4/47 release
between steps 324–343 and are indistinguishable from clean demos within the
truncation window.
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


# ── constants ─────────────────────────────────────────────────────────────────

# Truncation length = minimum successful episode length (from diagnosis).
TRUNC_T = 324

OBS_KEYS = [
    'robot0_eef_pos',
    'robot0_eef_quat',
    'robot0_gripper_qpos',
    'akita_black_bowl_1_pos',
    'akita_black_bowl_1_to_robot0_eef_pos',
    'plate_1_pos',
    'plate_1_to_robot0_eef_pos',
]


# ── helpers ──────────────────────────────────────────────────────────────────

def truncate_demo(obs_seq, action_seq, T=TRUNC_T):
    """Clip both sequences to the first T timesteps."""
    obs_trunc = {k: v[:T] for k, v in obs_seq.items()}
    act_trunc = action_seq[:T]
    return obs_trunc, act_trunc


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
    """
    obs_seq, action_seq = truncate_demo(obs_seq, action_seq)

    speed = _action_magnitudes(action_seq)
    T = len(speed)

    N = T + padding_zeros * T
    Mhat = np.abs(np.fft.rfft(speed, n=N)) / T
    if Mhat.max() < 1e-10:
        return 0.0
    Mhat = Mhat / Mhat.max()

    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    idx = freqs <= fc
    Mhat_fc = Mhat[idx]
    freqs_fc = freqs[idx]

    above = np.where(Mhat_fc >= amp_th)[0]
    if len(above) == 0:
        return 0.0
    Mhat_crop = Mhat_fc[: above[-1] + 1]
    freqs_crop = freqs_fc[: above[-1] + 1]

    dM = np.diff(Mhat_crop)
    df = np.diff(freqs_crop / fc)
    arc = -np.sqrt((dM ** 2 + df ** 2)).sum()
    return float(arc)


# ── Metric 2: Entropy ────────────────────────────────────────────────────────

def entropy(obs_seq, action_seq):
    """
    Negative std of the action sequence (averaged over dims).
    Higher = less variable = more consistent.
    """
    obs_seq, action_seq = truncate_demo(obs_seq, action_seq)
    return float(-action_seq.std(axis=0).mean())


# ── Metric 3: Gripper Timing ──────────────────────────────────────────────────

def gripper_timing(obs_seq, action_seq):
    """
    Length-independent structural-defect detector.

    Returns the normalized timestep at which the gripper first opens
    (gripper_qpos drops below 0.02 after being above 0.03), divided by
    TRUNC_T so the score lies in (0, 1].  If the gripper stays closed
    throughout the truncated window the demo scores 1.0 (best quality).
    Earlier release = lower score = worse quality.

    This metric directly targets the early-release structural defect without
    relying on episode length.  Honest AUROC ceiling after truncation: ~0.91
    (43/47 defects have t_release < TRUNC_T=324; 4/47 release in [324,343]
    and are undetectable within this window).
    """
    obs_seq, action_seq = truncate_demo(obs_seq, action_seq)

    if 'robot0_gripper_qpos' not in obs_seq:
        return 1.0

    gq = obs_seq['robot0_gripper_qpos'][:, 0]  # first finger
    was_closed = False
    for t, g in enumerate(gq):
        if g > 0.03:
            was_closed = True
        if was_closed and g < 0.02:
            return float(t / TRUNC_T)

    return 1.0  # gripper never released within window → clean demo signal


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
        """clean_demos: list of (obs_seq_dict, action_seq) tuples"""
        feats = []
        for obs_seq, a in clean_demos:
            obs_seq, a = truncate_demo(obs_seq, a)
            feats.append(_action_summary_features(a))
        feats = np.array(feats)
        feats_scaled = self.scaler.fit_transform(feats)
        self.iforest.fit(feats_scaled)
        self._fitted = True

    def score(self, obs_seq, action_seq):
        assert self._fitted, "Call fit() first"
        obs_seq, action_seq = truncate_demo(obs_seq, action_seq)
        feat = _action_summary_features(action_seq).reshape(1, -1)
        feat_scaled = self.scaler.transform(feat)
        return float(self.iforest.decision_function(feat_scaled)[0])


# ── Metric 5: Ensemble ───────────────────────────────────────────────────────

def ensemble(obs_seq, action_seq, w_smooth=0.5, w_grip=0.5):
    """
    Weighted combination of smoothness and gripper_timing.
    Length term removed after truncation fix — all demos are same length.
    """
    # truncation is applied inside each sub-metric
    s = smoothness(obs_seq, action_seq)
    g = gripper_timing(obs_seq, action_seq)
    return float(w_smooth * s + w_grip * g)


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
        feats = []
        for obs_seq, a in clean_demos:
            obs_seq, a = truncate_demo(obs_seq, a)
            feats.append(_state_action_summary(obs_seq, a))
        feats = np.array(feats)
        feats_scaled = self.scaler.fit_transform(feats)
        self.nn.fit(feats_scaled)
        self._fitted = True

    def score(self, obs_seq, action_seq):
        assert self._fitted, "Call fit() first"
        obs_seq, action_seq = truncate_demo(obs_seq, action_seq)
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
            obs_seq, _ = truncate_demo(obs_seq, np.zeros((1, 7)))
            obs = _obs_to_vec(obs_seq)
            all_means.append(obs.mean(axis=0))
        self.dataset_mean_state = np.mean(all_means, axis=0)
        self._fitted = True

    def score(self, obs_seq, action_seq):
        assert self._fitted, "Call fit() first"
        obs_seq, action_seq = truncate_demo(obs_seq, action_seq)
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
    'gripper_timing': gripper_timing,
    'ensemble': ensemble,
}

FITTABLE_METRIC_CLASSES = {
    'isolation_forest': IsolationForestScorer,
    'kNN': KNNScorer,
    'trajectory_alignment': TrajectoryAlignmentScorer,
}
