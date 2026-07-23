# features/rf_features.py

import numpy as np
import torch


def _to_numpy(x):
    """
    Convert torch.Tensor or np.ndarray to np.ndarray.
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _ensure_iq_shape(iq_batch: np.ndarray) -> np.ndarray:
    """
    Ensure IQ batch shape is [N, 2, L].

    Supported input:
        [N, 2, L]
        [N, L, 2]
        [2, L]
        [L, 2]
    """
    iq_batch = _to_numpy(iq_batch).astype(np.float32)

    if iq_batch.ndim == 2:
        # Single sample
        if iq_batch.shape[0] == 2:
            iq_batch = iq_batch[None, :, :]
        elif iq_batch.shape[-1] == 2:
            iq_batch = iq_batch.T[None, :, :]
        else:
            raise ValueError(f"Unsupported single IQ shape: {iq_batch.shape}")

    elif iq_batch.ndim == 3:
        if iq_batch.shape[1] == 2:
            pass  # [N, 2, L]
        elif iq_batch.shape[-1] == 2:
            iq_batch = np.transpose(iq_batch, (0, 2, 1))  # [N, L, 2] -> [N, 2, L]
        else:
            raise ValueError(f"Unsupported IQ batch shape: {iq_batch.shape}")
    else:
        raise ValueError(f"Unsupported IQ ndim: {iq_batch.ndim}")

    return iq_batch


def _safe_skew(x: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    """
    Safe skewness calculation.
    """
    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True) + eps
    return np.mean(((x - mean) / std) ** 3, axis=axis)


def _safe_kurtosis(x: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    """
    Safe kurtosis calculation.
    This returns the raw fourth standardized moment, not excess kurtosis.
    """
    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True) + eps
    return np.mean(((x - mean) / std) ** 4, axis=axis)


def _spectral_entropy(signal_complex: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Normalized spectral entropy.

    Args:
        signal_complex: complex IQ signal, shape [N, L]

    Returns:
        entropy: shape [N]
    """
    spectrum = np.fft.fft(signal_complex, axis=-1)
    power = np.abs(spectrum) ** 2

    prob = power / (np.sum(power, axis=-1, keepdims=True) + eps)

    entropy = -np.sum(prob * np.log(prob + eps), axis=-1)
    entropy = entropy / (np.log(signal_complex.shape[-1]) + eps)

    return entropy


def _spectral_centroid_bandwidth(signal_complex: np.ndarray, eps: float = 1e-8):
    """
    Spectral centroid and spectral bandwidth.

    Args:
        signal_complex: complex IQ signal, shape [N, L]

    Returns:
        centroid: shape [N]
        bandwidth: shape [N]
    """
    spectrum = np.fft.fft(signal_complex, axis=-1)
    power = np.abs(spectrum) ** 2

    prob = power / (np.sum(power, axis=-1, keepdims=True) + eps)

    freq_bins = np.linspace(0.0, 1.0, signal_complex.shape[-1], dtype=np.float32)[None, :]

    centroid = np.sum(prob * freq_bins, axis=-1)

    bandwidth = np.sqrt(
        np.sum(
            prob * (freq_bins - centroid[:, None]) ** 2,
            axis=-1,
        )
    )

    return centroid, bandwidth


def extract_rf_features_batch(iq_batch, eps: float = 1e-8) -> np.ndarray:
    """
    Extract lightweight classical RF features from IQ signals.

    Input:
        iq_batch:
            [N, 2, L]
            [N, L, 2]
            [2, L]
            [L, 2]

    Output:
        rf_features: [N, D_rf]

    Feature groups:
        1. Amplitude statistics
        2. Phase / frequency statistics
        3. I/Q imbalance statistics
        4. Spectral statistics

    Current output dimension:
        D_rf = 24
    """
    iq_batch = _ensure_iq_shape(iq_batch)

    i_data = iq_batch[:, 0, :]
    q_data = iq_batch[:, 1, :]

    complex_signal = i_data + 1j * q_data

    # ------------------------------------------------------------
    # 1. Amplitude features
    # ------------------------------------------------------------
    amp = np.sqrt(i_data ** 2 + q_data ** 2 + eps)

    amp_mean = np.mean(amp, axis=1)
    amp_std = np.std(amp, axis=1)
    amp_max = np.max(amp, axis=1)
    amp_min = np.min(amp, axis=1)
    amp_rms = np.sqrt(np.mean(amp ** 2, axis=1) + eps)

    power = amp ** 2
    papr = np.max(power, axis=1) / (np.mean(power, axis=1) + eps)

    amp_skew = _safe_skew(amp, axis=1, eps=eps)
    amp_kurt = _safe_kurtosis(amp, axis=1, eps=eps)

    # ------------------------------------------------------------
    # 2. Phase and frequency-related features
    # ------------------------------------------------------------
    phase = np.angle(complex_signal)
    phase_unwrapped = np.unwrap(phase, axis=1)

    # Use unwrapped phase for more stable statistics.
    phase_mean = np.mean(phase_unwrapped, axis=1)
    phase_std = np.std(phase_unwrapped, axis=1)

    # Phase difference approximates instantaneous frequency offset.
    phase_diff = np.diff(phase_unwrapped, axis=1)

    freq_offset_mean = np.mean(phase_diff, axis=1)
    freq_offset_std = np.std(phase_diff, axis=1)

    # ------------------------------------------------------------
    # 3. I/Q imbalance features
    # ------------------------------------------------------------
    i_power = np.mean(i_data ** 2, axis=1)
    q_power = np.mean(q_data ** 2, axis=1)

    iq_power_ratio = i_power / (q_power + eps)
    iq_power_diff = np.abs(i_power - q_power)

    i_mean = np.mean(i_data, axis=1)
    q_mean = np.mean(q_data, axis=1)

    i_std = np.std(i_data, axis=1)
    q_std = np.std(q_data, axis=1)

    iq_corr = np.mean(
        (i_data - i_mean[:, None]) * (q_data - q_mean[:, None]),
        axis=1,
    ) / ((i_std * q_std) + eps)

    # ------------------------------------------------------------
    # 4. Spectral features
    # ------------------------------------------------------------
    spec_entropy = _spectral_entropy(complex_signal, eps=eps)
    spec_centroid, spec_bandwidth = _spectral_centroid_bandwidth(complex_signal, eps=eps)

    features = np.stack(
        [
            # Amplitude features: 8
            amp_mean,
            amp_std,
            amp_max,
            amp_min,
            amp_rms,
            papr,
            amp_skew,
            amp_kurt,

            # Phase / frequency features: 4
            phase_mean,
            phase_std,
            freq_offset_mean,
            freq_offset_std,

            # I/Q imbalance features: 9
            i_power,
            q_power,
            iq_power_ratio,
            iq_power_diff,
            i_mean,
            q_mean,
            i_std,
            q_std,
            iq_corr,

            # Spectral features: 3
            spec_entropy,
            spec_centroid,
            spec_bandwidth,
        ],
        axis=1,
    )

    features = np.nan_to_num(
        features,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return features.astype(np.float32)


def get_rf_feature_names():
    """
    Return feature names in the same order as extract_rf_features_batch().
    """
    return [
        # Amplitude features
        "amp_mean",
        "amp_std",
        "amp_max",
        "amp_min",
        "amp_rms",
        "papr",
        "amp_skew",
        "amp_kurtosis",

        # Phase / frequency features
        "phase_mean",
        "phase_std",
        "freq_offset_mean",
        "freq_offset_std",

        # I/Q imbalance features
        "i_power",
        "q_power",
        "iq_power_ratio",
        "iq_power_diff",
        "i_mean",
        "q_mean",
        "i_std",
        "q_std",
        "iq_corr",

        # Spectral features
        "spectral_entropy",
        "spectral_centroid",
        "spectral_bandwidth",
    ]


def standardize_features(features, mean=None, std=None, eps: float = 1e-8):
    """
    Standardize features.

    If mean/std are None, fit them from the current features.

    Returns:
        standardized_features, mean, std
    """
    features = _to_numpy(features).astype(np.float32)

    if mean is None:
        mean = np.mean(features, axis=0, keepdims=True)

    if std is None:
        std = np.std(features, axis=0, keepdims=True)

    standardized = (features - mean) / (std + eps)

    standardized = np.nan_to_num(
        standardized,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return standardized.astype(np.float32), mean, std


def l2_normalize(features, eps: float = 1e-8):
    """
    L2 normalize features sample-wise.

    Note:
        For the first HDBSCAN experiment, prefer StandardScaler
        before concatenating deep features and RF features.
    """
    features = _to_numpy(features).astype(np.float32)
    norm = np.linalg.norm(features, axis=1, keepdims=True)
    normalized = features / (norm + eps)

    normalized = np.nan_to_num(
        normalized,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return normalized.astype(np.float32)


# Alias for clearer naming in hybrid-feature experiments.
extract_classic_rf_features = extract_rf_features_batch


if __name__ == "__main__":
    # Quick sanity check.
    dummy_iq = np.random.randn(8, 2, 128).astype(np.float32)
    feats = extract_rf_features_batch(dummy_iq)
    names = get_rf_feature_names()

    print("RF feature shape:", feats.shape)
    print("Number of feature names:", len(names))
    print("Feature names:", names)

    assert feats.shape[1] == len(names), "Feature dimension and name count mismatch."