"""ADS-B-oriented physical RF features for long pulse-modulated IQ records."""

from __future__ import annotations

import numpy as np


def _ensure_n2l(values):
    x = np.asarray(values, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f"Expected 3-D IQ data, got {x.shape}")
    if x.shape[1] == 2:
        return x
    if x.shape[-1] == 2:
        return np.transpose(x, (0, 2, 1))
    raise ValueError(f"Expected [N,2,L] or [N,L,2], got {x.shape}")


def _masked_stats(values, mask, eps=1e-8):
    count = np.maximum(mask.sum(axis=1), 1)
    mean = (values * mask).sum(axis=1) / count
    centered = (values - mean[:, None]) * mask
    std = np.sqrt((centered ** 2).sum(axis=1) / count + eps)
    abs_dev = (np.abs(values - mean[:, None]) * mask).sum(axis=1) / count
    return mean, std, abs_dev


def extract_adsb_feature_groups(iq_batch, eps=1e-8):
    """Return pre-registered ADS-B feature groups.

    Groups focus on pulse-edge distortion, oscillator/phase instability,
    I/Q non-circularity, and spectral shape.  No demodulated payload bits,
    ICAO address, or ground-truth labels are used.
    """
    x = _ensure_n2l(iq_batch)
    i_data, q_data = x[:, 0], x[:, 1]
    z = i_data + 1j * q_data
    amp = np.abs(z)
    power = amp ** 2

    # Pulse envelope and transient distortion.
    q10, q25, q50, q75, q90, q99 = np.quantile(
        amp, [0.10, 0.25, 0.50, 0.75, 0.90, 0.99], axis=1
    )
    threshold = q50[:, None] + 0.5 * (q90 - q50)[:, None]
    active = amp >= threshold
    duty = active.mean(axis=1)
    transitions = np.mean(active[:, 1:] != active[:, :-1], axis=1)
    diff_amp = np.diff(amp, axis=1)
    edge_abs = np.abs(diff_amp)
    rise = np.maximum(diff_amp, 0.0)
    fall = np.maximum(-diff_amp, 0.0)
    transient = np.stack(
        [
            q90 - q10,
            q75 - q25,
            q99 / (q50 + eps),
            np.max(power, axis=1) / (np.mean(power, axis=1) + eps),
            duty,
            transitions,
            edge_abs.mean(axis=1),
            edge_abs.std(axis=1),
            np.quantile(edge_abs, 0.90, axis=1),
            np.max(edge_abs, axis=1),
            rise.mean(axis=1),
            rise.std(axis=1),
            fall.mean(axis=1),
            fall.std(axis=1),
        ],
        axis=1,
    )

    # Oscillator/CFO and phase-noise statistics, restricted to active samples.
    phase_diff = np.angle(z[:, 1:] * np.conj(z[:, :-1]))
    phase_mask = active[:, 1:] & active[:, :-1]
    freq_mean, freq_std, freq_mad = _masked_stats(phase_diff, phase_mask, eps)
    phase_accel = np.diff(phase_diff, axis=1)
    accel_mask = phase_mask[:, 1:] & phase_mask[:, :-1]
    acc_mean, acc_std, acc_mad = _masked_stats(phase_accel, accel_mask, eps)
    oscillator = np.stack(
        [
            freq_mean,
            freq_std,
            freq_mad,
            acc_mean,
            acc_std,
            acc_mad,
            np.mean(np.abs(phase_diff), axis=1),
            np.std(phase_diff, axis=1),
        ],
        axis=1,
    )

    # I/Q imbalance, DC leakage and non-circularity.
    i_mean, q_mean = i_data.mean(axis=1), q_data.mean(axis=1)
    i_center = i_data - i_mean[:, None]
    q_center = q_data - q_mean[:, None]
    i_power = np.mean(i_center ** 2, axis=1)
    q_power = np.mean(q_center ** 2, axis=1)
    cross = np.mean(i_center * q_center, axis=1)
    total_power = i_power + q_power + eps
    pseudo_cov = np.mean((z - z.mean(axis=1, keepdims=True)) ** 2, axis=1)
    covariance_delta = np.sqrt((i_power - q_power) ** 2 + 4.0 * cross ** 2)
    eig_ratio = (total_power + covariance_delta) / (total_power - covariance_delta + eps)
    iq_hardware = np.stack(
        [
            np.log((i_power + eps) / (q_power + eps)),
            (i_power - q_power) / total_power,
            cross / (np.sqrt(i_power * q_power) + eps),
            np.abs(pseudo_cov) / total_power,
            np.angle(pseudo_cov),
            np.sqrt(i_mean ** 2 + q_mean ** 2) / (np.sqrt(total_power) + eps),
            eig_ratio,
        ],
        axis=1,
    )

    # Normalized spectrum shape; absolute received power is deliberately absent.
    spectrum = np.fft.fftshift(np.fft.fft(z, axis=1), axes=1)
    spec_power = np.abs(spectrum) ** 2
    probability = spec_power / (spec_power.sum(axis=1, keepdims=True) + eps)
    freq = np.linspace(-1.0, 1.0, z.shape[1], dtype=np.float32)[None, :]
    centroid = np.sum(probability * freq, axis=1)
    centered_freq = freq - centroid[:, None]
    bandwidth = np.sqrt(np.sum(probability * centered_freq ** 2, axis=1) + eps)
    skew = np.sum(probability * (centered_freq / (bandwidth[:, None] + eps)) ** 3, axis=1)
    kurt = np.sum(probability * (centered_freq / (bandwidth[:, None] + eps)) ** 4, axis=1)
    entropy = -np.sum(probability * np.log(probability + eps), axis=1) / np.log(z.shape[1])
    flatness = np.exp(np.mean(np.log(spec_power + eps), axis=1)) / (
        np.mean(spec_power, axis=1) + eps
    )
    peak_ratio = np.max(spec_power, axis=1) / (np.mean(spec_power, axis=1) + eps)
    cumulative = np.cumsum(probability, axis=1)
    rolloff_index = np.argmax(cumulative >= 0.90, axis=1)
    rolloff = rolloff_index / float(z.shape[1] - 1)
    spectral = np.stack(
        [entropy, centroid, bandwidth, skew, kurt, flatness, peak_ratio, rolloff],
        axis=1,
    )

    groups = {
        "adsb_transient_14d": transient,
        "adsb_oscillator_8d": oscillator,
        "adsb_iq_hardware_7d": iq_hardware,
        "adsb_spectral_8d": spectral,
        "adsb_hardware_23d": np.concatenate([oscillator, iq_hardware, spectral], axis=1),
        "adsb_all37d": np.concatenate([transient, oscillator, iq_hardware, spectral], axis=1),
    }
    return {
        name: np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        for name, value in groups.items()
    }


def extract_adsb_all37(iq_batch):
    return extract_adsb_feature_groups(iq_batch)["adsb_all37d"]
