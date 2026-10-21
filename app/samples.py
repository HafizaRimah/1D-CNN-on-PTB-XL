"""
Sample ECG generator and parser for the web app.
Provides realistic synthetic 12-lead ECG signals for testing:
1. Normal Sinus Rhythm (NORM)
2. Inferior Myocardial Infarction (MI)
3. Conduction Disturbance / Bundle Branch Block (CD)
4. Left Ventricular Hypertrophy (HYP)
"""

import numpy as np
from typing import Dict, Tuple

STANDARD_LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def generate_synthetic_ecg(condition: str = "NORM", length: int = 1000) -> np.ndarray:
    """
    Generates a 12-lead ECG signal of shape (12, 1000) at 100 Hz (10 seconds)
    simulating physiological cardiac depolarization and condition-specific patterns.
    """
    t = np.linspace(0, 10, length)  # 10 seconds at 100 Hz
    hr_bpm = 72
    rr_interval = 60.0 / hr_bpm  # ~0.833 s per heartbeat

    # Base P-QRS-T template generator for one beat
    signals = np.zeros((12, length), dtype=np.float32)

    # Number of beats in 10 seconds (~12 beats)
    beat_times = np.arange(0.2, 10.0, rr_interval)

    # Lead projection multipliers (approximate anatomical projections)
    lead_projections = {
        "I": 0.8, "II": 1.0, "III": 0.3, "aVR": -0.9, "aVL": 0.2, "aVF": 0.7,
        "V1": -0.6, "V2": -0.4, "V3": 0.5, "V4": 1.1, "V5": 1.3, "V6": 1.2
    }

    for idx, lead in enumerate(STANDARD_LEAD_NAMES):
        scale = lead_projections[lead]
        lead_signal = np.zeros(length, dtype=np.float32)

        for bt in beat_times:
            # Time relative to R-peak
            dt = t - bt

            # P-wave (at dt = -0.16s, width 0.04s)
            p_amp = 0.15 * scale
            lead_signal += p_amp * np.exp(-((dt + 0.16) ** 2) / (2 * (0.035 ** 2)))

            # Q-wave (at dt = -0.04s)
            q_amp = -0.15 * abs(scale) if scale > 0 else 0.1
            lead_signal += q_amp * np.exp(-((dt + 0.04) ** 2) / (2 * (0.015 ** 2)))

            # R-wave (at dt = 0.0s, sharp peak)
            r_amp = 1.2 * scale
            if condition == "HYP" and lead in ["V5", "V6"]:
                r_amp *= 2.2  # Voltage criteria for Left Ventricular Hypertrophy
            lead_signal += r_amp * np.exp(-(dt ** 2) / (2 * (0.02 ** 2)))

            # S-wave (at dt = +0.05s)
            s_amp = -0.3 * abs(scale)
            if condition == "HYP" and lead in ["V1", "V2"]:
                s_amp *= 2.4  # Deep S wave in V1-V2 for HYP
            elif condition == "CD" and lead in ["V1", "V2"]:
                # Widened, notched QRS (RSR' rabbit ear pattern for bundle branch block)
                lead_signal += 0.8 * np.exp(-((dt - 0.06) ** 2) / (2 * (0.03 ** 2)))
            lead_signal += s_amp * np.exp(-((dt - 0.05) ** 2) / (2 * (0.018 ** 2)))

            # T-wave (at dt = +0.25s)
            t_amp = 0.3 * scale
            # ST-segment elevation for Inferior MI in II, III, aVF
            if condition == "MI" and lead in ["II", "III", "aVF"]:
                # Prominent ST elevation and peaked T wave
                lead_signal += 0.5 * np.exp(-((dt - 0.15) ** 2) / (2 * (0.08 ** 2)))
                t_amp = 0.6
            elif condition == "MI" and lead in ["I", "aVL"]:
                # Reciprocal ST depression
                lead_signal -= 0.3 * np.exp(-((dt - 0.15) ** 2) / (2 * (0.08 ** 2)))

            lead_signal += t_amp * np.exp(-((dt - 0.25) ** 2) / (2 * (0.06 ** 2)))

        # Add subtle baseline drift and mild noise
        drift = 0.05 * np.sin(2 * np.pi * 0.1 * t)
        noise = np.random.normal(0, 0.015, length)
        signals[idx] = lead_signal + drift + noise

    return signals
