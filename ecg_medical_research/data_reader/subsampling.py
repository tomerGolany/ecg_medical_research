"""Sub-sample ECG records."""
import numpy as np


def subsample_ecg(ecg_signal: np.ndarray):
    """Subsample all ECGs to 10 seconds duration sampled at 500HZ - 5000 examples."""
    if ecg_signal.shape[1] == 10000:
        sub_sampled_ecg = ecg_signal[:, ::2]
    elif ecg_signal.shape[1] == 5499 or ecg_signal.shape[1] == 8499:
        sub_sampled_ecg = ecg_signal[:, :5000]
    elif ecg_signal.shape[1] == 4999:
        sub_sampled_ecg = np.concatenate([ecg_signal, np.zeros((12, 1))], axis=1)
    else:
        raise ValueError(f"Unexpected ECG shape: {ecg_signal.shape}")
    return sub_sampled_ecg


if __name__ == "__main__":

    """
    Sampling rates:
    500.0     13761
    1000.0     1121
    Name: samp_rates, dtype: int64
    
    Durations:
    10.998    13718
    10.000     1121
    9.998        42
    16.998        1
    Name: duration, dtype: int64
    
    Number of samples:
    5499     13718
    10000     1121
    4999        42
    8499         1
    """

    fake_ecg_1 = np.ones((12, 10000))
    print(subsample_ecg(fake_ecg_1).shape)

    fake_ecg_2 = np.ones((12, 5499))
    print(fake_ecg_2[:, :5000].shape)

    fake_ecg_3 = np.ones((12, 8499))
    print(fake_ecg_3[:, :5000].shape)

    fake_ecg_4 = np.ones((12, 4999))
    print(np.concatenate([fake_ecg_4, np.zeros((12, 1))], axis=1).shape)