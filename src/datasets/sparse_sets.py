"""
Sparse direction subsets following the SONICOM measurement protocol.

These subsets correspond to:
    - 3 measurements
    - 5 measurements
    - 19 measurements
    - 100 measurements

Indices refer to the fixed SONICOM 793-direction grid.
"""

import numpy as np


# ===============================
# 3 measurements
# ===============================
MEAS3 = np.array([
    4, 11, 414
], dtype=np.int64)


# ===============================
# 5 measurements
# (replace with your actual indices)
# ===============================
MEAS5 = np.array([0, 4, 8, 203, 612], dtype=np.int64)


# ===============================
# 19 measurements
# (replace with your actual indices)
# ===============================
MEAS19 = np.array([
    0,
    4,
    8,
    11,
    14,
    18,
    22,
    265,
    269,
    273,
    278,
    282,
    286,
    529,
    533,
    537,
    542,
    546,
    550,
], dtype=np.int64)


# ===============================
# 100 measurements
# (replace with your actual indices)
# ===============================
MEAS100 = np.array([
    0,
    8,
    19,
    25,
    33,
    38,
    50,
    57,
    65,
    67,
    75,
    84,
    92,
    103,
    117,
    122,
    130,
    134,
    142,
    149,
    159,
    168,
    176,
    184,
    195,
    201,
    209,
    214,
    226,
    233,
    241,
    243,
    251,
    260,
    268,
    279,
    293,
    298,
    306,
    310,
    318,
    325,
    335,
    344,
    352,
    360,
    371,
    377,
    385,
    390,
    402,
    409,
    417,
    419,
    427,
    436,
    444,
    455,
    469,
    474,
    482,
    486,
    494,
    501,
    511,
    520,
    528,
    536,
    547,
    553,
    561,
    566,
    578,
    585,
    593,
    595,
    603,
    612,
    620,
    631,
    645,
    650,
    658,
    662,
    670,
    677,
    687,
    696,
    704,
    712,
    723,
    729,
    737,
    742,
    754,
    761,
    769,
    771,
    779,
    788,
], dtype=np.int64)


# ===============================
# Utility
# ===============================
def get_sparse_set(n_measurements: int) -> np.ndarray:
    """
    Returns sparse direction indices for given number of measurements.

    Args:
        n_measurements: 3, 5, 19, or 100

    Returns:
        didxs: np.ndarray (D,)
    """
    if n_measurements == 3:
        return MEAS3
    elif n_measurements == 5:
        return MEAS5
    elif n_measurements == 19:
        return MEAS19
    elif n_measurements == 100:
        return MEAS100
    else:
        raise ValueError(
            f"Unsupported number of measurements: {n_measurements}. "
            f"Choose from 3, 5, 19, 100."
        )
