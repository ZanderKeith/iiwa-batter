import numpy as np

# At certain steps of the process, include the human-created things as an initial guess

SWING_IMPACT = {
    "iiwa14": {
        "plate_position": np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float64),
        "plate_velocity": np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float64),
    }
}

COARSE_LINK = {
    "iiwa14": {
        "initial_position": np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float64),
        # Wow we're really doing this, aren't we?
    }
}