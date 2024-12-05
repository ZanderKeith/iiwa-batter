from importlib.resources import files

PACKAGE_ROOT = files("iiwa_batter")

CONTACT_DT = 2e-5  # Required for accurate hydroelastic contact simulation
PITCH_DT = 1e-4  # Required to accurately plot ball's flight path and be consistent with the contact dt
CONTROL_DT = 10e-3  # How often the control of the robot is updated. (got decent results with 50ms too)
REALTIME_DT = CONTACT_DT*20 # Used for low fidelity simulation
BLENDER_DT = (1/60) # 60 fps

NUM_JOINTS = 7  # Everything has 7 joints, cool

TEST_CASES = {
    "perfect": {
        "name": "Perfect Measurements",
        "pitch_speed_measurement_error": 0,
        "pitch_position_measurement_error": 0,
        "ball_position_measurement_error": 0,
        "ball_velocity_measurement_error": 0,
        "ball_position_sample_distribution": 0.01, # 1 cm
        "ball_velocity_sample_distribution": 0.1   # 0.1 m/s
    },
    "low_noise": {
        "name": "Low Noise",
        "pitch_speed_measurement_error": 2.23, # 1 m/s
        "pitch_position_measurement_error": 0.02, # 2 cm
        "ball_position_measurement_error": 0.01, # 1 cm
        "ball_velocity_measurement_error": 0.1, # 0.1 m/s
        "ball_position_sample_distribution": 0.012, # 1.2 cm
        "ball_velocity_sample_distribution": 0.12   # 0.12 m/s
    },
    "high_noise": {
        "name": "High Noise",
        "pitch_speed_measurement_error": 4.47, # 2 m/s
        "pitch_position_measurement_error": 0.1, # 10 cm
        "ball_position_measurement_error": 0.05, # 5 cm
        "ball_velocity_measurement_error": 0.5, # 0.5 m/s
        "ball_position_sample_distribution": 0.024, # 2.4 cm
        "ball_velocity_sample_distribution": 0.6   # 0.6 m/s
    }
}