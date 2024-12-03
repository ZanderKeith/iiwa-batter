import os
import time

import dill
import numpy as np

from iiwa_batter import (
    PACKAGE_ROOT,
    CONTROL_DT,
    CONTACT_DT,
    NUM_JOINTS,
)
from iiwa_batter.physics import (
    find_ball_initial_velocity,
    ball_flight_path,
    FLIGHT_TIME_MULTIPLE,
)
from iiwa_batter.swing_optimization.stochastic_gradient_descent import(
    PITCH_START_POSITION,
)
from iiwa_batter.trajectory_library import (
    LIBRARY_POSITIONS,
    LIBRARY_SPEEDS_MPH,
    MAIN_SPEED,
    MAIN_POSITION,
    Trajectory,
)
from iiwa_batter.robot_constraints.get_joint_constraints import JOINT_CONSTRAINTS
from iiwa_batter.swing_simulator import setup_simulator, reset_systems, run_swing_simulation, parse_simulation_state

from iiwa_batter.real_time_operation import real_time_operation

NUM_NEW_PITCHES = 20

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
        "ball_velocity_sample_distribution": 0.12   # 0.1 m/s
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

def make_pitches(robot):
    np.random.seed(0)

    speed_min = min(LIBRARY_SPEEDS_MPH)
    speed_max = max(LIBRARY_SPEEDS_MPH)

    y_positions = [position[1] for position in LIBRARY_POSITIONS]
    z_positions = [position[2] for position in LIBRARY_POSITIONS]

    y_min = min(y_positions)
    y_max = max(y_positions)

    z_min = min(z_positions)
    z_max = max(z_positions)

    pitches = []
    # for speed in LIBRARY_SPEEDS_MPH:
    #     for position in LIBRARY_POSITIONS:
    #         pitches.append((speed, position))
    
    for i in range(NUM_NEW_PITCHES):
        speed = np.random.uniform(speed_min, speed_max)
        position = np.array([0, np.random.uniform(y_min, y_max), np.random.uniform(z_min, z_max)])
        pitches.append((speed, position))

    # Save the pitches
    save_directory = f"{PACKAGE_ROOT}/../benchmarks/{robot}"
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    with open(f"{save_directory}/pitches.dill", "wb") as f:
        dill.dump(pitches, f)


def benchmark_pitches(robot, case):
    with open(f"{PACKAGE_ROOT}/../benchmarks/{robot}/pitches.dill", "rb") as f:
        pitches = dill.load(f)
    case_dict = TEST_CASES[case]

    # Initial position is same for all pitches
    start_position_loader = Trajectory(robot, MAIN_SPEED, MAIN_POSITION, "main")
    main_initial_position, _, _ = start_position_loader.load_best_trajectory()
    simulator, diagram = setup_simulator(torque_trajectory={}, model_urdf=robot, dt=CONTACT_DT, robot_constraints=JOINT_CONSTRAINTS[robot])
    
    total_pitches = len(pitches)
    total_hits = 0
    total_fouls = 0
    total_strikes = 0
    total_collisions = 0
    benchmark_results = {"taken_trajectories": []}
    for speed, position in pitches:
        # Run the real-time operation
        taken_trajectory, _ = real_time_operation(
            robot, 
            speed, 
            position,
            pitch_position_measurement_error=case_dict["pitch_position_measurement_error"],
            pitch_speed_measurement_error=case_dict["pitch_speed_measurement_error"],
            joint_position_measurement_error=0,
            joint_velocity_measurement_error=0,
            joint_position_sample_distribution=0,
            joint_velocity_sample_distribution=0,
            ball_position_measurement_error=case_dict["ball_position_measurement_error"],
            ball_velocity_measurement_error=case_dict["ball_velocity_measurement_error"],
            ball_position_sample_distribution=case_dict["ball_position_sample_distribution"],
            ball_velocity_sample_distribution=case_dict["ball_velocity_sample_distribution"],)
        

        ball_initial_velocity_world, flight_time = find_ball_initial_velocity(speed, position)
        reset_systems(diagram, new_torque_trajectory=taken_trajectory)
        status_dict = run_swing_simulation(
            simulator=simulator,
            diagram=diagram,
            start_time=0,
            end_time=flight_time*FLIGHT_TIME_MULTIPLE+CONTROL_DT,
            initial_joint_positions=main_initial_position,
            initial_joint_velocities=np.zeros(NUM_JOINTS),
            initial_ball_position=PITCH_START_POSITION,
            initial_ball_velocity=ball_initial_velocity_world,
        )

        # Check the result
        result = status_dict["result"]
        if result == "collision":
            total_collisions += 1
            print("COLLISION")
        elif result == "miss":
            total_strikes += 1
            print("MISS")
        elif result == "hit":
            # Check if the ball is fair
            ball_position, ball_velocity = parse_simulation_state(simulator, diagram, "ball")
            path = ball_flight_path(ball_position, ball_velocity)
            land_location = path[-1, :2]
            distance = np.linalg.norm(land_location)
            if land_location[0] < 0:
                total_fouls += 1
                print("FOUL")
            elif np.abs(np.arctan(land_location[1] / land_location[0])) > np.pi / 4:
                total_fouls += 1
                print("FOUL")
            else:
                total_hits += 1
                print("FAIR")

        benchmark_results["taken_trajectories"].append(taken_trajectory)

    benchmark_results["total_pitches"] = total_pitches
    benchmark_results["total_hits"] = total_hits
    benchmark_results["total_fouls"] = total_fouls
    benchmark_results["total_strikes"] = total_strikes
    benchmark_results["total_collisions"] = total_collisions
    benchmark_results["case"] = case_dict

    with open(f"{PACKAGE_ROOT}/../benchmarks/{robot}/{case}.dill", "wb") as f:
        dill.dump(benchmark_results, f)

if __name__ == "__main__":
    with open(f"{PACKAGE_ROOT}/../benchmarks/iiwa14/perfect.dill", "rb") as f:
        benchmark_results = dill.load(f)
    start_time = time.time()
    make_pitches("iiwa14")
    benchmark_pitches("iiwa14", "perfect")
    benchmark_pitches("iiwa14", "low_noise")
    benchmark_pitches("iiwa14", "high_noise")
    end_time = time.time()
    print(f"Total time: {end_time-start_time}")

