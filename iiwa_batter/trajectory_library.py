import os
import shutil
import time
from multiprocessing import Pool

import dill
import numpy as np

from iiwa_batter import (
    PACKAGE_ROOT,
    CONTACT_DT,
    PITCH_DT,
    CONTROL_DT,
    NUM_JOINTS,
)

from iiwa_batter.physics import (
    PITCH_START_POSITION,
    FLIGHT_TIME_MULTIPLE,
    find_ball_initial_velocity,
)

from iiwa_batter.swing_simulator import (
    parse_simulation_state,
    run_swing_simulation,
    setup_simulator,
)

from iiwa_batter.robot_constraints.get_joint_constraints import JOINT_CONSTRAINTS
from iiwa_batter.swing_optimization.stochastic_gradient_descent import (
    make_torque_trajectory,
    expand_control_vector,
)
from iiwa_batter.naive_full_trajectory_optimization import (
    run_naive_full_trajectory_optimization,
    run_naive_full_trajectory_optimization_hot_start,
    run_naive_full_trajectory_optimization_hot_start_torque_only,
)

NUM_PROCESSES = 8

NUM_INITIAL_POSITIONS = 8
MAIN_COARSE_ITERATIONS = 10 # TODO: Change this to 1000
MAIN_FINE_ITERATIONS = 10
GROUP_COARSE_ITERATIONS = 2
GROUP_FINE_ITERATIONS = 1

LIBRARY_SPEEDS_MPH = [90, 80, 70]
LIBRARY_POSITIONS = [
    [0, 0, 0.6],
]

class Trajectory:
    def __init__(self, robot, target_speed_mph, target_position, type):
        self.robot = robot
        self.target_speed_mph = target_speed_mph
        self.target_position = target_position
        if type not in ["fine", "group_coarse", "final"] and type not in [f"coarse_{i}" for i in range(NUM_INITIAL_POSITIONS)]:
            raise ValueError("Invalid type")
        self.type = type

    def data_directory(self):
        return f"{PACKAGE_ROOT}/../trajectories/{self.robot}/{self.target_speed_mph}_{self.target_position}"
    
    def save_training_results(self, results):
        if not os.path.exists(self.data_directory()):
            os.makedirs(self.data_directory())
        with open(f"{self.data_directory()}/{self.type}.dill", "wb") as f:
            dill.dump(results, f)

    def load_training_results(self):
        with open(f"{self.data_directory()}/{self.type}.dill", "rb") as f:
            return dill.load(f)
        
    def recover_trajectory_paths(self):
        training_results = self.load_training_results()

        initial_joint_positions = training_results["best_initial_position"]
        control_vector = training_results["best_control_vector"]
        dt = training_results["optimized_dt"]

        robot_constraints = JOINT_CONSTRAINTS[self.robot]

        ball_initial_velocity, ball_time_of_flight = find_ball_initial_velocity(self.target_speed_mph, self.target_position)
        trajectory_timesteps = np.arange(0, ball_time_of_flight+CONTROL_DT, CONTROL_DT)
        torque_trajectory = make_torque_trajectory(control_vector, trajectory_timesteps)
        simulator, diagram = setup_simulator(torque_trajectory, dt=dt, robot_constraints=robot_constraints)

        status_dict = run_swing_simulation(
            simulator=simulator,
            diagram=diagram,
            start_time=0,
            end_time=ball_time_of_flight*FLIGHT_TIME_MULTIPLE,
            initial_joint_positions=initial_joint_positions,
            initial_joint_velocities=np.zeros(NUM_JOINTS),
            initial_ball_position=PITCH_START_POSITION,
            initial_ball_velocity=ball_initial_velocity,
            record_state=True,
        )

        return status_dict["state"]

def main_coarse_optimization(robot, target_speed_mph, target_position, initial_position_index):
    save_directory = f"{PACKAGE_ROOT}/../trajectories/{robot}/{target_speed_mph}_{target_position}"
    optimization_name = f"coarse_{initial_position_index}"
    if not os.path.exists(f"{save_directory}/{optimization_name}.dill"):
        run_naive_full_trajectory_optimization(
            robot=robot,
            target_velocity_mph=target_speed_mph,
            target_position=target_position,
            optimization_name=optimization_name,
            save_directory=save_directory,
            simulation_dt=PITCH_DT,
            iterations=MAIN_COARSE_ITERATIONS,
            save_interval=10,
            initial_position_index=initial_position_index,
        )

    trajectory = Trajectory(robot, target_speed_mph, target_position, f"coarse_{initial_position_index}")
    results = trajectory.load_training_results()

    return results["final_best_reward"], initial_position_index



def make_trajectory_library(robot):
    start_time = time.time()

    main_target_speed_mph = LIBRARY_SPEEDS_MPH[0]
    main_target_position = LIBRARY_POSITIONS[0]
    save_directory = f"{PACKAGE_ROOT}/../trajectories/{robot}/{main_target_speed_mph}_{main_target_position}"
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    # 1. Find the optimal swing for the main target.
    # This will take the form of starting from several initial positions,
    # a coarse optimization and then a fine optimization.
    main_pool = Pool(NUM_PROCESSES)
    main_results = main_pool.starmap(main_coarse_optimization, [(robot, main_target_speed_mph, main_target_position, i) for i in range(NUM_INITIAL_POSITIONS)])
    main_answers = []
    for result in main_results:
        main_answers.append(result)

    # Pick the best swing from the coarse optimizations
    best_index_reward = -np.inf
    best_index = None
    for answer in main_answers:
        if answer[0] > best_index_reward:
            best_index_reward = answer[0]
            best_index = answer[1]
        
    print(f"Best index: {best_index}")

    # Now do the fine optimization
    optimization_name = "fine"
    if not os.path.exists(f"{save_directory}/{optimization_name}.dill"):
        trajectory = Trajectory(robot, main_target_speed_mph, main_target_position, f"coarse_{best_index}")
        results = trajectory.load_training_results()
        best_initial_position = results["best_initial_position"]
        best_control_vector = results["best_control_vector"]
        run_naive_full_trajectory_optimization_hot_start(
            robot=robot,
            target_velocity_mph=main_target_speed_mph,
            target_position=main_target_position,
            optimization_name=optimization_name,
            save_directory=save_directory,
            present_initial_position=best_initial_position,
            present_control_vector=best_control_vector,
            simulation_dt=CONTACT_DT,
            iterations=MAIN_FINE_ITERATIONS,
            save_interval=1,
        )

    trajectory = Trajectory(robot, main_target_speed_mph, main_target_position, "fine")
    results = trajectory.load_training_results()
    print(f"Best reward from main swing: {results['final_best_reward']}")
    best_initial_position = results["best_initial_position"]
    best_control_vector = results["best_control_vector"]

    # TODO: Add plots of the training progress.

    # 2. Using the best swing at the main target as an initial guess, do a coarse optimization for the rest of the targets
    for target_speed_mph in LIBRARY_SPEEDS_MPH:
        for target_position in LIBRARY_POSITIONS:
            if target_speed_mph == main_target_speed_mph and target_position == main_target_position:
                continue
            save_directory = f"{PACKAGE_ROOT}/../trajectories/{robot}/{target_speed_mph}_{target_position}"
            if os.path.exists(f"{save_directory}/group_coarse.dill"):
                continue
            elif not os.path.exists(save_directory):
                os.makedirs(save_directory)

            # There's a little dosido here where the slower balls need a longer trajectory
            # In this case, we expand the control vector to start with zeros, since the robot can afford to wait to initiate the swing
            _, ball_time_of_flight = find_ball_initial_velocity(target_speed_mph, target_position)
            trajectory_timesteps = np.arange(0, ball_time_of_flight+CONTROL_DT, CONTROL_DT)
            if trajectory_timesteps.size > best_control_vector.shape[0]:
                initial_control_vector = expand_control_vector(best_control_vector, len(trajectory_timesteps))
            else:
                initial_control_vector = best_control_vector

            run_naive_full_trajectory_optimization_hot_start_torque_only(
                robot=robot,
                target_velocity_mph=target_speed_mph,
                target_position=target_position,
                optimization_name="group_coarse",
                save_directory=save_directory,
                initial_joint_positions=best_initial_position,
                present_control_vector=initial_control_vector,
                simulation_dt=PITCH_DT,
                iterations=GROUP_COARSE_ITERATIONS,
            )

    # Using the best swing from the coarse optimization, do a fine optimization for the rest of the targets
    for target_speed_mph in LIBRARY_SPEEDS_MPH:
        for target_position in LIBRARY_POSITIONS:
            if target_speed_mph == main_target_speed_mph and target_position == main_target_position:
                continue
            save_directory = f"{PACKAGE_ROOT}/../trajectories/{robot}/{target_speed_mph}_{target_position}"
            if os.path.exists(f"{save_directory}/fine.dill"):
                continue
            trajectory = Trajectory(robot, target_speed_mph, target_position, "group_coarse")
            results = trajectory.load_training_results()
            best_initial_position = results["best_initial_position"]
            best_control_vector = results["best_control_vector"]
            run_naive_full_trajectory_optimization_hot_start_torque_only(
                robot=robot,
                target_velocity_mph=target_speed_mph,
                target_position=target_position,
                optimization_name="fine",
                save_directory=save_directory,
                initial_joint_positions=best_initial_position,
                present_control_vector=best_control_vector,
                simulation_dt=CONTACT_DT,
                iterations=GROUP_FINE_ITERATIONS,
            )

    print(f"Finished in {time.time() - start_time:.1f} seconds")

def reset(robot):
    robot_directory = f"{PACKAGE_ROOT}/../trajectories/{robot}"
    try:
        shutil.rmtree(robot_directory)
    except FileNotFoundError:
        pass

if __name__ == "__main__":
    robot = "iiwa14"
    reset(robot)
    make_trajectory_library(robot)