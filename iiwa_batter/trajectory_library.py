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
    STRIKE_ZONE_Z,
    STRIKE_ZONE_WIDTH,
    STRIKE_ZONE_HEIGHT,
    SWING_IMPACT_BOUNDING_BOX,
    find_ball_initial_velocity,
)

from iiwa_batter.swing_simulator import (
    parse_simulation_state,
    run_swing_simulation,
    setup_simulator,
)

from iiwa_batter.robot_constraints.get_joint_constraints import JOINT_CONSTRAINTS
from iiwa_batter.swing_optimization.stochastic_gradient_descent import (
    initialize_control_vector,
    make_torque_trajectory,
    expand_control_vector,
    find_initial_positions,
)
from iiwa_batter.swing_optimization.swing_impact import (
    calculate_plate_time_and_ball_state,
    run_swing_impact_optimization,
    run_swing_link_optimization,
    VELOCITY_CAP_FRACTION
)
from iiwa_batter.naive_full_trajectory_optimization import (
    run_naive_full_trajectory_optimization,
    run_naive_full_trajectory_optimization_hot_start,
    run_naive_full_trajectory_optimization_hot_start_torque_only,
)



#STATE = "TEST"
#STATE = "LEARNING_RATE_TUNING"
STATE="FINAL"

if STATE == "FINAL":
    NUM_PROCESSES = 4
    NUM_INITIAL_POSITIONS = 16
    MAIN_IMPACT_ITERATIONS = 1000
    COARSE_LINK_ITERATIONS = 1000
    MAIN_COARSE_ITERATIONS = 1000
    MAIN_FINE_ITERATIONS = 100
    GROUP_COARSE_ITERATIONS = 1000
    GROUP_FINE_ITERATIONS = 100

if STATE == "LEARNING_RATE_TUNING":
    NUM_PROCESSES = 4
    GROUP_COARSE_ITERATIONS = 10
    GROUP_FINE_ITERATIONS = 10

if STATE == "TEST":
    NUM_PROCESSES = 2
    NUM_INITIAL_POSITIONS = NUM_PROCESSES
    MAIN_IMPACT_ITERATIONS = 2
    COARSE_LINK_ITERATIONS = 2
    MAIN_COARSE_ITERATIONS = 2
    MAIN_FINE_ITERATIONS = 1
    GROUP_COARSE_ITERATIONS = 2
    GROUP_FINE_ITERATIONS = 2

GROUP_COARSE_LEARNING_RATE = 10
GROUP_COARSE_TORQUE_VARIANCE = 1
GROUP_FINE_LEARNING_RATE = 10
GROUP_FINE_TORQUE_VARIANCE = 0.1

LIBRARY_SPEEDS_MPH = [90, 80, 70]

z_top = STRIKE_ZONE_HEIGHT/2 + STRIKE_ZONE_Z
z_bot = STRIKE_ZONE_Z - STRIKE_ZONE_HEIGHT/2
y_left = -STRIKE_ZONE_WIDTH/2
y_right = STRIKE_ZONE_WIDTH/2
LIBRARY_POSITIONS = [
    [0, 0, STRIKE_ZONE_Z],
    [0, y_left, z_top],
    [0, y_right, z_top],
    [0, y_left, z_bot],
    [0, y_right, z_bot],
    [0, 0, z_top],
    [0, 0, z_bot],
    [0, y_left, STRIKE_ZONE_Z],
    [0, y_right, STRIKE_ZONE_Z],
]

MAIN_SPEED = LIBRARY_SPEEDS_MPH[0]
MAIN_POSITION = LIBRARY_POSITIONS[0]

class Trajectory:
    allowed_types = ["main", "transfer", "group_coarse", "tune_fine"]
    
    def __init__(self, robot, target_speed_mph, target_position, type):
        self.robot = robot
        self.target_speed_mph = target_speed_mph
        self.target_position = target_position

        if type not in Trajectory.allowed_types:
            raise ValueError(f"Invalid type: {type}")
        self.type = type

    def data_directory(self):
        if self.type == "main":
            return f"{PACKAGE_ROOT}/../trajectories/{self.robot}/main"
        elif self.type == "transfer":
            return f"{PACKAGE_ROOT}/../trajectories/{self.robot}/transfer"
        else:
            return f"{PACKAGE_ROOT}/../trajectories/{self.robot}/library/{self.target_speed_mph}_{self.target_position}"
    
    def result_path(self):
        if self.type in ["main", "transfer"]:
            return f"{self.data_directory()}/{self.target_speed_mph}.dill"
        else:
            return f"{self.data_directory()}/{self.type}.dill"

    def save_training_results(self, results):
        if not os.path.exists(self.data_directory()):
            os.makedirs(self.data_directory())
        with open(self.result_path(), "wb") as f:
            dill.dump(results, f)

    def load_training_results(self):
        with open(self.result_path(), "rb") as f:
            return dill.load(f)
        
    def load_best_trajectory(self):
        status_dict = self.load_training_results()
        initial_position = status_dict["best_initial_position"]
        control_vector = status_dict["best_control_vector"]
        dt = status_dict["optimized_dt"]
        return initial_position, control_vector, dt
        
    def recover_trajectory_paths(self):
        training_results = self.load_training_results()

        initial_joint_positions = training_results["best_initial_position"]
        control_vector = training_results["best_control_vector"]
        dt = training_results["optimized_dt"]

        robot_constraints = JOINT_CONSTRAINTS[self.robot]

        ball_initial_velocity, ball_time_of_flight = find_ball_initial_velocity(self.target_speed_mph, self.target_position)
        trajectory_timesteps = np.arange(0, ball_time_of_flight+CONTROL_DT, CONTROL_DT)
        torque_trajectory = make_torque_trajectory(control_vector, trajectory_timesteps)
        simulator, diagram = setup_simulator(torque_trajectory, model_urdf=self.robot, dt=dt, robot_constraints=robot_constraints)

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

def group_coarse_optimization(robot, target_speed_mph, target_position):
    # Already know how to swing at main position for all speeds, skip this one
    if target_position == MAIN_POSITION:
        return
    save_directory = f"{PACKAGE_ROOT}/../trajectories/{robot}/library/{target_speed_mph}_{target_position}"
    # Can already swing at this position and probably hit something
    if os.path.exists(f"{save_directory}/group_coarse.dill"):
        return
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # I don't think we need to do this anymore since the original trajectories are already long enough to hit the ball
    # # There's a little dosido here where the slower balls need a longer trajectory
    # # In this case, we expand the control vector to start with zeros, since the robot can afford to wait to initiate the swing
    # _, ball_time_of_flight = find_ball_initial_velocity(target_speed_mph, target_position)
    # trajectory_timesteps = np.arange(0, ball_time_of_flight+CONTROL_DT, CONTROL_DT)
    # if trajectory_timesteps.size > best_control_vector.shape[0]:
    #     initial_control_vector = expand_control_vector(best_control_vector, len(trajectory_timesteps))
    # else:
    #     initial_control_vector = best_control_vector

    if target_speed_mph == MAIN_SPEED:
        trajectory = Trajectory(robot, target_speed_mph, MAIN_POSITION, "main")
    else:
        trajectory = Trajectory(robot, target_speed_mph, MAIN_POSITION, "transfer")

    results = trajectory.load_training_results()
    best_initial_position = results["best_initial_position"]
    initial_control_vector = results["best_control_vector"]

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
        learning_rate=GROUP_COARSE_LEARNING_RATE,
        torque_variance=GROUP_COARSE_TORQUE_VARIANCE,
    )

def group_fine_optimization(robot, target_speed_mph, target_position):
    save_directory = f"{PACKAGE_ROOT}/../trajectories/{robot}/library/{target_speed_mph}_{target_position}"
    if os.path.exists(f"{save_directory}/tune_fine.dill"):
        return
    
    # Copy over the main swing results and stop
    if target_position == MAIN_POSITION:
        if target_speed_mph == MAIN_SPEED:
            main_trajectory = Trajectory(robot, target_speed_mph, MAIN_POSITION, "main")
        else:
            main_trajectory = Trajectory(robot, target_speed_mph, MAIN_POSITION, "transfer")

        main_results = main_trajectory.load_training_results()
        new_trajectory = Trajectory(robot, target_speed_mph, target_position, "tune_fine")
        new_trajectory.save_training_results(main_results)
        return

    trajectory = Trajectory(robot, target_speed_mph, target_position, "group_coarse")
    results = trajectory.load_training_results()
    best_initial_position = results["best_initial_position"]
    best_control_vector = results["best_control_vector"]
    run_naive_full_trajectory_optimization_hot_start_torque_only(
        robot=robot,
        target_velocity_mph=target_speed_mph,
        target_position=target_position,
        optimization_name="tune_fine",
        save_directory=save_directory,
        initial_joint_positions=best_initial_position,
        present_control_vector=best_control_vector,
        simulation_dt=CONTACT_DT,
        iterations=GROUP_FINE_ITERATIONS,
        learning_rate=GROUP_FINE_LEARNING_RATE,
        torque_variance=GROUP_FINE_TORQUE_VARIANCE,
    )

def make_trajectory_library(robot):
    start_time = time.time()

    save_directory = f"{PACKAGE_ROOT}/../trajectories/{robot}/library"
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # 1. Using the best swing at the main target as an initial guess, do a coarse optimization for the rest of the targets
    print("Starting group coarse optimization")
    group_coarse_pool = Pool(NUM_PROCESSES)
    group_coarse_pool.starmap(group_coarse_optimization, [(robot, target_speed_mph, target_position) for target_speed_mph in LIBRARY_SPEEDS_MPH for target_position in LIBRARY_POSITIONS])
    group_coarse_pool.close()

    # 2. Using the best swing from the coarse optimization, do a fine optimization for the rest of the targets
    print("Starting group fine optimization")
    group_fine_pool = Pool(NUM_PROCESSES)
    group_fine_pool.starmap(group_fine_optimization, [(robot, target_speed_mph, target_position) for target_speed_mph in LIBRARY_SPEEDS_MPH for target_position in LIBRARY_POSITIONS])
    group_fine_pool.close()

    print(f"Library creation for {robot} finished in {time.time() - start_time:.1f} seconds")

def reset(robot):
    library_directory = f"{PACKAGE_ROOT}/../trajectories/{robot}/library"
    try:
        shutil.rmtree(library_directory)
    except FileNotFoundError:
        pass

if __name__ == "__main__":
    robots = ["iiwa14"]
    for robot in robots:
        #reset(robot)
        make_trajectory_library(robot)