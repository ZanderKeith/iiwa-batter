import os

import dill
import numpy as np

from iiwa_batter import (
    PACKAGE_ROOT,
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
)

NUM_INITIAL_POSITIONS = 8
MAIN_COARSE_ITERATIONS = 1000
MAIN_FINE_ITERATIONS = 10
GROUP_COARSE_ITERATIONS = 200
GROUP_FINE_ITERATIONS = 10

LIBRARY_SPEEDS_MPH = [70, 80, 90]
LIBRARY_POSITIONS = [
    [0, 0, 0.6],
]

class Trajectory:
    def __init__(self, robot, target_speed_mph, target_position, type):
        self.robot = robot
        self.target_speed_mph = target_speed_mph
        self.target_position = target_position
        if type not in ["coarse_0", "coarse_1", "coarse_2", "coarse_3", "fine", "group_coarse", "final"]:
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


def make_trajectory_library(targets, main_target):
    # Here are the steps to complete this process:

    # 1. Find the optimal swing for the main target.
    # This will take the form of starting from several initial positions,
    # a coarse optimization and then a fine optimization.

    # 2. Using the best swing at the main target, do a coarse optimization for the rest of the targets
    # This loop will adjust the initial position and the control vector of the swing to hit each target
    # After this is done, another fine optimization will be done

    # And that's it!

    pass