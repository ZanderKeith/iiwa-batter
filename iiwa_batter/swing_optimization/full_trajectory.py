import numpy as np
from multiprocessing import Pool

from pydrake.all import (
    Diagram,
    Simulator,
)
from iiwa_batter import (
    CONTACT_DT,
    PITCH_DT,
    CONTROL_DT,
    NUM_JOINTS,
)

from iiwa_batter.physics import (
    PITCH_START_POSITION,
    FLIGHT_TIME_MULTIPLE,
    find_ball_initial_velocity,
    ball_flight_path,
    ball_distance_multiplier,
)
from iiwa_batter.robot_constraints.get_joint_constraints import JOINT_CONSTRAINTS
from iiwa_batter.swing_simulator import (
    parse_simulation_state,
    reset_systems,
    run_swing_simulation,
    setup_simulator,
)
from iiwa_batter.swing_optimization.stochastic_gradient_descent import (
    make_torque_trajectory,
    perturb_vector,
    descent_step,
)

# This actually works somewhat well... I'm surprised it isn't unbearably slow
# This shall be the backup plan in case the more 'intelligently designed' optimization doesn't work

def full_trajectory_reward(
    simulator: Simulator,
    diagram: Diagram,
    initial_joint_positions,
    control_vector,
    ball_initial_velocity,
    ball_time_of_flight,
):

    trajectory_timesteps = np.arange(0, ball_time_of_flight+CONTROL_DT, CONTROL_DT)
    torque_trajectory = make_torque_trajectory(control_vector, trajectory_timesteps)
    reset_systems(diagram, torque_trajectory)

    status_dict = run_swing_simulation(
        simulator,
        diagram,
        start_time=0,
        end_time=ball_time_of_flight*FLIGHT_TIME_MULTIPLE,
        initial_joint_positions=initial_joint_positions,
        initial_joint_velocities=np.zeros(NUM_JOINTS),
        initial_ball_position=PITCH_START_POSITION,
        initial_ball_velocity=ball_initial_velocity,
    )

    result = status_dict["result"]
    if result == "collision":
        severity = status_dict["contact_severity"]
        if severity <= 10:
            return (-1 * severity) - 20
        else:
            return (-10 * np.log10(severity)) - 20
    elif result == "miss":
        return -10 * status_dict["closest_approach"]
    elif result == "hit":
        ball_position, ball_velocity = parse_simulation_state(simulator, diagram, "ball")
        path = ball_flight_path(ball_position, ball_velocity)
        land_location = path[-1, :2]
        distance = np.linalg.norm(land_location)
        multiplier = ball_distance_multiplier(land_location)
        return distance * multiplier
    else:
        raise ValueError(f"Unknown result: {result}")

def single_full_trajectory_torque_only(
    simulator: Simulator,
    diagram: Diagram,
    initial_joint_positions,
    original_control_vector,
    ball_initial_velocity,
    ball_time_of_flight,
    torque_constraints,
    learning_rate=1,
    iterations=10,
):
    """Run stochastic optimization on the control vector for a single full trajectory, only updating the torque values"""

    present_control_vector = original_control_vector

    best_reward = -np.inf
    best_control_vector = present_control_vector
    for _ in range(iterations):
        present_reward = full_trajectory_reward(
            simulator,
            diagram,
            initial_joint_positions,
            present_control_vector,
            ball_initial_velocity,
            ball_time_of_flight,
        )

        if present_reward > best_reward:
            best_reward = present_reward
            best_control_vector = present_control_vector

        perturbed_control_vector = perturb_vector(present_control_vector, 1, torque_constraints, -torque_constraints)
        perturbed_reward = full_trajectory_reward(
            simulator,
            diagram,
            initial_joint_positions,
            perturbed_control_vector,
            ball_initial_velocity,
            ball_time_of_flight,
        )

        if perturbed_reward > best_reward:
            best_reward = perturbed_reward
            best_control_vector = perturbed_control_vector

        updated_control_vector = descent_step(
            present_control_vector,
            perturbed_control_vector,
            present_reward,
            perturbed_reward,
            learning_rate,
            torque_constraints,
            -torque_constraints,
        )

        present_control_vector = updated_control_vector

    return best_control_vector, best_reward


def single_full_trajectory_torque_and_position(
    simulator: Simulator,
    diagram: Diagram,
    original_initial_joint_positions,
    original_control_vector,
    ball_initial_velocity,
    ball_time_of_flight,
    position_constraints_upper,
    position_constraints_lower,
    torque_constraints,
    best_reward=-np.inf,
    learning_rate=1,
    iterations=10,
    return_best=True,
):
    """Run stochastic optimization on the control vector for a single full trajectory, only updating the torque values"""

    present_initial_position = original_initial_joint_positions
    present_control_vector = original_control_vector

    for i in range(iterations):
        present_reward = full_trajectory_reward(
            simulator,
            diagram,
            present_initial_position,
            present_control_vector,
            ball_initial_velocity,
            ball_time_of_flight,
        )

        if present_reward > best_reward:
            best_reward = present_reward
            best_initial_position = present_initial_position
            best_control_vector = present_control_vector

        perturbed_initial_position = perturb_vector(present_initial_position, np.deg2rad(1), position_constraints_upper, position_constraints_lower)
        perturbed_control_vector = perturb_vector(present_control_vector, 1, torque_constraints, -torque_constraints)
        perturbed_reward = full_trajectory_reward(
            simulator,
            diagram,
            perturbed_initial_position,
            perturbed_control_vector,
            ball_initial_velocity,
            ball_time_of_flight,
        )

        if perturbed_reward > best_reward:
            best_reward = perturbed_reward
            best_initial_position = perturbed_initial_position
            best_control_vector = perturbed_control_vector

        updated_initial_position = descent_step(
            present_initial_position,
            perturbed_initial_position,
            present_reward,
            perturbed_reward,
            learning_rate,
            position_constraints_upper,
            position_constraints_lower,
        )
        updated_control_vector = descent_step(
            present_control_vector,
            perturbed_control_vector,
            present_reward,
            perturbed_reward,
            learning_rate,
            torque_constraints,
            -torque_constraints,
        )

        # Only update the present values if we are not on the last iteration
        if i < iterations - 1:
            present_initial_position = updated_initial_position
            present_control_vector = updated_control_vector
        
    if return_best:
        return best_initial_position, best_control_vector, best_reward
    else:
        return updated_initial_position, updated_control_vector, present_reward


def multi_full_trajectory(
    targets,
    robot_constraints,
    original_initial_position,
    simulation_dt=CONTACT_DT,
    substeps=8, # How many single runs to do between updating the initial position
    save_interval=100,
    top_level_iterations=1000, # How many times to update the initial position
):
    # Initialize the simulator and diagram
    # Keeping these separate for each trajectory, can't re-make them over and over or it causes memory issues
    target_details = {}
    for i, target in enumerate(targets):
        # Determine how many timesteps are needed for the control vector
        simulator, diagram = setup_simulator(
            torque_trajectory={},
            dt = simulation_dt,
            robot_constraints=robot_constraints,
            model_urdf=robot_constraints["model"]
        )
        target_dict = {
            "simulator": simulator,
            "diagram": diagram,
            "target": target,
            "substeps": substeps,
            "initial_position": original_initial_position,
        }
        target_details[i] = target_dict

    best_reward = -np.inf
    best_initial_position = original_initial_position
    for i in range(top_level_iterations):
        pass

        # Determine the loss using the current initial position

        # Determine the loss after after perturbing the initial position

        # Update the initial position based on the gradient