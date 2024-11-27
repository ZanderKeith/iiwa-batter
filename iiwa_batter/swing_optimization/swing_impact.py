import time

import dill
import numpy as np

from pydrake.all import (
    Diagram,
    Simulator,
)
from iiwa_batter import (
    PACKAGE_ROOT,
    CONTACT_DT,
    CONTROL_DT,
    NUM_JOINTS,
)
from iiwa_batter.physics import (
    find_ball_initial_velocity,
    PITCH_START_POSITION,
    SWING_IMPACT_BOUNDING_BOX,
)
from iiwa_batter.swing_simulator import (
    setup_simulator,
    run_swing_simulation,
    parse_simulation_state
)
from iiwa_batter.robot_constraints.get_joint_constraints import JOINT_CONSTRAINTS
from iiwa_batter.swing_optimization.stochastic_gradient_descent import (
    make_trajectory_timesteps,
    find_initial_positions,
    perturb_vector,
    descent_step
)
from iiwa_batter.swing_optimization.partial_trajectory import (
    partial_trajectory_reward
)
from iiwa_batter.plotting import plot_learning

def swing_link_reward():
    pass

def dummy_torque_trajectory(plate_time):
    # Not applying any torques, just letting the bat fly at the ball for a few full timesteps
    dummy_trajectory = {0: np.zeros(NUM_JOINTS), plate_time+5*CONTROL_DT: np.zeros(NUM_JOINTS)}
    return dummy_trajectory

def calculate_plate_time_and_ball_state(target_speed_mph, target_position, simulation_dt=CONTACT_DT):
    ball_initial_velocity, ball_time_of_flight = find_ball_initial_velocity(target_speed_mph, target_position)
    control_timesteps = make_trajectory_timesteps(ball_time_of_flight)
    # Get the timestep just before the ball crosses the plate
    plate_time = control_timesteps[control_timesteps < ball_time_of_flight][-1]

    # Get the position of the ball at that time
    simulator, diagram = setup_simulator(torque_trajectory={}, model_urdf="iiwa14", dt=simulation_dt, add_contact=False)
    run_swing_simulation(
        simulator=simulator,
        diagram=diagram,
        start_time=0,
        end_time=plate_time,
        initial_joint_positions=np.zeros(7),
        initial_joint_velocities=np.zeros(7),
        initial_ball_position=PITCH_START_POSITION,
        initial_ball_velocity=ball_initial_velocity
    )
    ball_position, ball_velocity = parse_simulation_state(simulator, diagram, "ball")
    return plate_time, ball_position, ball_velocity

def single_swing_impact_optimization(
    simulator: Simulator,
    diagram: Diagram,
    plate_time,
    plate_ball_position,
    plate_ball_velocity,
    plate_joint_positions,
    plate_joint_velocities,
    position_constraints_upper,
    position_constraints_lower,
    velocity_constraints_upper,
    velocity_constraints_lower,
    learning_rate=1,
):

    present_reward = partial_trajectory_reward(
        simulator=simulator,
        diagram=diagram,
        start_time=plate_time,
        initial_joint_positions=plate_joint_positions,
        initial_joint_velocities=plate_joint_velocities,
        initial_ball_position=plate_ball_position,
        initial_ball_velocity=plate_ball_velocity,
        torque_trajectory=dummy_torque_trajectory(plate_time),
    )

    perturbed_joint_positions = perturb_vector(plate_joint_positions, learning_rate, position_constraints_upper, position_constraints_lower)
    perturbed_joint_velocities = perturb_vector(plate_joint_velocities, learning_rate, velocity_constraints_upper, velocity_constraints_lower)
    perturbed_reward = partial_trajectory_reward(
        simulator=simulator,
        diagram=diagram,
        start_time=plate_time,
        initial_joint_positions=perturbed_joint_positions,
        initial_joint_velocities=perturbed_joint_velocities,
        initial_ball_position=plate_ball_position,
        initial_ball_velocity=plate_ball_velocity,
        torque_trajectory=dummy_torque_trajectory(plate_time),
    )

    updated_joint_positions = descent_step(
        plate_joint_positions,
        perturbed_joint_positions,
        present_reward,
        perturbed_reward,
        np.deg2rad(learning_rate),
        position_constraints_upper,
        position_constraints_lower
    )
    updated_joint_velocities = descent_step(
        plate_joint_velocities,
        perturbed_joint_velocities,
        present_reward,
        perturbed_reward,
        np.deg2rad(learning_rate),
        velocity_constraints_upper,
        velocity_constraints_lower
    )

    return updated_joint_positions, updated_joint_velocities, present_reward

def run_swing_impact_optimization(
    robot,
    optimization_name,
    save_directory,
    plate_time,
    plate_ball_position,
    plate_ball_velocity,
    velocity_cap_fraction=0.8,
    simulation_dt=CONTACT_DT,
    iterations=100,
    initial_position_index=0,
    learning_rate=1,
    save_interval=1,
    debug_prints=False
):
    start_time = time.time()
    robot_constraints = JOINT_CONSTRAINTS[robot]
    position_constraints_upper = np.array([joint[1] for joint in robot_constraints["joint_range"].values()])
    position_constraints_lower = np.array([joint[0] for joint in robot_constraints["joint_range"].values()])
    velocity_constraints_abs = np.array([velocity for velocity in robot_constraints["joint_velocity"].values()])*velocity_cap_fraction

    simulator, diagram = setup_simulator(torque_trajectory={}, model_urdf=robot, dt=simulation_dt, robot_constraints=robot_constraints)
    present_joint_positions = find_initial_positions(
        simulator, 
        diagram, 
        robot_constraints, 
        initial_position_index+1,
        bounding_box=SWING_IMPACT_BOUNDING_BOX
    )[initial_position_index]
    present_joint_velocities = np.random.uniform(-velocity_constraints_abs, velocity_constraints_abs, NUM_JOINTS)

    training_results = {
        "learning": {}
    }
    best_reward = -np.inf
    for i in range(iterations):
        next_joint_positions, next_joint_velocities, present_reward = single_swing_impact_optimization(
            simulator=simulator,
            diagram=diagram,
            plate_time=plate_time,
            plate_ball_position=plate_ball_position,
            plate_ball_velocity=plate_ball_velocity,
            plate_joint_positions=present_joint_positions,
            plate_joint_velocities=present_joint_velocities,
            position_constraints_upper=position_constraints_upper,
            position_constraints_lower=position_constraints_lower,
            velocity_constraints_upper=velocity_constraints_abs,
            velocity_constraints_lower=-velocity_constraints_abs,
            learning_rate=learning_rate
        )

        if present_reward > best_reward:
            best_reward = present_reward
            best_joint_positions = present_joint_positions
            best_joint_velocities = present_joint_velocities
        
        if i % save_interval == 0:
            training_results["learning"][i] = {
                "present_reward": present_reward,
                "best_reward_so_far": best_reward
            }
            training_results["best_joint_positions"] = best_joint_positions
            training_results["best_joint_velocities"] = best_joint_velocities

        if i < iterations - 1:
            present_joint_positions = next_joint_positions
            present_joint_velocities = next_joint_velocities

        if debug_prints:
            print(f"Swing impact {i}: present reward: {present_reward:.3f}, best reward: {best_reward:.3f}")

    training_results["best_joint_positions"] = best_joint_positions
    training_results["best_joint_velocities"] = best_joint_velocities
    training_results["final_best_reward"] = best_reward
    training_results["optimized_dt"] = simulation_dt

    total_time = time.time() - start_time
    training_results["total_time"] = total_time
    if debug_prints:
        print(f"Swing impact {i} total time: {total_time:.3f}")

    with open(f"{save_directory}/{optimization_name}.dill", "wb") as f:
        dill.dump(training_results, f)

    plot_learning(training_results, f"{save_directory}/learning_{optimization_name}.png")
