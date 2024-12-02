import numpy as np

from pydrake.all import (
    Diagram,
    Simulator,
)
from iiwa_batter import (
    NUM_JOINTS,
)
from iiwa_batter.physics import (
    FLIGHT_TIME_MULTIPLE,
    ball_flight_path,
    ball_distance_multiplier,
)
from iiwa_batter.swing_simulator import (
    parse_simulation_state,
    reset_systems,
    run_swing_simulation,
)
from iiwa_batter.swing_optimization.stochastic_gradient_descent import (
    make_torque_trajectory,
    perturb_vector,
    descent_step,
)


def partial_trajectory_reward(
    simulator: Simulator,
    diagram: Diagram,
    start_time,
    initial_joint_positions,
    initial_joint_velocities,
    initial_ball_position,
    initial_ball_velocity,
    torque_trajectory,
    meshcat=None,
):
    reset_systems(diagram, torque_trajectory)
    status_dict = run_swing_simulation(
        simulator,
        diagram,
        start_time=start_time,
        end_time=max(torque_trajectory.keys()),
        initial_joint_positions=initial_joint_positions,
        initial_joint_velocities=initial_joint_velocities,
        initial_ball_position=initial_ball_position,
        initial_ball_velocity=initial_ball_velocity,
        meshcat=meshcat
    )

    result = status_dict["result"]
    if result == "collision":
        severity = status_dict["contact_severity"]
        if severity <= 10:
            return ((-1 * severity) - 1) / 10
        else:
            return ((-10 * np.log10(severity)) - 1) / 10
    elif result == "miss":
        return -10 * status_dict["closest_approach"]
    elif result == "hit":
        ball_position, ball_velocity = parse_simulation_state(simulator, diagram, "ball")
        path = ball_flight_path(ball_position, ball_velocity)
        land_location = path[-1, :2]
        distance = np.linalg.norm(land_location)
        multiplier = ball_distance_multiplier(land_location)
        return distance * multiplier * 0.01 # Keep the reward in a reasonable range
    else:
        raise ValueError(f"Unknown result: {result}")
