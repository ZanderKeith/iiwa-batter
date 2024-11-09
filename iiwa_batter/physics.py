import numpy as np

from iiwa_batter.assets.make_assets import BALL_DRAG_COEFFICIENT, BALL_MASS, BALL_RADIUS

AIR_DENSITY_FENWAY = 1.29


def mph_to_mps(mph):
    return mph * 0.44704


def mps_to_mph(mps):
    return mps / 0.44704


def feet_to_meters(feet):
    return feet * 0.3048


def meters_to_feet(meters):
    return meters / 0.3048


def ball_flight_path(initial_position, initial_velocity, timestep=1e-2):
    g = 9.81  # gravitational acceleration (m/s^2)

    ball_cross_section = np.pi * BALL_RADIUS**2

    def ball_force(velocity):
        velocity_magnitude = np.linalg.norm(velocity)
        velocity_vector = velocity / velocity_magnitude
        drag_prefactor = (
            0.5 * AIR_DENSITY_FENWAY * BALL_DRAG_COEFFICIENT * ball_cross_section
        )
        drag_force = -drag_prefactor * (velocity_magnitude**2) * velocity_vector
        gravity_force = np.array([0, 0, -BALL_MASS * g])
        return drag_force + gravity_force

    def step(state):
        position, velocity, t = state
        new_velocity = velocity + (ball_force(velocity) / BALL_MASS) * timestep
        new_position = position + new_velocity * timestep
        return new_position, new_velocity, t + timestep

    positions = [initial_position]

    state = (np.array(initial_position), np.array(initial_velocity), 0.0)
    while state[0][2] > 0:
        state = step(state)
        positions.append(state[0])

    return np.array(positions)


def ball_distance_mph(initial_speed_mph, angle=45):
    """Just to compare with my intuition, what is the distance you get from a particular exit velocity"""
    initial_speed = mph_to_mps(initial_speed_mph)
    initial_position = [0.0, 0.0, 1.0]  # 1 meter above the ground
    initial_velocity = [
        0.0,
        initial_speed * np.cos(np.deg2rad(angle)),
        initial_speed * np.sin(np.deg2rad(angle)),
    ]
    path = ball_flight_path(initial_position, initial_velocity)
    distance = np.linalg.norm(path[-1, :2] - path[0, :2])
    distance_feet = meters_to_feet(distance)
    return distance_feet


def exit_velo_mph(states, range_check=0.5):
    """Determine the ball's speed in mph after it has traveled more than range_check meters"""

    # Find the first point where the ball has traveled more than range_check meters
    initial_position = states[0][0]
    for state in states:
        position = state[0]
        distance = np.linalg.norm(position - initial_position)
        if distance > range_check:
            break

    velocity = state[1]
    speed = np.linalg.norm(velocity)
    speed_mph = mps_to_mph(speed)

    return speed_mph


def parse_ball_state(state):
    """From the Drake ball_state output, parse the position and velocity"""

    # The ball state is a 13-element vector
    # Indices 4, 5, 6 are position
    # Indices 10, 11, 12 are velocity
    # TODO(ZanderKeith) if need be, go figure out what the others are

    position = state[4:7]
    velocity = state[10:13]

    return position, velocity


if __name__ == "__main__":
    # Example usage:
    initial_position = [0.0, 0.0, 1.0]  # 1 meter above the ground
    initial_velocity = [1.0, 1.0, 10.0]  # Initial velocity in m/s
    path = ball_flight_path(initial_position, initial_velocity)

    # Get maximum distance for several angles
    angles = np.linspace(30, 45, 20)
    distances = [ball_distance_mph(70, angle) for angle in angles]
    print(distances)
