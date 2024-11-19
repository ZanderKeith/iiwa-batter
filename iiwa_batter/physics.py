import numpy as np

AIR_DENSITY_FENWAY = 1.29
g_constant = 9.81

INCHES_TO_METERS = 0.0254

# https://en.wikipedia.org/wiki/Baseball_(ball)#:~:text=A%20regulation%20baseball%20is%209,(0.142%20to%200.149%20kg).
ball_diameter_inches = 2.9
ball_diameter = ball_diameter_inches * INCHES_TO_METERS
BALL_RADIUS = ball_diameter / 2

BALL_MASS = 0.1455  # kg
BALL_DRAG_COEFFICIENT = (
    0.3  # https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/drag-on-a-baseball/
)


def mph_to_mps(mph):
    return mph * 0.44704


def mps_to_mph(mps):
    return mps / 0.44704


def feet_to_meters(feet):
    return feet * 0.3048


PLATE_OFFSET_Y = 1.2
PITCH_START_X = feet_to_meters(60.5)  # Pitcher's mound is 60.5 feet from home plate
PITCH_START_Z = feet_to_meters(5.9)
PITCH_START_POSITION = np.array([PITCH_START_X, 0, PITCH_START_Z])
STRIKE_ZONE_Z = 0.6


def meters_to_feet(meters):
    return meters / 0.3048


def ball_force(velocity):
    ball_cross_section = np.pi * BALL_RADIUS**2
    velocity_magnitude = np.linalg.norm(velocity)
    velocity_vector = velocity / velocity_magnitude
    drag_prefactor = (
        0.5 * AIR_DENSITY_FENWAY * BALL_DRAG_COEFFICIENT * ball_cross_section
    )
    drag_force = -drag_prefactor * (velocity_magnitude**2) * velocity_vector
    gravity_force = np.array([0, 0, -BALL_MASS * g_constant])
    return drag_force + gravity_force


def step(state, timestep):
    position, velocity, t = state
    new_velocity = velocity + (ball_force(velocity) / BALL_MASS) * timestep
    new_position = position + new_velocity * timestep
    return new_position, new_velocity, t + timestep


def ball_flight_path(initial_position, initial_velocity, timestep=1e-2):
    positions = [initial_position]

    state = (np.array(initial_position), np.array(initial_velocity), 0.0)
    while state[0][2] > 0:
        state = step(state, timestep)
        positions.append(state[0])

    return np.array(positions)


def find_initial_velocity(
    ball_horizontal_speed_mph,
    ball_target_position: np.array,
    ball_initial_position=PITCH_START_POSITION,
):
    """Given the initial position of the ball, find the initial velocity that will get the ball
    to a target position in space when pitched at a particular speed, and also find the time it takes to get .
    This accounts for the effects of gravity but not air resistance.

    Parameters:
    ----------
    ball_horizontal_speed_mph (float): Initial speed of the ball in miles per hour.
    ball_target_position (np.array): Target position of the ball [x, y, z] in meters.
    ball_initial_position (np.array): Initial position of the ball [x, y, z] in meters.

    Returns:
    tuple: (initial_velocity_vector (np.array), flight_time (float))
    """

    # Convert speed to m/s
    ball_initial_speed = mph_to_mps(ball_horizontal_speed_mph)

    # Calculate displacement vector
    displacement = ball_target_position - ball_initial_position

    # Horizontal distance and time of flight
    horizontal_distance = np.linalg.norm(displacement[:2])
    time_of_flight = horizontal_distance / ball_initial_speed

    # Determine the velocity change from gravity
    gravity_velocity = -g_constant * time_of_flight

    # Launch the ball with half the gravity velocity upwards so it cancels by the time the ball reaches the target
    vx = ball_initial_speed * displacement[0] / horizontal_distance
    vy = ball_initial_speed * displacement[1] / horizontal_distance
    vz = (
        ball_initial_speed * displacement[2] / horizontal_distance
        - 0.5 * gravity_velocity
    )

    # Construct the velocity vector
    initial_velocity_vector = np.array([vx, vy, vz])

    return initial_velocity_vector, time_of_flight


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
