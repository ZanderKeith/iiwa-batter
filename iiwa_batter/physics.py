import numpy as np

from iiwa_batter.assets.make_assets import BALL_MASS, BALL_RADIUS, BALL_DRAG_COEFFICIENT

AIR_DENSITY_FENWAY = 1.29

def ball_flight_path(initial_position, initial_velocity, timestep=1e-2):
    g = 9.81  # gravitational acceleration (m/s^2)

    ball_cross_section = np.pi * BALL_RADIUS**2

    def ball_force(velocity):
        velocity_magnitude = np.linalg.norm(velocity)
        velocity_vector = velocity / velocity_magnitude
        drag_prefactor = 0.5 * AIR_DENSITY_FENWAY * BALL_DRAG_COEFFICIENT * ball_cross_section
        drag_force = -drag_prefactor * (velocity_magnitude**2) * velocity_vector
        gravity_force = np.array([0, 0, -BALL_MASS * g])
        return drag_force + gravity_force

    def step(state):
        position, velocity, t = state
        new_velocity = velocity + (ball_force(velocity)/BALL_MASS) * timestep
        new_position = position + new_velocity * timestep
        return new_position, new_velocity, t + timestep

    positions = [initial_position]

    state = (np.array(initial_position), np.array(initial_velocity), 0.0)
    while state[0][2] > 0:
        state = step(state)
        positions.append(state[0])

    return np.array(positions)

def exit_velo_angle(initial_speed_mph, angle=45):
    """Just to compare with my intuition, what is the distance you get from a particular exit velocity"""
    initial_speed = initial_speed_mph * 0.44704  # convert to m/s
    initial_position = [0.0, 0.0, 1.0]  # 1 meter above the ground
    initial_velocity = [0.0, initial_speed * np.cos(np.deg2rad(angle)), initial_speed * np.sin(np.deg2rad(angle))]
    path = ball_flight_path(initial_position, initial_velocity)
    distance = np.linalg.norm(path[-1, :2] - path[0, :2])
    distance_feet = distance * 3.28084
    return distance_feet

# Example usage:
initial_position = [0.0, 0.0, 1.0]  # 1 meter above the ground
initial_velocity = [1.0, 1.0, 10.0]  # Initial velocity in m/s
path = ball_flight_path(initial_position, initial_velocity)

# Get maximum distance for several angles
angles = np.linspace(30, 45, 20)
distances = [exit_velo_angle(101, angle) for angle in angles]
print(distances)