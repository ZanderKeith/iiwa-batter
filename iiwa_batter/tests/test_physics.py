import numpy as np

from iiwa_batter.physics import (
    ball_flight_path,
    ball_distance_mph,
    ball_distance_multiplier
)

def test_ball_distance_multiplier():

    fair_balls = [
        [1, 0],
        [10, 0],
        [50, 20],
        [50, -20]
    ]
    for ball in fair_balls:
        assert ball_distance_multiplier(ball) == 1

    backwards_balls = [
        [-1, 0],
        [-10, 0],
        [-50, 20],
        [-50, -20],
        [-1, -90],
    ]
    for ball in backwards_balls:
        assert ball_distance_multiplier(ball) == -1

    foul_balls = [
        [1, -2],
        [1, 2],
        [49, 50],
        [49, -50],
    ]
    for ball in foul_balls:
        assert ball_distance_multiplier(ball) == 0.5

    long_balls = [
        [100, 10],
        [100, -10],
    ]
    for ball in long_balls:
        assert ball_distance_multiplier(ball) == 1.2

def test_ball_path():
    # With a given initial position and velocity, ensure the path of the ball makes sense
    initial_position = [0, 0, 1]
    initial_velocity = [10, 0, 10]

    path = ball_flight_path(initial_position, initial_velocity)

    final_position = path[-1]

    assert final_position[0] > 10
    assert final_position[1] == 0
    assert final_position[2] < 1

def test_ball_distance_mph():
    # Ensure the distance calculation is close to intuition
    distance_feet = ball_distance_mph(110, 40)
    assert distance_feet > 400
    assert distance_feet < 430