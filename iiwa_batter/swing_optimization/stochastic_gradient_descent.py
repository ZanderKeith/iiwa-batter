def reward(simulator, station):
    # Calculate reward
    # If ball position is negative, we missed the ball and should penalize
    # Otherwise, return reward based on the distance the ball travels

    context = simulator.get_context()
    plant = station.GetSubsystemByName("plant")
    plant_context = plant.GetMyContextFromRoot(context)
    ball = plant.GetModelInstanceByName("ball")

    ball_position = plant.GetPositions(plant_context, ball)[4:]

    if ball_position[0] < 0:
        reward = -10 * strike_distance
    else:
        # Determine distance ball travels
        ball_velocity = plant.GetVelocities(plant_context, ball)[3:]
        path = ball_flight_path(ball_position, ball_velocity)
        land_location = path[-1, :2]
        distance = np.linalg.norm(land_location)  # Absolute distance from origin
        # If the ball is traveling backwards, reward is negative distance
        if land_location[0] < 0:
            reward = -distance
        # If the ball is foul (angle > +/- 45 degrees), reward is half the distance
        elif np.abs(np.arctan(land_location[1] / land_location[0])) > np.pi / 4:
            reward = distance / 2
        # Otherwise, return the distance
        else:
            reward = distance

    return reward
