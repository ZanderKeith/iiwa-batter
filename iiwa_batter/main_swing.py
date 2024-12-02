import os

import numpy as np

from iiwa_batter import (
    PACKAGE_ROOT,
    CONTACT_DT,
)
from iiwa_batter.swing_optimization.graduate_student_descent import (
    COARSE_LINK,
)
from iiwa_batter.trajectory_library import (
    STATE,
    LIBRARY_SPEEDS_MPH,
    LIBRARY_POSITIONS,
    Trajectory,
)
from iiwa_batter.naive_full_trajectory_optimization import (
    run_naive_full_trajectory_optimization_hot_start,
    run_naive_full_trajectory_optimization_hot_start_torque_only,
)

if STATE == "FINAL":
    ITERATIONS = 100
if STATE == "LEARNING_RATE_TUNING":
    ITERATIONS = 10
if STATE == "TEST":
    ITERATIONS = 1

LEARNING_RATE = 10
POSITION_VARIANCE = np.deg2rad(1)/10
TORQUE_VARIANCE = 1/10

def optimize_main_swing(robot):
    save_directory = f"{PACKAGE_ROOT}/../trajectories/{robot}/main"
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    main_target_speed_mph = LIBRARY_SPEEDS_MPH[0]
    main_target_position = LIBRARY_POSITIONS[0]
    optimization_name = f"{main_target_speed_mph}"

    if not os.path.exists(f"{save_directory}/{optimization_name}.dill"):
        # Start from the student's best guess
        best_initial_position = COARSE_LINK[robot]["initial_position"]
        best_control_vector = COARSE_LINK[robot]["control_vector"][main_target_speed_mph]
        best_reward = -np.inf
        print(f"Fine tuning student's trajectory for {robot} at {main_target_speed_mph} over {ITERATIONS} iterations")
    else:
        # Improve from where was left off previously
        trajectory = Trajectory(robot, main_target_speed_mph, main_target_position, "main")
        results = trajectory.load_training_results()
        best_initial_position = results["best_initial_position"]
        best_control_vector = results["best_control_vector"]
        best_reward = results["final_best_reward"]
        print(f"Improving trajectory from previous optimization for {robot} at {main_target_speed_mph} with reward {best_reward} over {ITERATIONS} iterations")

    run_naive_full_trajectory_optimization_hot_start(
        robot=robot,
        target_velocity_mph=main_target_speed_mph,
        target_position=main_target_position,
        optimization_name=optimization_name,
        save_directory=save_directory,
        present_initial_position=best_initial_position,
        present_control_vector=best_control_vector,
        simulation_dt=CONTACT_DT,
        iterations=ITERATIONS,
        learning_rate=LEARNING_RATE,
        position_variance=POSITION_VARIANCE,
        torque_variance=TORQUE_VARIANCE,
    )

    new_trajectory = Trajectory(robot, main_target_speed_mph, main_target_position, "main")
    new_results = new_trajectory.load_training_results()
    new_best_reward = new_results["final_best_reward"]
    print(f"{robot} best reward from main swing: {new_best_reward}")
    if new_best_reward < best_reward:
        print(f"Reward {new_best_reward} < {best_reward}, reverting to previous optimization")
        trajectory.save_training_results(results)


def transfer_main_swing(robot, target_speed_mph):
    # Now that we have the best initial position for the fastest pitch, transfer it to the other pitch speeds
    save_directory = f"{PACKAGE_ROOT}/../trajectories/{robot}/transfer"
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    main_target_position = LIBRARY_POSITIONS[0]
    optimization_name = f"{target_speed_mph}"

    if not os.path.exists(f"{save_directory}/{optimization_name}.dill"):
        # Start from the student's best guess
        best_control_vector = COARSE_LINK[robot]["control_vector"][target_speed_mph]
        best_reward = -np.inf
        main_trajectory = Trajectory(robot, LIBRARY_SPEEDS_MPH[0], main_target_position, "main")
        main_trajectory_results = main_trajectory.load_training_results()
        initial_position = main_trajectory_results["best_initial_position"]
        print(f"Fine tuning student's trajectory for {robot} at {target_speed_mph} over {ITERATIONS} iterations")
    else:
        # Improve from where was left off previously
        trajectory = Trajectory(robot, target_speed_mph, main_target_position, "transfer")
        results = trajectory.load_training_results()
        best_control_vector = results["best_control_vector"]
        initial_position = results["best_initial_position"]
        best_reward = results["final_best_reward"]
        print(f"Improving trajectory from previous optimization for {robot} at {target_speed_mph} with reward {best_reward} over {ITERATIONS} iterations")

    run_naive_full_trajectory_optimization_hot_start_torque_only(
        robot=robot,
        target_velocity_mph=target_speed_mph,
        target_position=main_target_position,
        optimization_name=optimization_name,
        save_directory=save_directory,
        initial_joint_positions=initial_position,
        present_control_vector=best_control_vector,
        simulation_dt=CONTACT_DT,
        iterations=ITERATIONS,
        learning_rate=LEARNING_RATE,
    )

    new_trajectory = Trajectory(robot, target_speed_mph, main_target_position, "transfer")
    new_results = new_trajectory.load_training_results()
    new_best_reward = new_results["final_best_reward"]
    print(f"{robot} best reward from main swing: {new_best_reward}")

    if new_best_reward < best_reward:
        print(f"Reward {new_best_reward} < {best_reward}, reverting to previous optimization")
        trajectory.save_training_results(results)

    return

if __name__ == "__main__":
    main_swing_robots = ["slugger"]
    for robot in main_swing_robots:
        optimize_main_swing(robot)
    # for robot in transfer_robots:
    #     for speed in LIBRARY_SPEEDS_MPH[1:]:
    #         transfer_main_swing(robot, speed)


"""
Old stuff when I was trying to do a fully sample-based optimization.
Turns out that is running into the whole "parameter space is huge" problem
Instead, making a more informed guess at the initial position and control vector and optimizing from there.

# def main_swing_impact_optimization(robot, target_speed_mph, target_position, plate_time, plate_ball_position, plate_ball_velocity, plate_joint_position, plate_joint_velocity, plate_position_index):
#     save_directory = f"{PACKAGE_ROOT}/../trajectories/{robot}/{target_speed_mph}_{target_position}"
#     optimization_name = f"impact_{plate_position_index}"
#     if not os.path.exists(f"{save_directory}/{optimization_name}.dill"):
#         run_swing_impact_optimization(
#             robot=robot,
#             optimization_name=optimization_name,
#             save_directory=save_directory,
#             plate_time=plate_time,
#             plate_ball_position=plate_ball_position,
#             plate_ball_velocity=plate_ball_velocity,
#             present_joint_positions=plate_joint_position,
#             present_joint_velocities=plate_joint_velocity,
#             learning_rate=MAIN_INITIAL_LEARNING_RATE,
#             simulation_dt=CONTACT_DT,
#             iterations=MAIN_IMPACT_ITERATIONS,
#         )

#     trajectory = Trajectory(robot, target_speed_mph, target_position, optimization_name)
#     results = trajectory.load_training_results()

#     return results["final_best_reward"], plate_position_index

# def main_coarse_link_optimization(robot, target_speed_mph, target_position, plate_time, searched_initial_position, swing_impact_index, initial_position_index):
#     save_directory = f"{PACKAGE_ROOT}/../trajectories/{robot}/{target_speed_mph}_{target_position}"
#     optimization_name = f"coarse_link_impact{swing_impact_index}_pos{initial_position_index}"
#     if not os.path.exists(f"{save_directory}/{optimization_name}.dill"):
#         if initial_position_index == 0:
#             searched_control_vector = COARSE_LINK[robot]["control_vector"]
#         else:
#             robot_constraints = JOINT_CONSTRAINTS[robot]
#             _, ball_time_of_flight = find_ball_initial_velocity(target_speed_mph, target_position)
#             searched_control_vector = initialize_control_vector(robot_constraints, ball_time_of_flight)
    
#         impact_trajectory = Trajectory(robot, target_speed_mph, target_position, f"impact_{swing_impact_index}")
#         impact_results = impact_trajectory.load_training_results()

#         # We don't care about linking to bad impacts for the final optimization
#         if impact_results["final_best_reward"] < 0 and STATE == "FINAL":
#             return -np.inf, swing_impact_index, initial_position_index
        
#         plate_joint_positions = impact_results["best_joint_positions"]
#         plate_joint_velocities = impact_results["best_joint_velocities"]
#         ball_time_of_flight = find_ball_initial_velocity(target_speed_mph, target_position)[1]

#         run_swing_link_optimization(
#             robot=robot,
#             optimization_name=optimization_name,
#             save_directory=save_directory,
#             ball_time_of_flight=ball_time_of_flight,
#             plate_time=plate_time,
#             plate_joint_positions=plate_joint_positions,
#             plate_joint_velocities=plate_joint_velocities,
#             present_initial_position=searched_initial_position,
#             present_control_vector=searched_control_vector,
#             learning_rate=COARSE_LINK_LEARNING_RATE,
#             iterations=COARSE_LINK_ITERATIONS,
#         )

#     trajectory = Trajectory(robot, target_speed_mph, target_position, optimization_name)
#     results = trajectory.load_training_results()

#     return results["final_best_reward"], swing_impact_index, initial_position_index

    # 1. Find the optimal swing for the main target.
    print("Finding swing impact for main target")
    plate_time, plate_ball_position, plate_ball_velocity = calculate_plate_time_and_ball_state(main_target_speed_mph, main_target_position)
    main_pool = Pool(NUM_PROCESSES)
    main_answers = []
    if not os.path.exists(f"{save_directory}/impact_0.dill"):
        generated_plate_positions = find_initial_positions(robot, NUM_INITIAL_POSITIONS-1, bounding_box=SWING_IMPACT_BOUNDING_BOX)
        searched_plate_positions = [SWING_IMPACT[robot]["plate_position"]] + generated_plate_positions
        robot_constraints = JOINT_CONSTRAINTS[robot]
        velocity_constraints_abs = np.array([velocity for velocity in robot_constraints["joint_velocity"].values()])*VELOCITY_CAP_FRACTION
        generated_plate_velocities = [np.random.uniform(-velocity_constraints_abs, velocity_constraints_abs, NUM_JOINTS) for position in range(NUM_INITIAL_POSITIONS-1)]
        searched_plate_velocities = [SWING_IMPACT[robot]["plate_velocity"]] + generated_plate_velocities
        main_results = main_pool.starmap(main_swing_impact_optimization, [(robot, main_target_speed_mph, main_target_position, plate_time, plate_ball_position, plate_ball_velocity, searched_plate_positions[i], searched_plate_velocities[i], i) for i in range(NUM_INITIAL_POSITIONS)])
    else:
        # Already did this, just go load it
        searched_plate_positions = [np.zeros(NUM_JOINTS) for i in range(NUM_INITIAL_POSITIONS)]
        searched_plate_velocities = [np.zeros(NUM_JOINTS) for i in range(NUM_INITIAL_POSITIONS)]
        main_results = main_pool.starmap(main_swing_impact_optimization, [(robot, main_target_speed_mph, main_target_position, plate_time, plate_ball_position, plate_ball_velocity, searched_plate_positions[i], searched_plate_velocities[i], i) for i in range(NUM_INITIAL_POSITIONS)])

    for result in main_results:
        main_answers.append(result)

    # 2. Pick the best swing impact to optimize further
    best_index_reward = -np.inf
    best_impact_index = None
    for answer in main_answers:
        if answer[0] > best_index_reward:
            best_index_reward = answer[0]
            best_impact_index = answer[1]
    if best_impact_index is None:
        # Welp, we're testing stuff
        best_impact_index = 0
    print("Finding trajectory to reach main target")
    generated_initial_positions = find_initial_positions(robot, NUM_INITIAL_POSITIONS-1)
    searched_initial_positions = [COARSE_LINK[robot]["initial_position"]] + generated_initial_positions
    main_results = main_pool.starmap(main_coarse_link_optimization, [(robot, main_target_speed_mph, main_target_position, plate_time, searched_initial_positions[j], best_impact_index, j) for j in range(NUM_INITIAL_POSITIONS)])
    main_answers = []
    for result in main_results:
        main_answers.append(result)
    main_pool.close()

    best_index_reward = -np.inf
    best_link_index = None
    for answer in main_answers:
        if answer[0] > best_index_reward:
            best_index_reward = answer[0]
            best_link_index = answer[2]
    if best_link_index is None:
        best_link_index = 0

"""