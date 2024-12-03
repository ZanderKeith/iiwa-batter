import numpy as np
import pydot
from manipulation.station import LoadScenario, MakeHardwareStation
from pydrake.all import (
    BasicVector,
    ContactResults,
    Diagram,
    DiagramBuilder,
    LeafSystem,
    Simulator,
    Value,
)

from iiwa_batter import (
    CONTACT_DT,
    NUM_JOINTS,
    PACKAGE_ROOT,
    PITCH_DT,
)
from iiwa_batter.physics import (
    PITCH_START_X,
    PITCH_START_Z,
    PLATE_OFFSET_Y,
    STRIKE_ZONE_Z,
)


def make_model_directive(dt=CONTACT_DT, model_urdf="iiwa14", add_contact=True):
    # We're pitching the ball from +x to -x
    # The robot is sitting next to the origin

    # Rotates the base towards the plate just a little so the first joint doesn't max out before crossing the strike zone
    base_rotation_deg = np.rad2deg(3 * np.pi / 4)

    if add_contact:
        contact_directive = f"""
- add_model:
    name: floor
    file: file://{PACKAGE_ROOT}/assets/floor.sdf
- add_weld:
    parent: iiwa::base
    child: floor::base
    X_PC:
        translation: [0, 0, -0.1]
- add_model:
    name: iiwa_link_0_collision
    file: file://{PACKAGE_ROOT}/assets/collision_cylinder_0.sdf
- add_weld:
    parent: iiwa::base
    child: iiwa_link_0_collision::base
    X_PC:
        translation: [0, 0, 0.075]
- add_model:
    name: iiwa_link_1_collision
    file: file://{PACKAGE_ROOT}/assets/collision_cylinder_1.sdf
- add_weld:
    parent: iiwa::iiwa_link_1
    child: iiwa_link_1_collision::base
    X_PC:
        translation: [0, 0, 0.2]
        rotation: !Rpy
            deg: [90, 0, 0]
- add_model:
    name: iiwa_link_2_collision
    file: file://{PACKAGE_ROOT}/assets/collision_cylinder_2.sdf
- add_weld:
    parent: iiwa::iiwa_link_2
    child: iiwa_link_2_collision::base
    X_PC:
        translation: [0, 0.23, 0.015]
        rotation: !Rpy
                deg: [90, 0, 0]
- add_model:
    name: iiwa_link_3_collision
    file: file://{PACKAGE_ROOT}/assets/collision_cylinder_3.sdf
- add_weld:
    parent: iiwa::iiwa_link_3
    child: iiwa_link_3_collision::base
    X_PC:
        translation: [0, 0, 0.22]
        rotation: !Rpy
            deg: [90, 0, 0]
- add_model:
    name: iiwa_link_4_collision
    file: file://{PACKAGE_ROOT}/assets/collision_cylinder_4.sdf
- add_weld:
    parent: iiwa::iiwa_link_4
    child: iiwa_link_4_collision::base
    X_PC:
        translation: [0, 0.2, 0.015]
        rotation: !Rpy
                deg: [90, 0, 0]
- add_model:
    name: iiwa_link_6_collision
    file: file://{PACKAGE_ROOT}/assets/collision_cylinder_6.sdf
- add_weld:
    parent: iiwa::iiwa_link_6
    child: iiwa_link_6_collision::base
    X_PC:
        translation: [0, 0.02, -0.005]
        rotation: !Rpy
            deg: [90, 0, 0]
"""
    else:
        contact_directive = ""

    model_directive = f"""
directives:
- add_model:
    name: iiwa
    file: file://{PACKAGE_ROOT}/assets/{model_urdf}.urdf
    default_joint_positions:
        iiwa_joint_1: [0]
        iiwa_joint_2: [0]
        iiwa_joint_3: [0]
        iiwa_joint_4: [0]
        iiwa_joint_5: [0]
        iiwa_joint_6: [0]
        iiwa_joint_7: [0]
- add_weld:
    parent: world
    child: iiwa::base
    X_PC:
        rotation: !Rpy
            deg: [0, 0, {base_rotation_deg}]
        translation: [0, {PLATE_OFFSET_Y}, 0]
{contact_directive}
- add_model:
    name: bat
    file: file://{PACKAGE_ROOT}/assets/bat.sdf
- add_weld:
    parent: iiwa::iiwa_link_ee
    child: bat::base
    X_PC:
        translation: [0.04, 0, 0.3]
- add_model:
    name: sweet_spot
    file: file://{PACKAGE_ROOT}/assets/sweet_spot.sdf
- add_weld:
    parent: bat::base
    child: sweet_spot::base
    X_PC:
        translation: [0, 0, 0.25]
- add_model:
    name: handle
    file: file://{PACKAGE_ROOT}/assets/handle.sdf
- add_weld:
    parent: bat::base
    child: handle::base
    X_PC:
        translation: [0, 0, -0.3]
- add_model:
    name: ball
    file: file://{PACKAGE_ROOT}/assets/ball.sdf
    default_free_body_pose:
        base:
            translation: [{PITCH_START_X}, 0, {PITCH_START_Z}]
- add_model:
    name: strike_zone
    file: file://{PACKAGE_ROOT}/assets/strike_zone.sdf
- add_weld:
    parent: world
    child: strike_zone::base
    X_PC:
        translation: [0, 0, {STRIKE_ZONE_Z}]

plant_config:
    contact_model: "hydroelastic"
    discrete_contact_approximation: "lagged"
    time_step: {dt}
    penetration_allowance: 8e-2
"""

    return model_directive


class TorqueTrajectorySystem(LeafSystem):
    """System that outputs a torque trajectory for the robot to follow.
    Uses rectilinear interpolation to determine the torque at each timestep.
    """

    def __init__(self, torque_trajectory: dict[float, np.ndarray]):
        LeafSystem.__init__(self)
        self.DeclareVectorOutputPort(
            "trajectory_torque", BasicVector(NUM_JOINTS), self.calculate_torque
        )
        self.dummy_input = self.DeclareVectorInputPort(
            "dummy", BasicVector(1)  # Dummy port to trigger the output
        )
        self.update_trajectory(torque_trajectory)
        self.set_name("torque_trajectory_system")

    def calculate_torque(self, context, output):
        time = context.get_time()
        # Perform rectilinear interpolation by finding the largest timestep that is less than or equal to the current time
        timestep = self.programmed_timesteps[self.programmed_timesteps <= time][-1]
        # if time in [0.0, 0.038, 0.02, 0.018000000000000002]:
        #     output.SetFromVector(np.zeros(NUM_JOINTS))
        # else:
        output.SetFromVector(self.torque_trajectory[timestep])
        #print(f"Time: {time}\nTorque: {self.torque_trajectory[timestep][0]}")

        # This is a hack to make the contact simulation run every timestep, since I want to
        # stop the expensive hydroelastic contact ASAP if possible
        self.dummy_input.Eval(context)

    def update_trajectory(self, torque_trajectory: dict[float, np.ndarray]):
        if len(torque_trajectory) == 0:
            torque_trajectory = {0: np.zeros(NUM_JOINTS)}
        self.torque_trajectory = torque_trajectory
        self.programmed_timesteps = np.array(list(torque_trajectory.keys()))


class EnforceJointLimitSystem(LeafSystem):
    """System that enforces joint limits on the robot by modifying the commanded torques.

    If the speed constraint is about to be violated, slightly reverse torques.
    If the position constraint is about to be violated, highly reverse torques.

    Also keeps track of the cumulative amount of how much the limits have been broken,
    to be used in the loss function.
    """

    def __init__(self, joint_limits: dict, limit_tolerance=0.95):
        LeafSystem.__init__(self)
        self._torque_input_port = self.DeclareVectorInputPort(
            "desired_torque", BasicVector(NUM_JOINTS)
        )
        self._state_input_port = self.DeclareVectorInputPort(
            "joint_state", BasicVector(2 * NUM_JOINTS)
        )
        self.DeclareVectorOutputPort(
            "enforced_torque", BasicVector(NUM_JOINTS), self.enforce_limits
        )
        self.DeclareVectorOutputPort(
            "cumulative_limit_break", BasicVector(1), self.get_cumulative_limit_break
        )
        self.joint_limits = joint_limits
        self.limit_tolerance = limit_tolerance # When we get to this percent of the limit, start enforcing it.
        self.fill_joint_limits(joint_limits)
        self.set_name("enforce_joint_limit_system")
        self.reset()
    
    def fill_joint_limits(self, joint_limits):
        # Turn the joint limits into a vector for easy access later
        try:
            self.joint_range_upper = np.array([joint_limits["joint_range"][str(joint + 1)][1] for joint in range(NUM_JOINTS)])
            self.joint_range_lower = np.array([joint_limits["joint_range"][str(joint + 1)][0] for joint in range(NUM_JOINTS)])
        except KeyError:
            self.joint_range_upper = np.ones(NUM_JOINTS) * 1e8
            self.joint_range_lower = np.ones(NUM_JOINTS) * -1e8
        
        try:
            self.joint_velocity_abs = np.array([joint_limits["joint_velocity"][str(joint + 1)] for joint in range(NUM_JOINTS)])
        except KeyError:
            self.joint_velocity_abs = np.ones(NUM_JOINTS) * 1e8

    def enforce_limits(self, context, output):
        desired_torque = self._torque_input_port.Eval(context)
        iiwa_state = self._state_input_port.Eval(context)
        # joint_positions = iiwa_state[:NUM_JOINTS] # Position checking is done in the urdf
        joint_velocities = iiwa_state[NUM_JOINTS:]

        velocity_overshoot = np.maximum(np.abs(joint_velocities) - self.joint_velocity_abs * self.limit_tolerance, 0)
        self.cumulative_limit_break += np.sum(velocity_overshoot)
        torque_correction = -1200 * (1 + 10*velocity_overshoot) * np.sign(joint_velocities)
        valid_torque = np.where(velocity_overshoot <= 0, desired_torque, 0)

        output_torque = np.where(velocity_overshoot > 0, valid_torque + torque_correction, desired_torque)
        
        output.SetFromVector(output_torque)

    def get_cumulative_limit_break(self, context, output):
        # This is typically ~1, so scale it up in the loss function
        output.SetFromVector(self.cumulative_limit_break)

    def reset(self):
        self.cumulative_limit_break = np.zeros(1)


class CollisionCheckSystem(LeafSystem):
    """System that checks for collisions in the robot and sets a flag with their severity if detected.
    
    Note that this doesn't actually prevent hits between the ball and the robot, but I've set the hydroelastic contacts of the robot to be so soft
    that the ball won't get hardly any distance, meaning optimization will probably find a better solution.
    """

    def __init__(self, simulator_dt, collision_threshold=3):
        """
        Parameters
        ----------
        simulator_dt : float
            The timestep of the simulator, used to scale the collision severity
        collision_threshold : int, optional
            The number of timesteps to allow before early termination, by default 3
            Decrease this to stop collisions faster, increase to get more accurate collision severity
        """
        LeafSystem.__init__(self)
        self._contact_port = self.DeclareAbstractInputPort(
            "contact_results", Value(ContactResults())
        )
        self._ball_port = self.DeclareVectorInputPort(
            "ball_state", BasicVector(13) # 7 position + 6 velocity
        )
        self.DeclareVectorOutputPort(
            "collision_severity", BasicVector(1), self.check_collision
        )
        self.simulator_dt = simulator_dt
        self.collision_threshold = collision_threshold
        self.reset()
        self.set_name("collision_check_system")

    def check_collision(self, context, output):
        if not self.initialized:
            self.initialized = True
            output.SetFromVector(self.collision_severity*CONTACT_DT/self.simulator_dt)
            return
        # This only gets run when the output is evaluated...
        contact_results = self._contact_port.Eval(context)
        num_hydroelastic_contacts = contact_results.num_hydroelastic_contacts()
        if num_hydroelastic_contacts > 0:
            # Iterate through the contacts and if it isn't a collision with the ball, report severity
            for i in range(num_hydroelastic_contacts):
                contact_info = contact_results.hydroelastic_contact_info(i)
                contact_location = contact_info.contact_surface().centroid()
                ball_location = self._ball_port.Eval(context)[4:7]
                distance = np.linalg.norm(contact_location - ball_location)
                if distance < 0.3:
                    continue
                else:
                    collision_force = contact_results.hydroelastic_contact_info(i).F_Ac_W()
                    rotational = np.linalg.norm(collision_force.rotational())
                    tranalational = np.linalg.norm(collision_force.translational())
                    self.collision_severity[0] += rotational + tranalational

        output.SetFromVector(self.collision_severity*CONTACT_DT/self.simulator_dt)

        if self.collision_severity[0] > 0:
            self.num_collision_timesteps += 1
            if self.num_collision_timesteps > self.collision_threshold:
                self.early_terminate()

    def early_terminate(self):
        # This is stupid but I couldn't figure out how else to stop the simulation when a collision is detected
        if self.terminated:
            return
        self.terminated = True
        raise EOFError("Collision detected! Terminating simulation.")

    def reset(self):
        self.collision_severity = np.zeros(1)
        self.num_collision_timesteps = 0
        self.terminated = False
        self.initialized = False


def setup_simulator(torque_trajectory, model_urdf, dt=CONTACT_DT, meshcat=None, plot_diagram=False, add_contact=True, robot_constraints=None):
    """Set up the simulator to run a swing given a torque trajectory.

    Parameters
    ----------
    torque_trajectory : dict[float, np.ndarray]
        A dictionary of timesteps and the torques to be applied at those timesteps.
    dt : float, optional
        The timestep to use for the simulation, by default CONTACT_DT
    meshcat : Meshcat, optional
        The meshcat instance to use for visualization, by default None
    plot_diagram : bool, optional
        Whether to plot the diagram, by default False

    Returns
    -------
    Simulator, Diagram
        The simulator and diagram objects (need them both later on)
    """

    builder = DiagramBuilder()
    model_directive = make_model_directive(dt, add_contact=add_contact, model_urdf=model_urdf)
    scenario = LoadScenario(data=model_directive)
    station = builder.AddSystem(MakeHardwareStation(scenario, meshcat))

    trajectory_torque_system = builder.AddSystem(
        TorqueTrajectorySystem(torque_trajectory)
    )
    if robot_constraints is None:
        builder.Connect(
            trajectory_torque_system.get_output_port(0),
            station.GetInputPort("iiwa_actuation"),
        )
    else:
        enforce_joint_limit_system = builder.AddSystem(EnforceJointLimitSystem(robot_constraints))
        builder.Connect(
            trajectory_torque_system.get_output_port(0),
            enforce_joint_limit_system.GetInputPort("desired_torque"),
        )
        builder.Connect(
            station.GetOutputPort("iiwa_state"),
            enforce_joint_limit_system.GetInputPort("joint_state"),
        )
        builder.Connect(
            enforce_joint_limit_system.GetOutputPort("enforced_torque"),
            station.GetInputPort("iiwa_actuation"),
        )

    collision_check_system = builder.AddSystem(CollisionCheckSystem(simulator_dt=dt))
    builder.Connect(
        station.GetOutputPort("contact_results"),
        collision_check_system.GetInputPort("contact_results"),
    )
    builder.Connect(
        station.GetOutputPort("ball_state"),
        collision_check_system.GetInputPort("ball_state"),
    )
    # I want the collision check to get run every timestep, so I'm connecting it to an input of a thing which gets run every timestep
    builder.Connect(
        collision_check_system.GetOutputPort("collision_severity"),
        trajectory_torque_system.GetInputPort("dummy"),
    )

    diagram = builder.Build()
    simulator = Simulator(diagram)

    if plot_diagram:
        pydot.graph_from_dot_data(diagram.GetGraphvizString(max_depth=2))[0].write_png(
            f"{PACKAGE_ROOT}/tests/figures/simulation_diagram.png"
        )

    return simulator, diagram


def reset_systems(diagram: Diagram, new_torque_trajectory=None):
    if new_torque_trajectory is not None:
        torque_trajectory_system = diagram.GetSubsystemByName(
            "torque_trajectory_system"
        )
        torque_trajectory_system.update_trajectory(new_torque_trajectory)

    collision_check_system = diagram.GetSubsystemByName("collision_check_system")
    collision_check_system.reset()
    try:
        enforce_joint_limit_system = diagram.GetSubsystemByName("enforce_joint_limit_system")
        enforce_joint_limit_system.reset()
    except RuntimeError:
        # No joint limits have been set, so system doesn't exist
        pass


def parse_simulation_state(simulator: Simulator, diagram: Diagram, system_name: str):
    """Given a simulator and diagram, parse the state of either the iiwa or the ball in the diagram.

    Parameters:
    -----------
    simulator : Simulator
    diagram : Diagram
    system_name : str
        The name of the system to parse the state of (iiwa, ball, or sweet_spot)
    """

    station: Diagram = diagram.GetSubsystemByName("station")
    plant: Diagram = station.GetSubsystemByName("plant")
    simulator_context = simulator.get_context()
    plant_context = plant.GetMyContextFromRoot(simulator_context)

    if system_name == "iiwa":
        iiwa = plant.GetModelInstanceByName("iiwa")
        joint_positions = plant.GetPositions(plant_context, iiwa)
        joint_velocities = plant.GetVelocities(plant_context, iiwa)
        return joint_positions, joint_velocities
    elif system_name == "ball":
        ball = plant.GetModelInstanceByName("ball")
        ball_position = plant.GetPositions(plant_context, ball)[4:]
        ball_velocity = plant.GetVelocities(plant_context, ball)[3:]
        return ball_position, ball_velocity
    elif system_name == "sweet_spot":
        sweet_spot = plant.GetModelInstanceByName("sweet_spot")
        sweet_spot_body = plant.GetRigidBodyByName("base", sweet_spot)
        sweet_spot_pose = plant.EvalBodyPoseInWorld(plant_context, sweet_spot_body)
        sweet_spot_position = sweet_spot_pose.translation()
        # Idk how to get the velocity of a welded object, but we don't need it for now
        return sweet_spot_position
    elif system_name == "handle":
        handle = plant.GetModelInstanceByName("handle")
        handle_body = plant.GetRigidBodyByName("base", handle)
        handle_pose = plant.EvalBodyPoseInWorld(plant_context, handle_body)
        handle_position = handle_pose.translation()
        # Idk how to get the velocity of a welded object, but we don't need it for now
        return handle_position
    elif system_name == "time":
        return simulator_context.get_time()
    elif system_name == "iiwa_actuation":
        iiwa = plant.GetModelInstanceByName("iiwa")
        return plant.GetInputPort("iiwa_actuation").Eval(plant_context)


def run_swing_simulation(
    simulator: Simulator,
    diagram: Diagram,
    start_time,
    end_time,
    initial_joint_positions,
    initial_joint_velocities,
    initial_ball_position,
    initial_ball_velocity,
    meshcat=None,
    check_dt=PITCH_DT,
    record_state=False,
) -> dict:
    """Run a swing simulation from start_time to end_time with the given initial conditions.

    Parameters
    ----------
    simulator : Simulator
        The simulator object to use for the simulation.
    diagram : Diagram
        The diagram object to use for the simulation. Expected to contain an instance of TorqueTrajectorySystem, with a torque trajectory already set.
    start_time : float
        The time to start the simulation.
    end_time : float
        The time to end the simulation. The actual final time will be at least this, but may be slightly longer to line up with the timestep.
    initial_joint_positions : np.ndarray
        The initial joint positions of the robot. MUST be of length NUM_JOINTS.
    initial_joint_velocities : np.ndarray
        The initial joint velocities of the robot. MUST be of length NUM_JOINTS.
    initial_ball_position : np.ndarray
        The initial position of the ball. MUST be of length 3.
    initial_ball_velocity : np.ndarray
        The initial velocity of the ball. MUST be of length 3.
    meshcat : Meshcat, optional
        The meshcat instance to use for visualization, by default None
    check_dt : float, optional
        The timestep to use for checking the simulation, by default PITCH_DT
    robot_constraints : dict, optional
        A dictionary of constraints to enforce on the robot, by default None

    Returns
    -------
    status_dict
        A dictionary containing the results of the simulation
    """

    if meshcat is not None:
        meshcat.Delete()

    station: Diagram = diagram.GetSubsystemByName("station")
    plant: Diagram = station.GetSubsystemByName("plant")
    collision_check_system: LeafSystem = diagram.GetSubsystemByName("collision_check_system")
    simulator_context = simulator.get_context()
    plant_context = plant.GetMyContextFromRoot(simulator_context)
    collision_check_system_context = collision_check_system.GetMyContextFromRoot(simulator_context)

    simulator.get_mutable_context().SetTime(start_time)
    simulator.Initialize()

    # Set the initial states of the iiwa and the ball
    iiwa = plant.GetModelInstanceByName("iiwa")
    ball = plant.GetModelInstanceByName("ball")
    plant.SetPositions(plant_context, iiwa, initial_joint_positions)
    plant.SetVelocities(plant_context, iiwa, initial_joint_velocities)
    plant.SetPositions(
        plant_context,
        ball,
        np.concatenate([np.ones(1), np.zeros(3), initial_ball_position]),
    )
    plant.SetVelocities(
        plant_context, ball, np.concatenate([np.zeros(3), initial_ball_velocity])
    )

    # Start time equaling end time means we don't actually want to advance the simulation,
    # just wanted to initialize the positions and velocities
    if start_time == end_time:
        return

    # Set up here so we don't need to re-evaluate every time
    sweet_spot = plant.GetModelInstanceByName("sweet_spot")
    sweet_spot_body = plant.GetRigidBodyByName("base", sweet_spot)

    # Run the pitch
    timebase = np.arange(start_time, end_time+check_dt, check_dt)
    timebase = timebase[timebase <= end_time] # This ensures all timesteps are less than or equal to the end time
    timebase[-1] = end_time # This forces the last timestep to be the end time

    if meshcat is not None:
        meshcat.StartRecording()

    result = None
    state_dict = {}
    closest_approach = np.inf
    collision_severity = 0
    for t in timebase:
        try:
            simulator.AdvanceTo(t)
            collision_severity = collision_check_system.GetOutputPort("collision_severity").Eval(collision_check_system_context)[0]
        except EOFError:
            # Collision detected, stop the simulation
            result = "collision"
            break
        
        # Determine the present position of the ball and sweet spot
        ball_position = plant.GetPositions(plant_context, ball)[4:]
        sweet_spot_pose = plant.EvalBodyPoseInWorld(plant_context, sweet_spot_body)
        sweet_spot_position = sweet_spot_pose.translation()
        distance = np.linalg.norm(ball_position - sweet_spot_position)
        if distance < closest_approach:
            closest_approach = distance
        
        if record_state:
            state_dict[t] = {
                "iiwa": parse_simulation_state(simulator, diagram, "iiwa"),
                "ball": parse_simulation_state(simulator, diagram, "ball"),
            }

    if meshcat is not None:
        meshcat.PublishRecording()

    if result is None:
        # Get final position of the ball
        ball_position, ball_velocity = parse_simulation_state(simulator, diagram, "ball")
        if ball_position[0] > -0.2:
            result = "hit"
        else:
            result = "miss"

    status_dict = {
        "result": result,
        "contact_severity": collision_severity,
        "state": state_dict,
        "closest_approach": closest_approach,
    }

    return status_dict

