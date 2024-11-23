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


def make_model_directive(dt=CONTACT_DT, model_urdf="iiwa14_limitless"):
    # We're pitching the ball from +x to -x
    # The robot is sitting next to the origin

    # Rotates the base towards the plate just a little so the first joint doesn't max out before crossing the strike zone
    base_rotation_deg = np.rad2deg(np.pi / 4)

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
        translation: [0, 0, 0.1]
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
        translation: [0, 0, 0.4]
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
        output.SetFromVector(self.torque_trajectory[timestep])

        # This is a hack to make the contact simulation run every timestep, since I want to
        # stop the expensive hydroelastic contact ASAP if possible
        self.dummy_input.Eval(context)

    def update_trajectory(self, torque_trajectory: dict[float, np.ndarray]):
        if len(torque_trajectory) == 0:
            torque_trajectory = {0: np.array([0] * NUM_JOINTS)}
        self.torque_trajectory = torque_trajectory
        self.programmed_timesteps = np.array(list(torque_trajectory.keys()))


class EnforceJointLimitSystem(LeafSystem):
    """System that enforces joint limits on the robot by modifying the commanded torques.

    If the speed constraint is about to be violated, slightly reverse torques.
    If the position constraint is about to be violated, highly reverse torques.

    Might be able to get away with doing this in the 'collision check' loop instead of its own leaf system... to be determined.
    """


class CollisionCheckSystem(LeafSystem):
    """System that checks for collisions in the robot and sets a flag with their severity if detected."""

    def __init__(self):
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

        self.reset()
        self.set_name("collision_check_system")

    def check_collision(self, context, output):
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
                if distance < 0.1:
                    continue
                else:
                    collision_force = contact_results.hydroelastic_contact_info(i).F_Ac_W()
                    rotational = np.linalg.norm(collision_force.rotational())
                    tranalational = np.linalg.norm(collision_force.translational())
                    self.collision_severity[0] += rotational + tranalational

        output.SetFromVector(self.collision_severity)

        if self.collision_severity[0] > 0:
            self.num_collision_timesteps += 1
            if self.num_collision_timesteps > 10:
                self.early_terminate()

    def early_terminate(self):
        # I couldn't figure out how else to stop my simulation when a collision is detected
        # This is cheese but it works
        raise EOFError("Collision detected! Terminating simulation.")

    def reset(self):
        self.collision_severity = np.zeros(1)
        self.num_collision_timesteps = 0


def setup_simulator(torque_trajectory, dt=CONTACT_DT, meshcat=None, plot_diagram=False):
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
    model_directive = make_model_directive(dt)
    scenario = LoadScenario(data=model_directive)
    station = builder.AddSystem(MakeHardwareStation(scenario, meshcat))

    trajectory_torque_system = builder.AddSystem(
        TorqueTrajectorySystem(torque_trajectory)
    )
    builder.Connect(
        trajectory_torque_system.get_output_port(0),
        station.GetInputPort("iiwa_actuation"),
    )

    collision_check_system = builder.AddSystem(CollisionCheckSystem())
    builder.Connect(
        station.GetOutputPort("contact_results"),
        collision_check_system.GetInputPort("contact_results"),
    )
    builder.Connect(
        station.GetOutputPort("ball_state"),
        collision_check_system.GetInputPort("ball_state"),
    )
    # I want this to get run every timestep, so I'm connecting it to an input of a thing which gets run every timestep
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


def reset_simulator(simulator: Simulator, diagram: Diagram, new_torque_trajectory=None):
    simulator_context = simulator.get_mutable_context()
    simulator_context.SetTime(0)

    if new_torque_trajectory is not None:
        torque_trajectory_system = diagram.GetSubsystemByName(
            "torque_trajectory_system"
        )
        torque_trajectory_system.update_trajectory(new_torque_trajectory)

    collision_check_system = diagram.GetSubsystemByName("collision_check_system")
    collision_check_system.reset()

    simulator.Initialize()


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
    check_dt=PITCH_DT * 100,
    robot_constraints=None,
):
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
    """

    if meshcat is not None:
        meshcat.Delete()

    station: Diagram = diagram.GetSubsystemByName("station")
    plant: Diagram = station.GetSubsystemByName("plant")
    collision_check_system: LeafSystem = diagram.GetSubsystemByName("collision_check_system")
    simulator_context = simulator.get_context()
    station_context = station.GetMyContextFromRoot(simulator_context)
    plant_context = plant.GetMyContextFromRoot(simulator_context)
    collision_check_system_context = collision_check_system.GetMyContextFromRoot(simulator_context)
    # context.SetTime(start_time) # TODO: Is this needed?
    # plant_context.SetTime(start_time) # TODO: Is this needed?
    simulator.AdvanceTo(start_time)

    # Set the initial states of the iiwa and the ball
    iiwa = plant.GetModelInstanceByName("iiwa")
    plant.SetPositions(plant_context, iiwa, initial_joint_positions)
    plant.SetVelocities(plant_context, iiwa, initial_joint_velocities)

    ball = plant.GetModelInstanceByName("ball")
    plant.SetPositions(
        plant_context,
        ball,
        np.concatenate([np.ones(1), np.zeros(3), initial_ball_position]),
    )
    plant.SetVelocities(
        plant_context, ball, np.concatenate([np.zeros(3), initial_ball_velocity])
    )

    # Set initial iiwa state
    iiwa = plant.GetModelInstanceByName("iiwa")
    plant.SetPositions(plant_context, iiwa, initial_joint_positions)
    plant.SetVelocities(plant_context, iiwa, [0] * NUM_JOINTS)

    # Run the pitch
    timebase = np.arange(start_time, end_time + check_dt, check_dt)

    if meshcat is not None:
        meshcat.StartRecording()

    strike_distance = None
    for t in timebase:
        try:
            simulator.AdvanceTo(t)
        except EOFError:
            # Read the collision severity port and stop here
            contact_results = diagram.GetOutputPort("collision_severity")
            break

        #collision_check_system.GetOutputPort("collision_severity").Eval(collision_check_system_context)
        

        # contact_results = station.GetOutputPort("contact_results")

        # Enforce joint position and velocity limits... eventually
        # joint_positions = plant.GetPositions(plant_context, iiwa)

        # Record the distance between the sweet spot of the bat and the ball at the point when the ball passes through the strike zone
        # to include in our loss function
        # if t >= time_of_flight and strike_distance is None:
        #     plant.HasModelInstanceNamed("sweet_spot")
        #     sweet_spot = plant.GetModelInstanceByName("sweet_spot")
        #     sweet_spot_body = plant.GetRigidBodyByName("base", sweet_spot)
        #     sweet_spot_pose = plant.EvalBodyPoseInWorld(plant_context, sweet_spot_body)
        #     ball_position = plant.GetPositions(plant_context, ball)[4:]
        #     strike_distance = np.linalg.norm(
        #         ball_position - sweet_spot_pose.translation()
        #     )

    if meshcat is not None:
        meshcat.PublishRecording()
