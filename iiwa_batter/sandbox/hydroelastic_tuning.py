import numpy as np

from pydrake.all import (
    AddDefaultVisualization,
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    LoadModelDirectives,
    LoadModelDirectivesFromString,
    Parser,
    ProcessModelDirectives,
    RigidTransform,
    RollPitchYaw,
    Simulator,
    SpatialForce,
)

from manipulation.station import LoadScenario, MakeHardwareStation
from manipulation.utils import ConfigureParser, RenderDiagram

from iiwa_batter import PACKAGE_ROOT
from iiwa_batter.physics import parse_ball_state, exit_velo_mph

# Throw the ball at a stationary bat and ensure the measured coefficient of restitution is about correct

def make_model_directive(ball_pos, dt):
    model_directive = f"""
directives: 
- add_model:
    name: bat
    file: file://{PACKAGE_ROOT}/assets/bat.sdf
    default_free_body_pose:
        base:
            translation: [0, 0, 0]
- add_weld:
    parent: world
    child: bat::base
    X_PC:
        translation: [0, 0, 0]
- add_model:
    name: ball
    file: file://{PACKAGE_ROOT}/assets/ball.sdf
    default_free_body_pose:
        base:
            translation: [{ball_pos[0]}, {ball_pos[1]}, {ball_pos[2]}]

plant_config:
    contact_model: "hydroelastic"
    discrete_contact_approximation: "lagged"
    time_step: {dt}
    penetration_allowance: 8e-2
"""
    
    return model_directive
    
def run_hydroelastic_tuning(meshcat, ball_pos, ball_velocity_x, record_time=1.0, dt=1e-2, debug_plot=False):

    if meshcat is not None:
        meshcat.Delete()

    builder = DiagramBuilder()

    model_directive = make_model_directive(ball_pos, dt)

    scenario = LoadScenario(data=model_directive)
    station = builder.AddSystem(MakeHardwareStation(scenario, meshcat))
    diagram = builder.Build()
    simulator = Simulator(diagram)
    context = simulator.get_context()

    # Set initial velocity of the ball
    plant = station.GetSubsystemByName("plant")
    plant_context = plant.GetMyContextFromRoot(context)
    ball = plant.GetModelInstanceByName("ball")
    plant.SetVelocities(plant_context, ball, np.array([0, 0, 0] + [ball_velocity_x] + [0, 0]))

    if debug_plot:
        import pydot
        RenderDiagram(station, max_depth=1)
        pydot.graph_from_dot_data(station.GetGraphvizString(max_depth=1))[0].write_png('station_render.png')

    if meshcat is not None:
        meshcat.StartRecording()

    times = np.linspace(0, record_time, int(record_time/dt))
    ball_x_positions = np.zeros_like(times)
    ball_x_velocities = np.zeros_like(times)
    for i, time in enumerate(times):
        # if time == 0:
        #     # Make a spatial force to apply to the ball
        #     spatial_force = SpatialForce([0, 0, 0], [0, 0, 0])
        #     station.GetInputPort("applied_spatial_force").FixValue(station_context, spatial_force)
        simulator.AdvanceTo(time)
        #print(plant.GetVelocities(plant_context, ball))
        ball_x_positions[i] = plant.GetPositions(plant_context, ball)[4]
        ball_x_velocities[i] = plant.GetVelocities(plant_context, ball)[3]

    if meshcat is not None:
        meshcat.PublishRecording()

    final_velocity_x = plant.GetVelocities(plant_context, ball)[3]

    min_x = min(ball_x_positions)
    print(f"Minimum x position: {min_x}")
    print(f"X position error: {min_x-0.06799999999999917}")

    return calculate_coefficient_of_restitution(ball_velocity_x, final_velocity_x), ball_x_positions, ball_x_velocities, times


def calculate_coefficient_of_restitution(initial_velocity, exit_velocity):
    # yeah yeah this is a simple function whatever 
    # https://en.wikipedia.org/wiki/Coefficient_of_restitution
    return abs(exit_velocity / initial_velocity)