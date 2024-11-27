from iiwa_batter import PACKAGE_ROOT
from iiwa_batter.physics import BALL_MASS, BALL_RADIUS, INCHES_TO_METERS, STRIKE_ZONE_WIDTH, STRIKE_ZONE_HEIGHT


def compliant_bat(
    bat_modulus, mesh_resolution, mu_dynamic, hunt_crossley_dissipation, rigid_bat
):
    # bat_modulus = 1.2e9 / 100

    # https://en.wikipedia.org/wiki/Baseball_bat
    bat_length_inches = 34
    bat_diameter_inches = 2.6
    bat_length = bat_length_inches * INCHES_TO_METERS
    bat_radius = bat_diameter_inches * INCHES_TO_METERS / 2

    bat_mass = 0.94

    # TODO: add inertia!
    if rigid_bat:
        bat_model = "rigid_hydroelastic"
    else:
        bat_model = "compliant_hydroelastic"

    # Typical baseball bat is about
    return f"""<?xml version="1.0"?>
    <sdf version="1.7">
      <model name="bat">
        <pose>0 0 0 0 0 0</pose>
        <link name="base">
          <inertial>
            <mass>{bat_mass}</mass>
            <inertia>
              <ixx>0.016666</ixx> <ixy>0.0</ixy> <ixz>0.0</ixz>
              <iyy>0.016666</iyy> <iyz>0.0</iyz>
              <izz>0.004166</izz>
            </inertia>
          </inertial>
          <collision name="collision">
            <geometry>
              <cylinder>
                <radius>{bat_radius}</radius>
                <length>{bat_length}</length>
              </cylinder>
            </geometry>
            <drake:proximity_properties>
              <drake:{bat_model}/>
              <drake:hydroelastic_modulus>{bat_modulus}</drake:hydroelastic_modulus>
              <drake:mu_dynamic>{mu_dynamic}</drake:mu_dynamic>
              <drake:hunt_crossley_dissipation>{hunt_crossley_dissipation}</drake:hunt_crossley_dissipation>
              <drake:mesh_resolution_hint>{mesh_resolution}</drake:mesh_resolution_hint>
            </drake:proximity_properties>
          </collision>
          <visual name="visual">
            <geometry>
              <cylinder>
                <radius>{bat_radius}</radius>
                <length>{bat_length}</length>
              </cylinder>
            </geometry>
            <material>
              <diffuse>1.0 1.0 1.0 0.5</diffuse>
            </material>
          </visual>
        </link>
      </model>
    </sdf>
    """


def compliant_ball(
    ball_modulus, mesh_resolution, mu_dynamic, hunt_crossley_dissipation
):
    return f"""<?xml version="1.0"?>
    <sdf version="1.7">
      <model name="ball">
        <pose>0 0 0 0 0 0</pose>
        <link name="base">
          <inertial>
            <mass>{BALL_MASS}</mass>
            <inertia>
              <ixx>0.004166</ixx> <ixy>0.0</ixy> <ixz>0.0</ixz>
              <iyy>0.004166</iyy> <iyz>0.0</iyz>
              <izz>0.004166</izz>
            </inertia>
          </inertial>
          <collision name="collision">
            <geometry>
              <sphere>
                <radius>{BALL_RADIUS}</radius>
              </sphere>
            </geometry>
            <drake:proximity_properties>
              <drake:compliant_hydroelastic/>
              <drake:hydroelastic_modulus>{ball_modulus}</drake:hydroelastic_modulus>
              <drake:mu_dynamic>{mu_dynamic}</drake:mu_dynamic>
              <drake:hunt_crossley_dissipation>{hunt_crossley_dissipation}</drake:hunt_crossley_dissipation>
              <drake:mesh_resolution_hint>{mesh_resolution}</drake:mesh_resolution_hint>
            </drake:proximity_properties>
          </collision>
          <visual name="visual">
            <geometry>
              <sphere>
                <radius>{BALL_RADIUS}</radius>
              </sphere>
            </geometry>
            <material>
              <diffuse>1.0 1.0 1.0 0.5</diffuse>
            </material>
          </visual>
        </link>
      </model>
    </sdf>
    """


def tee():
    mesh_resolution = 0.01
    radius = 0.01
    length = 1
    modulus = 1e9
    return f"""<?xml version="1.0"?>
    <sdf version="1.7">
      <model name="tee">
        <pose>0 0 0 0 0 0</pose>
        <link name="base">
          <inertial>
            <mass>1</mass>
            <inertia>
              <ixx>0.016666</ixx> <ixy>0.0</ixy> <ixz>0.0</ixz>
              <iyy>0.014166</iyy> <iyz>0.0</iyz>
              <izz>0.004166</izz>
            </inertia>
          </inertial>
          <collision name="collision">
            <geometry>
              <cylinder>
                <radius>{radius}</radius>
                <length>{length}</length>
              </cylinder>
            </geometry>
            <drake:proximity_properties>
              <drake:compliant_hydroelastic/>
              <drake:hydroelastic_modulus>{modulus}</drake:hydroelastic_modulus>
              <drake:mu_dynamic>0.5</drake:mu_dynamic>
              <drake:hunt_crossley_dissipation>1.25</drake:hunt_crossley_dissipation>
              <drake:mesh_resolution_hint>{mesh_resolution}</drake:mesh_resolution_hint>
            </drake:proximity_properties>
          </collision>
          <visual name="visual">
            <geometry>
              <cylinder>
                <radius>{radius}</radius>
                <length>{length}</length>
              </cylinder>
            </geometry>
            <material>
              <diffuse>1.0 1.0 1.0 0.5</diffuse>
            </material>
          </visual>
        </link>
      </model>
    </sdf>
    """


def strike_zone():
    # Strike zone is 17 inches wide
    width = STRIKE_ZONE_WIDTH
    # Eyeballing the height for between joints of the robot
    height = STRIKE_ZONE_HEIGHT

    return f"""<?xml version="1.0"?>
    <sdf version="1.7">
      <model name="strike_zone">
        <pose>0 0 0 0 0 0</pose>
        <link name="base">
          <visual name="visual">
            <geometry>
              <box>
                <size>0.01 {width} {height}</size>
              </box>
            </geometry>
            <material>
              <diffuse>1.0 1.0 1.0 0.5</diffuse>
            </material>
          </visual>
        </link>
      </model>
    </sdf>
    """


def sweet_spot():
    return """<?xml version="1.0"?>
    <sdf version="1.7">
      <model name="sweet_spot">
        <pose>0 0 0 0 0 0</pose>
        <link name="base">
          <inertial>
            <mass>0.0</mass>
              <inertia>
                  <ixx>0.0</ixx> <ixy>0.0</ixy> <ixz>0.0</ixz>
                  <iyy>0.0</iyy> <iyz>0.0</iyz>
                  <izz>0.0</izz>
              </inertia>
            </inertial>
          <visual name="visual">
            <geometry>
              <sphere>
                <radius>0.05</radius>
              </sphere>
            </geometry>
            <material>
              <diffuse>1.0 1.0 1.0 0.5</diffuse>
            </material>
          </visual>
        </link>
      </model>
    </sdf>
    """

def handle():
    return """<?xml version="1.0"?>
    <sdf version="1.7">
      <model name="handle">
        <pose>0 0 0 0 0 0</pose>
        <link name="base">
          <inertial>
            <mass>0.0</mass>
              <inertia>
                  <ixx>0.0</ixx> <ixy>0.0</ixy> <ixz>0.0</ixz>
                  <iyy>0.0</iyy> <iyz>0.0</iyz>
                  <izz>0.0</izz>
              </inertia>
            </inertial>
          <visual name="visual">
            <geometry>
              <sphere>
                <radius>0.05</radius>
              </sphere>
            </geometry>
            <material>
              <diffuse>1.0 1.0 1.0 0.5</diffuse>
            </material>
          </visual>
        </link>
      </model>
    </sdf>
    """


def collision_cylinder(
    radius, length, modulus, mu_dynamic, hunt_crossley_dissipation, mesh_resolution
):
    modulus = modulus / 100000  # Softer than the bat
    mu_dynamic = mu_dynamic * 10  # More friction than the bat
    hunt_crossley_dissipation = (
        hunt_crossley_dissipation * 10
    )  # Collision significantly more dissipative
    mesh_resolution = mesh_resolution * 100  # Coarser mesh
    return f"""<?xml version="1.0"?>
<sdf version="1.7">
    <model name="collision_cylinder">
        <pose>0 0 0 0 0 0</pose>
        <link name="base">
            <inertial>
                <mass>0.0</mass>
                <inertia>
                    <ixx>0.0</ixx> <ixy>0.0</ixy> <ixz>0.0</ixz>
                    <iyy>0.0</iyy> <iyz>0.0</iyz>
                    <izz>0.0</izz>
                </inertia>
            </inertial>
            <collision name="collision">
                <geometry>
                    <cylinder>
                    <radius>{radius}</radius>
                    <length>{length}</length>
                    </cylinder>
                </geometry>
                <drake:proximity_properties>
                    <drake:compliant_hydroelastic/>
                    <drake:hydroelastic_modulus>{modulus}</drake:hydroelastic_modulus>
                    <drake:mu_dynamic>{mu_dynamic}</drake:mu_dynamic>
                    <drake:hunt_crossley_dissipation>{hunt_crossley_dissipation}</drake:hunt_crossley_dissipation>
                    <drake:mesh_resolution_hint>{mesh_resolution}</drake:mesh_resolution_hint>
                </drake:proximity_properties>
            </collision>
            <visual name="visual">
                <geometry>
                    <cylinder>
                    <radius>{radius}</radius>
                    <length>{length}</length>
                    </cylinder>
                </geometry>
                <material>
                    <diffuse>1.0 1.0 1.0 0.2</diffuse>
                </material>
            </visual>
        </link>
    </model>
</sdf>
"""


def write_assets(
    bat_modulus,
    ball_modulus,
    ball_resolution,
    bat_resolution,
    mu_dynamic=0.5,
    hunt_crossley_dissipation=1.25,
    rigid_bat=False,
):
    with open(f"{PACKAGE_ROOT}/assets/bat.sdf", "w+") as f:
        f.write(
            compliant_bat(
                bat_modulus,
                bat_resolution,
                mu_dynamic,
                hunt_crossley_dissipation,
                rigid_bat,
            )
        )

    with open(f"{PACKAGE_ROOT}/assets/ball.sdf", "w+") as f:
        f.write(
            compliant_ball(
                ball_modulus, ball_resolution, mu_dynamic, hunt_crossley_dissipation
            )
        )

    with open(f"{PACKAGE_ROOT}/assets/tee.sdf", "w+") as f:
        f.write(tee())

    with open(f"{PACKAGE_ROOT}/assets/strike_zone.sdf", "w+") as f:
        f.write(strike_zone())

    with open(f"{PACKAGE_ROOT}/assets/sweet_spot.sdf", "w+") as f:
        f.write(sweet_spot())

    with open(f"{PACKAGE_ROOT}/assets/handle.sdf", "w+") as f:
        f.write(handle())

    # This counts from 0, I don't care about collision of the end effector with anything since it's already super close to the bat
    link_radii = [0.1, 0.1, 0.08, 0.08, 0.08, 0.08, 0.07]
    link_lengths = [0.15, 0.2, 0.22, 0.18, 0.15, 0.1, 0.18]
    for i, (radius, length) in enumerate(zip(link_radii, link_lengths)):
        with open(f"{PACKAGE_ROOT}/assets/collision_cylinder_{i}.sdf", "w+") as f:
            f.write(
                collision_cylinder(
                    radius,
                    length,
                    bat_modulus,
                    mu_dynamic,
                    hunt_crossley_dissipation,
                    bat_resolution,
                )
            )

    with open(f"{PACKAGE_ROOT}/assets/floor.sdf", "w+") as f:
        f.write(
            collision_cylinder(
                2,
                0.1,
                bat_modulus,
                mu_dynamic,
                hunt_crossley_dissipation,
                bat_resolution,
            )
        )


if __name__ == "__main__":
    bat_modulus = 1.2e12
    ball_modulus = 6e9
    mu_dynamic = 0.5
    rigid_bat = False
    write_assets(bat_modulus, ball_modulus, 1e-3, 1e-3, mu_dynamic, 0.014, rigid_bat)
