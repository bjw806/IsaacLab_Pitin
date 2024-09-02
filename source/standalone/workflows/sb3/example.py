import omni.isaac.lab.sim as sim_utils
# from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR

# print(ISAACLAB_NUCLEUS_DIR)

spawn=sim_utils.UsdFileCfg(
    usd_path="./robot/1115_urdf v13.usdz",
    rigid_props=sim_utils.RigidBodyPropertiesCfg(
        rigid_body_enabled=True,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=100.0,
        enable_gyroscopic_forces=True,
    ),
    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=False,
        solver_position_iteration_count=4,
        solver_velocity_iteration_count=0,
        sleep_threshold=0.005,
        stabilization_threshold=0.001,
    ),
),


