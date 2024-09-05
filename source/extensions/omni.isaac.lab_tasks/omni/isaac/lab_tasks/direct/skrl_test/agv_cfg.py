import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets import ArticulationCfg


class AGV_JOINT:
    MB_LW_REV = "jlw"
    MB_RW_REV = "jrw"
    MB_PZ_PRI = "jz"
    PZ_PY_PRI = "jy"
    PY_PX_PRI = "jx"
    PX_PR_REV = "jr"
    PR_LR_REV = "jlr"
    PR_RR_REV = "jrr"
    LR_LPIN_PRI = "jlpin"
    RR_RPIN_PRI = "jrpin"


AGV_CFG: ArticulationCfg = ArticulationCfg(
    # prim_path="{ENV_REGEX_NS}/AGV",
    spawn=sim_utils.UsdFileCfg(
        usd_path="./robot/usd/agv/agv_fixed_pin.usd",
        # rigid_props=sim_utils.RigidBodyPropertiesCfg(
        #     rigid_body_enabled=True,
        #     max_linear_velocity=1000.0,
        #     max_angular_velocity=1000.0,
        #     max_depenetration_velocity=100.0,
        #     enable_gyroscopic_forces=True,
        # ),
        # articulation_props=sim_utils.ArticulationRootPropertiesCfg(
        #     enabled_self_collisions=True,
        #     solver_position_iteration_count=4,
        #     solver_velocity_iteration_count=0,
        #     sleep_threshold=0.005,
        #     stabilization_threshold=0.001,
        # ),
        activate_contact_sensors=True,
    ),
    # init_state=ArticulationCfg.InitialStateCfg(
    #     pos=(0.0, 0.0, 0.0),
    #     joint_pos={
    #         AGV_JOINT.MB_LW_REV: 0.0,
    #         AGV_JOINT.MB_RW_REV: 0.0,
    #         AGV_JOINT.MB_PZ_PRI: 0.0,
    #         AGV_JOINT.PZ_PY_PRI: 0.0,
    #         AGV_JOINT.PY_PX_PRI: 0.0,
    #         AGV_JOINT.PX_PR_REV: 0.0,
    #         AGV_JOINT.PR_LR_REV: 0.0,
    #         AGV_JOINT.PR_RR_REV: 0.0,
    #         AGV_JOINT.LR_LPIN_PRI: 0.0,
    #         AGV_JOINT.RR_RPIN_PRI: 0.0,
    #     },
    # ),
    actuators={
        "wheel_actuator": ImplicitActuatorCfg(
            joint_names_expr=[AGV_JOINT.MB_LW_REV, AGV_JOINT.MB_RW_REV],
            effort_limit=200.0,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=0.0,
        ),
        "xyz_actuator": ImplicitActuatorCfg(
            joint_names_expr=[
                # AGV_JOINT.MB_PZ_PRI,
                AGV_JOINT.PZ_PY_PRI,
                AGV_JOINT.PY_PX_PRI,
            ],
            effort_limit=300.0,
            velocity_limit=100.0,
            stiffness=1000.0,
            damping=1000.0,
        ),
        "px_pr_rev_actuator": ImplicitActuatorCfg(
            joint_names_expr=[AGV_JOINT.PX_PR_REV],
            effort_limit=100.0,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=0.0,
        ),
        "pin_rev_actuator": ImplicitActuatorCfg(
            joint_names_expr=[AGV_JOINT.PR_LR_REV, AGV_JOINT.PR_RR_REV],
            effort_limit=200.0,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=0.0,
        ),
        "pin_pri_actuator": ImplicitActuatorCfg(
            joint_names_expr=[
                # AGV_JOINT.LR_LPIN_PRI,
                AGV_JOINT.RR_RPIN_PRI
            ],
            effort_limit=100.0,
            velocity_limit=50.0,
            stiffness=100.0,
            damping=100.0,
        ),
    },
)
