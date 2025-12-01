import genesis as gs
import open3d as o3d
from scipy.spatial.transform import Rotation
from simulation.utils.quat_utils import rotate_axis_quaternion
def get_object_bbox_and_height(object_ply, ori_axis):
    ply_o3d = o3d.io.read_triangle_mesh(object_ply)
    bbox = ply_o3d.get_axis_aligned_bounding_box()
    bbox_minbound = bbox.get_min_bound()
    bbox_maxbound = bbox.get_max_bound()
    if "x" in ori_axis:
        height = (bbox_maxbound[0] - bbox_minbound[0])
    elif "y" in ori_axis:
        height = (bbox_maxbound[1] - bbox_minbound[1])
    elif "z" in ori_axis:
        height = (bbox_maxbound[2] - bbox_minbound[2])
    else:
        raise NotImplementedError(f"unknown axis {ori_axis}")
    return bbox, height

def design_pnp_scene(scene_config, show_viewer=True):
    scene = gs.Scene(
        viewer_options = gs.options.ViewerOptions(
            camera_fov    = 60,
            max_FPS       = 30,
        ),
        vis_options = gs.options.VisOptions(
            show_world_frame = False,
            background_color=(0.8, 0.8, 0.8),
            ambient_light=(0.3, 0.3, 0.3),
        ),
        sim_options = gs.options.SimOptions(
            dt = 0.01,
            substeps=2,
        ),
        show_viewer = show_viewer,
    )
    if scene_config.robot.asset.endswith(".urdf"):
        robot = scene.add_entity(
            gs.morphs.URDF(
                file  = scene_config.robot.asset,
                pos   = (0, 0, 0),
                quat  = (1.0, 0.0, 0.0, 0.0), # we use w-x-y-z convention for quaternions,
                scale = 1.0,
            ),
            material=gs.materials.Rigid(gravity_compensation=1.0),
        )
    elif scene_config.robot.asset.endswith(".xml"):
        robot = scene.add_entity(
            gs.morphs.MJCF(
                file  = scene_config.robot.asset,
                pos   = (0, 0, 0),
                quat  = (1.0, 0.0, 0.0, 0.0), # we use w-x-y-z convention for quaternions,
                scale = 1.0,
            ),
            material=gs.materials.Rigid(gravity_compensation=1.0),
        )
    else:
        raise NotImplementedError(f"unknown robot asset type {scene_config.robot.asset}")
    # background = scene.add_entity(
    #     material=gs.materials.Rigid(rho=3000, friction=0.1),
    #     morph=gs.morphs.URDF(
    #         file  = scene_config.background.asset,
    #         pos   = (0, 0, 0),
    #         quat  = (1, 0, 0, 0), # we use w-x-y-z convention for quaternions,
    #         scale = 1.0,
    #         convexify=False,
    #         fixed=True,
    #     ),
    # )
    background = scene.add_entity(
        gs.morphs.Plane(pos   = (0, 0, -scene_config.robot.robot_to_table_height / 100)),
        material=gs.materials.Rigid(friction=0.1),
    )

    grasp_cam = scene.add_camera(
        res    = (1280, 720),
        pos    = (0.62, -0.05, 1.0),
        lookat = (0.62, -0.05, 0.0),
        fov    = 30,
        GUI    = False
    )
    desk_cam = scene.add_camera(
        res    = (1280, 720),
        pos    = (1.5, 0.0, 0.5),
        lookat = (0.62, 0.0, 0.0),
        fov    = 60,
        GUI    = False
    )
    cams = {"grasp_cam": grasp_cam, "desk_cam": desk_cam}
    object_active_bbox, object_active_height = get_object_bbox_and_height(scene_config.object_active.mesh_asset, scene_config.object_active.up_axis)
    # cm --> m
    active_pos   = (scene_config.object_active.default_position.x / 100,
                    scene_config.object_active.default_position.y / 100, 
                    object_active_height / 2 - scene_config.robot.robot_to_table_height / 100 + scene_config.object_active.height_thres / 100)
    active_quat  = rotate_axis_quaternion(scene_config.object_active.up_axis)
    object_active = scene.add_entity(
        material=gs.materials.Rigid(rho=3000, friction=0.1),
        morph=gs.morphs.URDF(
            file  = scene_config.object_active.asset,
            pos   = active_pos,
            quat  = active_quat, # we use w-x-y-z convention for quaternions,
            scale = 1.0,
            convexify=False,
            fixed=False,
        ),
        visualize_contact=False,
        # vis_mode = "collision",
    )
    object_passive_bbox, object_passive_height = get_object_bbox_and_height(scene_config.object_passive.mesh_asset, scene_config.object_passive.up_axis)
    # cm --> m
    passive_pos = (scene_config.object_passive.default_position.x / 100, 
                   scene_config.object_passive.default_position.y / 100, 
                   object_passive_height / 2 - scene_config.robot.robot_to_table_height / 100)
    passive_quat = rotate_axis_quaternion(scene_config.object_passive.up_axis)
    object_passive = scene.add_entity(
        morph=gs.morphs.URDF(
            file  = scene_config.object_passive.asset,
            pos   = passive_pos,
            quat  = passive_quat, # we use w-x-y-z convention for quaternions,
            scale = 1.0,
            convexify=False,
            fixed=True,
        ),
        visualize_contact=False,
        # vis_mode = "collision",
    )
    scene_dict = {
        "robot": robot,
        "background": background,
        "object_active": object_active,
        "object_passive": object_passive,
    }
    scene_asset_path_dict = {
        "robot": scene_config.robot.asset,
        "background": scene_config.background.asset,
        "object_active": scene_config.object_active.asset,
        "object_passive": scene_config.object_passive.asset,
    }
    default_poses = {
        "active_pos": active_pos,
        "active_quat": active_quat,
        "passive_pos": passive_pos,
        "passive_quat": passive_quat,
    }
    return scene, scene_config, scene_dict, scene_asset_path_dict, cams, default_poses