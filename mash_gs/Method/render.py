import math
import os

import torch
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from gaussian_splatting.Method.cmd import runCMD
from gaussian_splatting.Model.gaussians import GaussianModel


def renderTrainGS(output_folder_path, port=6007):
    if not os.path.exists(output_folder_path):
        print("[ERROR][render::renderTrainGS]")
        print("\t output_folder not exist!")
        print("\t output_folder_path:", output_folder_path)
        return False

    cmd = (
        "../gaussian-splatting/gaussian_splatting/Lib/sibr_core/install/bin/SIBR_remoteGaussian_app"
        + " --port "
        + str(port)
        + " --path "
        + output_folder_path
        + " --appPath "
        + "../gaussian-splatting/gaussian_splatting/Lib/sibr_core/install/bin/"
    )

    if not runCMD(cmd, True):
        print("[ERROR][render::renderTrainGS]")
        print("\t runCMD failed!")
        print("\t cmd:", cmd)
        return False

    return True


def renderGSResult(output_folder_path, iteration=None):
    if not os.path.exists(output_folder_path):
        print("[ERROR][render::renderGSResult]")
        print("\t output_folder not exist!")
        print("\t output_folder_path:", output_folder_path)
        return False

    iteration_root_folder_path = output_folder_path + "point_cloud/"
    if not os.path.exists(iteration_root_folder_path):
        print("[ERROR][render::renderGSResult]")
        print("\t iteration_root_folder not exist! please train and wait!")
        print("\t iteration_root_folder_path:", iteration_root_folder_path)
        return False

    if iteration is None:
        iteration_idx_list = []

        iteration_folder_name_list = os.listdir(iteration_root_folder_path)

        for iteration_folder_name in iteration_folder_name_list:
            if iteration_folder_name[:10] != "iteration_":
                continue

            iteration_idx_list.append(int(iteration_folder_name[10:]))

        if len(iteration_idx_list) == 0:
            print("[ERROR][render::renderGSResult]")
            print("\t iteration_folder not found! please train and wait!")
            print("\t iteration_root_folder_path:", iteration_root_folder_path)
            return False

        iteration_idx_list.sort()

        iteration = iteration_idx_list[-1]
        iteration_folder_path = (
            iteration_root_folder_path + "iteration_" + str(iteration) + "/"
        )
    else:
        iteration_folder_path = (
            output_folder_path + "point_cloud/iteration_" + str(iteration)
        )

    if not os.path.exists(iteration_folder_path):
        print("[ERROR][render::renderGSResult]")
        print("\t iteration_folder not exist!")
        print("\t iteration_folder_path:", iteration_folder_path)
        return False

    print("[INFO][render::renderGSResult]")
    print("\t start render result...")
    print("\t data loaded from:", iteration_folder_path)

    cmd = (
        "../gaussian-splatting/gaussian_splatting/Lib/sibr_core/install/bin/SIBR_gaussianViewer_app"
        + " --model-path "
        + output_folder_path
        + " --iteration "
        + str(iteration)
        + " --appPath "
        + "../gaussian-splatting/gaussian_splatting/Lib/sibr_core/install/bin/"
    )

    if not runCMD(cmd, True):
        print("[ERROR][render::renderGSResult]")
        print("\t runCMD failed!")
        print("\t cmd:", cmd)
        return False

    return True


def render(
    viewpoint_camera,
    pc: GaussianModel,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = (
        torch.zeros_like(
            pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    scales = pc.get_scaling
    rotations = pc.get_rotation

    shs = None
    colors_precomp = None
    if override_color is None:
        shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None,
    )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
    }
