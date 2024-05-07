import torch
import numpy as np
from torch import nn

from ma_sh.Model.mash import Mash
from camera_manage.Config.cameras import BasicPointCloud

from mash_gs.Method.color import RGB2SH
from mash_gs.Model.gaussians import GaussianModel
from mash_gs.Method.model import inverse_sigmoid
from simple_knn._C import distCUDA2



class MashGS(GaussianModel):
    def __init__(self, sh_degree: int, anchor_num: int=20,
        mask_degree_max: int = 3,
        sh_degree_max: int = 2,
        mask_boundary_sample_num: int = 36,
        sample_polar_num: int = 1000,
        sample_point_scale: float = 0.8,
        use_inv: bool = True,
        idx_dtype=torch.int64,
        dtype=torch.float32,
        device: str = "cpu"):
        super().__init__(sh_degree)

        self.mash = Mash(anchor_num, mask_degree_max, sh_degree_max, mask_boundary_sample_num,
                         sample_polar_num, sample_point_scale, use_inv, idx_dtype, dtype, device)
        self.mash.setGradState(True)

        # fixed gs position on boundary
        self.boundary_phis = self.mash.mask_boundary_phis
        self.boundary_theta_weights = torch.ones_like(self.mash.mask_boundary_phis)
        self.boundary_idxs = self.mash.mask_boundary_phi_idxs

        # active gs position on inner surface
        self.inner_phis = torch.tensor([0.0], dtype=dtype).to(device)
        self.inner_theta_weights = torch.tensor([0.0], dtype=dtype).to(device)
        self.inner_idxs = torch.tensor([0.0], dtype=dtype).to(device)
        return

    def initGSParams(self) -> bool:
        mask_boundary_thetas = self.mash.toMaskBoundaryThetas()
        self.inner_phis, self.inner_theta_weights, self.inner_idxs = self.mash.toInMaskSamplePolars(mask_boundary_thetas)[:3]
        self.inner_phis.requires_grad_(True)
        self.inner_theta_weights.requires_grad_(True)
        return True

    def capture(self):
        return (
            self.mash.anchor_num,
            self.mash.mask_degree_max,
            self.mash.sh_degree_max,
            self.mash.mask_boundary_sample_num,
            self.mash.mask_params,
            self.mash.sh_params,
            self.mash.rotate_vectors,
            self.mash.positions,
            self.boundary_phis,
            self.boundary_theta_weights,
            self.boundary_idxs,
            self.inner_phis,
            self.inner_theta_weights,
            self.inner_idxs,

            self.active_sh_degree,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (
            self.mash.anchor_num,
            self.mash.mask_degree_max,
            self.mash.sh_degree_max,
            self.mash.mask_boundary_sample_num,
            self.mash.mask_params,
            self.mash.sh_params,
            self.mash.rotate_vectors,
            self.mash.positions,
            self.boundary_phis,
            self.boundary_theta_weights,
            self.boundary_idxs,
            self.inner_phis,
            self.inner_theta_weights,
            self.inner_idxs,

            self.active_sh_degree,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,
        ) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_xyz(self):
        boundary_pts = self.mash.toWeightedSamplePoints(
            self.boundary_phis, self.boundary_theta_weights,
            self.boundary_idxs)
        inner_pts = self.mash.toWeightedSamplePoints(
            self.inner_phis, self.inner_theta_weights,
            self.inner_idxs)
        return torch.vstack([boundary_pts, inner_pts])

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = (
            torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2))
            .float()
            .cuda()
        )
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(
            distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()),
            0.0000001,
        )
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(
            0.1
            * torch.ones(
                (fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"
            )
        )

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(
            features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        return True
