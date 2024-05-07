import torch
import numpy as np
from torch import nn

from ma_sh.Config.weights import W0
from ma_sh.Config.constant import EPSILON
from ma_sh.Model.mash import Mash
from ma_sh.Method.pcd import getPointCloud, downSample
from camera_manage.Config.cameras import BasicPointCloud

from mash_gs.Method.model import (
    get_expon_lr_func,
    inverse_sigmoid,
)
from mash_gs.Model.gaussians import GaussianModel
from mash_gs.Method.model import inverse_sigmoid
from simple_knn._C import distCUDA2



class MashGS(GaussianModel):
    def __init__(self, sh_degree: int, anchor_num: int=4000,
        mask_degree_max: int = 1,
        sh_degree_max: int = 1,
        mask_boundary_sample_num: int = 10,
        sample_polar_num: int = 1000,
        sample_point_scale: float = 0.2,
        use_inv: bool = True,
        idx_dtype=torch.int64,
        dtype=torch.float32,
        device: str = "cuda"):
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
        self.inner_theta_weights = inverse_sigmoid(torch.tensor([0.0], dtype=dtype).to(device))
        self.inner_idxs = torch.tensor([0.0], dtype=dtype).to(device)
        return

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
            self.inner_phis, torch.sigmoid(self.inner_theta_weights),
            self.inner_idxs)
        return torch.vstack([boundary_pts, inner_pts])

    def initMash(self, pts: np.ndarray) -> bool:
        gt_pcd = getPointCloud(pts)
        gt_pcd.estimate_normals()

        surface_dist = 0.00001

        anchor_pcd = downSample(gt_pcd, self.mash.anchor_num)

        if anchor_pcd is None:
            print("[ERROR][MashGS::initMash]")
            print("\t downSample failed!")
            return False

        sample_pts = np.asarray(anchor_pcd.points)
        sample_normals = np.asarray(anchor_pcd.normals)

        sh_params = torch.ones_like(self.mash.sh_params) * EPSILON
        sh_params[:, 0] = surface_dist / W0[0]

        self.mash.loadParams(
            sh_params=sh_params,
            positions=sample_pts + surface_dist * sample_normals,
            face_forward_vectors=-sample_normals,
        )
        return True

    def initGSParams(self) -> bool:
        mask_boundary_thetas = self.mash.toMaskBoundaryThetas()
        inner_phis, inner_theta_weights, inner_idxs = self.mash.toInMaskSamplePolars(mask_boundary_thetas)[:3]
        inner_pts = self.mash.toWeightedSamplePoints(inner_phis, inner_theta_weights, inner_idxs)
        fps_idxs = self.mash.toFPSPointIdxs(inner_pts, inner_idxs)

        self.inner_phis = inner_phis[fps_idxs]
        self.inner_theta_weights = inner_theta_weights[fps_idxs]
        self.inner_idxs = inner_idxs[fps_idxs]

        self.inner_phis.requires_grad_(True)
        self.inner_theta_weights.requires_grad_(True)
        return True

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale

        pts = np.asarray(pcd.points)

        self.initMash(pts)
        self.initGSParams()

        fused_point_cloud = self.get_xyz
        features = (
            torch.zeros((self.boundary_idxs.shape[0] + self.inner_idxs.shape[0], 3, (self.max_sh_degree + 1) ** 2))
            .float()
            .cuda()
        )
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(
            distCUDA2(fused_point_cloud.float().cuda()),
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

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {
                "params": [self.mash.mask_params,
                           self.mash.sh_params,
                           self.mash.rotate_vectors,
                           self.mash.positions,
                           self.inner_phis,
                           self.inner_theta_weights],
                "lr": training_args.position_lr_init * self.spatial_lr_scale,
                "name": "xyz",
            },
            {
                "params": [self._features_dc],
                "lr": training_args.feature_lr,
                "name": "f_dc",
            },
            {
                "params": [self._features_rest],
                "lr": training_args.feature_lr / 20.0,
                "name": "f_rest",
            },
            {
                "params": [self._opacity],
                "lr": training_args.opacity_lr,
                "name": "opacity",
            },
            {
                "params": [self._scaling],
                "lr": training_args.scaling_lr,
                "name": "scaling",
            },
            {
                "params": [self._rotation],
                "lr": training_args.rotation_lr,
                "name": "rotation",
            },
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )
        return True