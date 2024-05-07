import os
from random import randint

import gaussian_splatting.Method.network as network_gui
import torch
from gaussian_splatting.Config.params import (
    ModelParams,
    OptimizationParams,
)
from gaussian_splatting.Config.train import getTrainConfig
from gaussian_splatting.Data.scene import Scene
from gaussian_splatting.Loss.loss import l1_loss, ssim
from gaussian_splatting.Method.model import safe_state
from gaussian_splatting.Method.render import render
from gaussian_splatting.Method.time import getCurrentTime
from gaussian_splatting.Method.train import prepare_output_and_logger, training_report
from gaussian_splatting.Model.gaussians import GaussianModel
from tqdm import tqdm


class Controller(object):
    def __init__(
        self,
        source_path=None,
        folder_name=None,
        resolution=None,
        iterations=None,
        port=None,
        percent_dense=None,
    ):
        train_config = getTrainConfig(folder_name, getCurrentTime())

        self.source_path = train_config["dataset_folder_path"]
        self.model_path = train_config["output_folder_path"]
        self.iterations = train_config["iterations"]
        self.resolution = train_config["resolution"]
        self.percent_dense = train_config["percent_dense"]
        self.ip = train_config["ip"]
        self.port = train_config["port"]

        self.detect_anomaly = train_config["detect_anomaly"]
        self.save_iterations = [i * 2500 for i in range(int(self.iterations / 2500))]
        if self.iterations not in self.save_iterations:
            self.save_iterations.append(self.iterations)
        self.test_iterations = self.save_iterations
        self.checkpoint_iterations = train_config["checkpoint_iterations"]
        self.quiet = train_config["quiet"]
        self.start_checkpoint = train_config["start_checkpoint"]

        if source_path is not None:
            self.source_path = source_path
        if resolution is not None:
            self.resolution = resolution
        if iterations is not None:
            self.iterations = iterations
        if port is not None:
            self.port = port
        if percent_dense is not None:
            self.percent_dense = percent_dense

        os.makedirs(self.model_path, exist_ok=True)

        # Params
        self.lp = ModelParams()
        self.lp.source_path = self.source_path
        self.lp.model_path = self.model_path
        self.lp.resolution = self.resolution

        self.op = OptimizationParams()
        self.op.percent_dense = self.percent_dense
        self.op.iterations = self.iterations

        # Model
        self.gaussians = GaussianModel(self.lp.sh_degree)
        self.first_iter = 0
        self.scene = Scene(self.lp, self.gaussians)
        self.gaussians.training_setup(self.op)

        # Log
        self.tb_writer = prepare_output_and_logger(self.lp)

        # Render
        bg_color = [1, 1, 1] if self.lp.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        print("Optimizing " + self.model_path)

        # Initialize system state (RNG)
        safe_state(self.quiet)

        # Start GUI server, configure and run training
        network_gui.init(self.ip, self.port)
        torch.autograd.set_detect_anomaly(self.detect_anomaly)
        return

    def keepServerAlive(self, iteration):
        if network_gui.conn is None:
            network_gui.try_connect()

        while network_gui.conn is not None:
            try:
                net_image_bytes = None
                (
                    custom_cam,
                    do_training,
                    keep_alive,
                    scaling_modifer,
                ) = network_gui.receive()
                if custom_cam is not None:
                    net_image = render(
                        custom_cam,
                        self.gaussians,
                        self.background,
                        scaling_modifer,
                    )["render"]
                    net_image_bytes = memoryview(
                        (torch.clamp(net_image, min=0, max=1.0) * 255)
                        .byte()
                        .permute(1, 2, 0)
                        .contiguous()
                        .cpu()
                        .numpy()
                    )
                network_gui.send(net_image_bytes, self.lp.source_path)
                if do_training and (
                    (iteration < int(self.op.iterations)) or not keep_alive
                ):
                    break
            except Exception:
                network_gui.conn = None
        return True

    def loadModel(self, model_file_path):
        if not os.path.exists(model_file_path):
            print("[ERROR][Trainer::loadModel]")
            print("\t model file not exist!")
            print("\t model_file_path:", model_file_path)
            return False

        (model_params, self.first_iter) = torch.load(self.start_checkpoint)
        self.gaussians.restore(model_params, self.op)
        return True

    def saveModel(self, ply_file_path):
        self.gaussians.save_ply(ply_file_path)
        return True

    def trainStep(self, viewpoint_cam):
        # Render
        render_pkg = render(viewpoint_cam, self.gaussians, self.background)
        image = render_pkg["render"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - self.op.lambda_dssim) * Ll1 + self.op.lambda_dssim * (
            1.0 - ssim(image, gt_image)
        )
        loss.backward()
        return Ll1, loss, render_pkg

    def train(self):
        if self.start_checkpoint:
            self.loadModel(self.start_checkpoint)

        iter_start = torch.cuda.Event(enable_timing=True)
        iter_end = torch.cuda.Event(enable_timing=True)

        viewpoint_stack = None
        ema_loss_for_log = 0.0

        progress_bar = tqdm(
            range(self.first_iter, self.op.iterations), desc="Training progress"
        )

        self.first_iter += 1
        for iteration in range(self.first_iter, self.op.iterations + 1):
            self.keepServerAlive(iteration)

            iter_start.record()

            self.gaussians.update_learning_rate(iteration)

            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 1000 == 0:
                self.gaussians.oneupSHdegree()

            # Pick a random Camera
            if not viewpoint_stack:
                viewpoint_stack = self.scene.getCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

            Ll1, loss, render_pkg = self.trainStep(viewpoint_cam)

            viewspace_point_tensor, visibility_filter, radii = (
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
            )

            iter_end.record()

            with torch.no_grad():
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                    progress_bar.update(10)
                if iteration == self.op.iterations:
                    progress_bar.close()

                # Log and save
                training_report(
                    self.tb_writer,
                    iteration,
                    Ll1,
                    loss,
                    l1_loss,
                    iter_start.elapsed_time(iter_end),
                    self.test_iterations,
                    self.scene,
                    render,
                    self.background,
                )
                if iteration in self.save_iterations:
                    print("\n[ITER {}] Saving gaussians".format(iteration))
                    self.scene.save(iteration)

                # Densification
                if iteration < self.op.densify_until_iter:
                    # Keep track of max radii in image-space for pruning
                    self.gaussians.max_radii2D[visibility_filter] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter],
                        radii[visibility_filter],
                    )
                    self.gaussians.add_densification_stats(
                        viewspace_point_tensor, visibility_filter
                    )

                    if (
                        iteration > self.op.densify_from_iter
                        and iteration % self.op.densification_interval == 0
                    ):
                        size_threshold = (
                            20 if iteration > self.op.opacity_reset_interval else None
                        )
                        self.gaussians.densify_and_prune(
                            self.op.densify_grad_threshold,
                            0.005,
                            self.scene.cameras_extent,
                            size_threshold,
                        )

                    if iteration % self.op.opacity_reset_interval == 0 or (
                        self.lp.white_background
                        and iteration == self.op.densify_from_iter
                    ):
                        self.gaussians.reset_opacity()

                # Optimizer step
                if iteration < self.op.iterations:
                    self.gaussians.optimizer.step()
                    self.gaussians.optimizer.zero_grad(set_to_none=True)

                if iteration in self.checkpoint_iterations:
                    print("\n[ITER {}] Saving Checkpoint".format(iteration))
                    torch.save(
                        (self.gaussians.capture(), iteration),
                        self.scene.model_path + "/chkpnt" + str(iteration) + ".pth",
                    )
        return True
