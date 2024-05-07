import os
import json
import random

from camera_manage.Method.colmap.scene import readColmapSceneInfo

from gaussian_splatting.Config.params import ModelParams
from gaussian_splatting.Model.gaussians import GaussianModel
from gaussian_splatting.Method.path import searchForMaxIteration
from gaussian_splatting.Method.camera import cameraList_from_camInfos, camera_to_JSON


class Scene(object):
    def __init__(
        self,
        lp: ModelParams,
        gaussians: GaussianModel,
        load_iteration=None,
    ):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = lp.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        self.cameras = {}

        self.point_cloud = None

        self.loadColmapDataset(lp)

        self.loadTrainedModel(load_iteration)
        return

    def loadColmapDataset(self, lp):
        if not os.path.exists(lp.source_path + "sparse/"):
            print("[ERROR][scene::__init__]")
            print("\t dataset not exist!")
            print("\t source_path:", lp.source_path + "sparse/")
            exit()

        scene_info = readColmapSceneInfo(lp.source_path, lp.images)

        if not self.loaded_iter:
            with open(scene_info.ply_path, "rb") as src_file, open(
                os.path.join(self.model_path, "input.ply"), "wb"
            ) as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.cameras:
                camlist.extend(scene_info.cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), "w") as file:
                json.dump(json_cams, file)

        # Multi-res consistent random shuffling
        random.shuffle(scene_info.cameras)

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        print("Loading Cameras")
        self.cameras[1.0] = cameraList_from_camInfos(scene_info.cameras, 1.0, lp)

        self.point_cloud = scene_info.point_cloud
        return True

    def loadTrainedModel(self, load_iteration):
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(
                    os.path.join(self.model_path, "point_cloud")
                )
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        if self.loaded_iter:
            self.gaussians.load_ply(
                os.path.join(
                    self.model_path,
                    "point_cloud",
                    "iteration_" + str(self.loaded_iter),
                    "point_cloud.ply",
                )
            )
        else:
            self.gaussians.create_from_pcd(self.point_cloud, self.cameras_extent)
        return True

    def save(self, iteration):
        point_cloud_path = os.path.join(
            self.model_path, "point_cloud/iteration_{}".format(iteration)
        )
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getCameras(self, scale=1.0):
        return self.cameras[scale]
