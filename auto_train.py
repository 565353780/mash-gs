import sys

sys.path.append("../camera-manage/")
sys.path.append("../colmap-manage/")
sys.path.append("../ma-sh/")

from colmap_manage.Module.colmap_manager import COLMAPManager
from colmap_manage.Module.dataset_manager import DatasetManager
from mash_gs.Module.trainer import Trainer

data_folder_name_dict = {
    "0": "NeRF/3vjia_simple",
    "1": "NeRF/wine",
    "2": "NeRF/cup_1",
    "3": "UrbanScene3D/PolyTech_fine_zhang",
    "4": "NeRF/oven-train",
    "5": "NeRF/real_fridge-train",
    "6": "NeRF/real_fridge_raw-train",
}

data_folder_name = data_folder_name_dict["5"]
video_file_path = "/home/chli/chLi/Dataset/NeRF/3vjia_person/3vjia_person.mp4"
video_file_path = None
down_sample_scale = 1
resolution = 1

scale = 1
show_image = False
print_progress = True
remove_old = False
remain_db = True
valid_percentage = 0.8
method_dict = {}

data_folder_path = "/home/chli/Dataset/" + data_folder_name + "/"
dataset_folder_path = (
    "../colmap-manage/output/" + data_folder_name.replace("/", "_") + "/"
)
output_folder_path = (
    "../mash-gs/output/" + data_folder_name.replace("/", "_") + "/"
)
image_folder_name = "images"
iterations = 400000
port = 6007
percentent_dense = 0.01

COLMAPManager(
    data_folder_path, video_file_path, down_sample_scale, print_progress=False
).autoGenerateData(remove_old, remain_db, valid_percentage)
DatasetManager().generateDataset(
    "gs", data_folder_path, dataset_folder_path, method_dict
)

Trainer(
    dataset_folder_path + "gs/",
    data_folder_name.replace("/", "_"),
    resolution,
    iterations,
    port,
    percentent_dense,
).train()
