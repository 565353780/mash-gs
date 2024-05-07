import sys

sys.path.append("../camera-manage/")

from mash_gs.Method.render import renderTrainGS
from mash_gs.Method.time import getLatestFolderName

data_folder_name_dict = {
    "0": "NeRF/3vjia_simple",
    "1": "NeRF/wine",
    "2": "NeRF/cup_1",
    "3": "UrbanScene3D/PolyTech_fine_zhang",
    "4": "NeRF/oven-train",
    "5": "NeRF/real_fridge-train",
}

data_folder_name = data_folder_name_dict["5"].replace("/", "_")
data_folder_name = getLatestFolderName(
    data_folder_name, "../mash-gs/output/"
)

output_folder_path = "../mash-gs/output/" + data_folder_name + "/"
port = 6007

renderTrainGS(output_folder_path, port)
