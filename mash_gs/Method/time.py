import os
from datetime import datetime


def getCurrentTime():
    now = datetime.now()
    return now.strftime("%Y%m%d_%H:%M:%S")


def getLatestFolderName(base_folder_name, folder_root_path):
    folder_name_list = os.listdir(folder_root_path)
    max_folder_date = None
    max_folder_h = None
    max_folder_m = None
    max_folder_s = None
    for folder_name in folder_name_list:
        if base_folder_name not in folder_name:
            continue

        folder_time = folder_name.split(base_folder_name)[1][1:]
        folder_date, folder_second = folder_time.split("_")
        folder_h, folder_m, folder_s = folder_second.split(":")
        if max_folder_date is None:
            max_folder_date = folder_date
            max_folder_h = folder_h
            max_folder_m = folder_m
            max_folder_s = folder_s
            continue

        if int(folder_date) > int(max_folder_date):
            max_folder_date = folder_date
            max_folder_h = folder_h
            max_folder_m = folder_m
            max_folder_s = folder_s
            continue

        if int(folder_h) > int(max_folder_h):
            max_folder_date = folder_date
            max_folder_h = folder_h
            max_folder_m = folder_m
            max_folder_s = folder_s
            continue

        if int(folder_m) > int(max_folder_m):
            max_folder_date = folder_date
            max_folder_h = folder_h
            max_folder_m = folder_m
            max_folder_s = folder_s
            continue

        if int(folder_s) > int(max_folder_s):
            max_folder_date = folder_date
            max_folder_h = folder_h
            max_folder_m = folder_m
            max_folder_s = folder_s
            continue

    latest_folder_name = (
        base_folder_name
        + "_"
        + max_folder_date
        + "_"
        + max_folder_h
        + ":"
        + max_folder_m
        + ":"
        + max_folder_s
    )
    return latest_folder_name
