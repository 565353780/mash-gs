TRAIN_CONFIG = {
    "dataset_folder_path": "../colmap-manage/output/<folder-name>/gs/",
    "output_folder_path": "../gaussian-splatting/output/<folder-name>/",
    "start_checkpoint": None,
    "iterations": 400000,
    "resolution": 8,
    "ip": "127.0.0.1",
    "port": 6007,
    "device": "cuda",
    "percent_dense": 0.01,
    "detect_anomaly": False,
    "checkpoint_iterations": [],
    "quiet": False,
}


def getTrainConfig(dataset_folder_name, log_folder_name):
    train_config = TRAIN_CONFIG
    if dataset_folder_name is not None:
        train_config["dataset_folder_path"] = train_config[
            "dataset_folder_path"
        ].replace("<folder-name>", dataset_folder_name)
    if log_folder_name is not None:
        if dataset_folder_name is not None:
            train_config["output_folder_path"] = train_config[
                "output_folder_path"
            ].replace("<folder-name>", dataset_folder_name + "_" + log_folder_name)
        else:
            train_config["output_folder_path"] = train_config[
                "output_folder_path"
            ].replace("<folder-name>", log_folder_name)
    return train_config
