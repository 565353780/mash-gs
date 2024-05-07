import sys

sys.path.append("../camera-manage/")
sys.path.append("../udf-manage/")

from mash_gs.Module.controller import Controller


def demo():
    controller = Controller()
    controller.train()
    return True
