import sys

sys.path.append("../camera-manage/")

from gaussian_splatting.Module.trainer import Trainer


def demo():
    trainer = Trainer()
    trainer.train()
    return True
