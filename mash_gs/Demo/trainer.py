import sys

sys.path.append("../camera-manage/")
sys.path.append("../../ASDF/ma-sh/")

from mash_gs.Module.trainer import Trainer


def demo():
    trainer = Trainer()
    trainer.train()
    return True
