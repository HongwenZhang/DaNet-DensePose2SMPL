import os
import torch
import random
import numpy as np

from train import TrainOptions, Trainer
from models.core.config import cfg, cfg_from_file
from easydict import EasyDict


if __name__ == '__main__':
    options = TrainOptions().parse_args()
    cfg_from_file(options.danet_cfg_file)
    cfg.DANET.REFINEMENT = EasyDict(cfg.DANET.REFINEMENT)
    cfg.MSRES_MODEL.EXTRA = EasyDict(cfg.MSRES_MODEL.EXTRA)

    trainer = Trainer(options)
    trainer.train()
