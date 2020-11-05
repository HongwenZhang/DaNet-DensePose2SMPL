# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import path_config
from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class JointsDataset(BaseDataset):
    def __init__(self, options, dataset, subset, use_augmentation, is_train=True):
        """
        Initialize the dataset.

        Args:
            self: (todo): write your description
            options: (dict): write your description
            dataset: (todo): write your description
            subset: (todo): write your description
            use_augmentation: (bool): write your description
            is_train: (bool): write your description
        """
        super().__init__(options, dataset, use_augmentation=use_augmentation, is_train=is_train)
        self.num_joints = 0
        self.pixel_std = 200
        self.flip_pairs = []
        self.parent_ids = []

        self.is_train = is_train
        self.root = path_config.DATASET_FOLDERS[dataset]
        self.image_set = subset

        # self.output_path = cfg.OUTPUT_DIR
        self.data_format = 'jpg'

    def _get_db(self):
        """
        Get the database.

        Args:
            self: (todo): write your description
        """
        raise NotImplementedError

    def evaluate(self, preds, output_dir, *args, **kwargs):
        """
        Evaluate the model.

        Args:
            self: (todo): write your description
            preds: (array): write your description
            output_dir: (str): write your description
        """
        raise NotImplementedError

    # def __len__(self,):
    #     return len(self.db)

    def select_data(self, db):
        """
        Selects the selected by the db.

        Args:
            self: (todo): write your description
            db: (array): write your description
        """
        db_selected = []
        for rec in db:
            num_vis = 0
            joints_x = 0.0
            joints_y = 0.0
            for joint, joint_vis in zip(
                    rec['joints_3d'], rec['joints_3d_vis']):
                if joint_vis[0] <= 0:
                    continue
                num_vis += 1

                joints_x += joint[0]
                joints_y += joint[1]
            if num_vis == 0:
                continue

            joints_x, joints_y = joints_x / num_vis, joints_y / num_vis

            area = rec['scale'][0] * rec['scale'][1] * (self.pixel_std**2)
            joints_center = np.array([joints_x, joints_y])
            bbox_center = np.array(rec['center'])
            diff_norm2 = np.linalg.norm((joints_center-bbox_center), 2)
            ks = np.exp(-1.0*(diff_norm2**2) / ((0.2)**2*2.0*area))

            metric = (0.2 / 16) * num_vis + 0.45 - 0.2 / 16
            if ks > metric:
                db_selected.append(rec)

        logger.info('=> num db: {}'.format(len(db)))
        logger.info('=> num selected db: {}'.format(len(db_selected)))
        return db_selected

