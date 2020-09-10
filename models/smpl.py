# This script is borrowed from https://github.com/nkolot/SPIN/blob/master/models/smpl.py

import torch
import numpy as np
import smplx
from smplx import SMPL as _SMPL
from smplx.body_models import ModelOutput
from smplx.lbs import vertices2joints
from collections import namedtuple

import path_config
import constants


class SMPL(_SMPL):
    """ Extension of the official SMPL implementation to support more joints """

    def __init__(self, *args, **kwargs):
        super(SMPL, self).__init__(*args, **kwargs)
        joints = [constants.JOINT_MAP[i] for i in constants.JOINT_NAMES]
        J_regressor_extra = np.load(path_config.JOINT_REGRESSOR_TRAIN_EXTRA)
        self.register_buffer('J_regressor_extra', torch.tensor(J_regressor_extra, dtype=torch.float32))
        self.joint_map = torch.tensor(joints, dtype=torch.long)
        self.ModelOutput = namedtuple('ModelOutput_', ModelOutput._fields + ('smpl_joints', 'joints_J19',))
        self.ModelOutput.__new__.__defaults__ = (None,) * len(self.ModelOutput._fields)

    def forward(self, *args, **kwargs):
        kwargs['get_skin'] = True
        smpl_output = super(SMPL, self).forward(*args, **kwargs)
        extra_joints = vertices2joints(self.J_regressor_extra, smpl_output.vertices)
        # smpl_output.joints: [B, 45, 3]  extra_joints: [B, 9, 3]
        vertices = smpl_output.vertices
        joints = torch.cat([smpl_output.joints, extra_joints], dim=1)
        smpl_joints = smpl_output.joints[:, :24]
        joints = joints[:, self.joint_map, :]   # [B, 49, 3]
        joints_J24 = joints[:, -24:, :]
        joints_J19 = joints_J24[:, constants.J24_TO_J19, :]
        output = self.ModelOutput(vertices=vertices,
                                  global_orient=smpl_output.global_orient,
                                  body_pose=smpl_output.body_pose,
                                  joints=joints,
                                  joints_J19=joints_J19,
                                  smpl_joints=smpl_joints,
                                  betas=smpl_output.betas,
                                  full_pose=smpl_output.full_pose)
        return output
