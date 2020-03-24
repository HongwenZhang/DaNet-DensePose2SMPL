import os
import torch
import json
import sys
import numpy as np
import pickle
from lib.utils.smpl import batch_global_rigid_transformation, batch_rodrigues, reflect_pose
import torch.nn as nn


class SMPL(nn.Module):
    '''
        copyright
        file:   https://github.com/MandyMo/pytorch_HMR/blob/master/src/SMPL.py

        date:   2018_05_03
        author: zhangxiong(1025679612@qq.com)
        mark:   the algorithm is cited from original SMPL
    '''
    def __init__(self, model_type=None, joint_type='cocoplus', obj_saveable=False, max_batch_size=20):
        super(SMPL, self).__init__()

        if joint_type not in ['cocoplus', 'lsp', 'smpl']:
            msg = 'unknow joint type: {}, it must be either "cocoplus" or "lsp" or "smpl"'.format(joint_type)
            sys.exit(msg)

        self.joint_type = joint_type

        # Now read the smpl model.
        if model_type == 'male':
            smpl_model_path = './data/SMPL_data/basicmodel_m_lbs_10_207_0_v1.0.0.pkl'
        elif model_type == 'neutral':
            smpl_model_path = './data/SMPL_data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'
        elif model_type == 'female':
            smpl_model_path = './data/SMPL_data/basicModel_f_lbs_10_207_0_v1.0.0.pkl'

        with open(smpl_model_path, 'rb') as f:
            model = pickle.load(f, encoding='iso-8859-1')

        if obj_saveable:
            self.faces = model['f']
        else:
            self.faces = None

        np_v_template = model['v_template']
        self.register_buffer('v_template', torch.from_numpy(np_v_template).float())
        self.size = [np_v_template.shape[0], 3]

        np_shapedirs = model['shapedirs'].x
        self.num_betas = np_shapedirs.shape[-1]
        np_shapedirs = np.reshape(np_shapedirs, [-1, self.num_betas]).T
        self.register_buffer('shapedirs', torch.from_numpy(np_shapedirs).float())

        np_J_regressor = model['J_regressor'].toarray().T
        self.register_buffer('J_regressor', torch.from_numpy(np_J_regressor).float())

        np_posedirs = model['posedirs']
        num_pose_basis = np_posedirs.shape[-1]
        np_posedirs = np.reshape(np_posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs', torch.from_numpy(np_posedirs).float())

        self.parents = model['kintree_table'][0].astype(np.int32)

        with open(os.path.join('./data/pretrained_model', 'joint_regressor.pkl'), 'rb') as f:
            np_joint_regressor = pickle.load(f, encoding='iso-8859-1')

        if joint_type == 'lsp':
            self.register_buffer('joint_regressor', torch.from_numpy(np_joint_regressor[:, :14]).float())
        else:
            self.register_buffer('joint_regressor', torch.from_numpy(np_joint_regressor).float())

        np_weights = model['weights']

        vertex_count = np_weights.shape[0]
        vertex_component = np_weights.shape[1]

        np_weights = np.tile(np_weights, (max_batch_size, 1))
        self.register_buffer('weight', torch.from_numpy(np_weights).float().reshape(-1, vertex_count, vertex_component))

        self.register_buffer('e3', torch.eye(3).float())

        self.cur_device = None

    def save_obj(self, verts, obj_mesh_name):
        if self.faces is None:
            msg = 'obj not saveable!'
            sys.exit(msg)

        with open(obj_mesh_name, 'w') as fp:
            for v in verts:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

            for f in self.faces:  # Faces are 1-based, not 0-based in obj files
                fp.write('f %d %d %d\n' % (f[0] + 1, f[1] + 1, f[2] + 1))

    def forward(self, beta, theta=None, Rs=None, get_skin=False, add_smpl_joint=False):
        if not self.cur_device:
            device = beta.device
            self.cur_device = torch.device(device.type, device.index)

        return_points = {}

        num_batch = beta.shape[0]

        v_shaped = torch.matmul(beta, self.shapedirs).view(-1, self.size[0], self.size[1]) + self.v_template
        Jx = torch.matmul(v_shaped[:, :, 0], self.J_regressor)
        Jy = torch.matmul(v_shaped[:, :, 1], self.J_regressor)
        Jz = torch.matmul(v_shaped[:, :, 2], self.J_regressor)
        J = torch.stack([Jx, Jy, Jz], dim=2)

        if Rs is None:
            Rs = batch_rodrigues(theta.view(-1, 3)).view(-1, 24, 3, 3)
        pose_feature = (Rs[:, 1:, :, :]).sub(1.0, self.e3).view(-1, 207)
        v_posed = torch.matmul(pose_feature, self.posedirs).view(-1, self.size[0], self.size[1]) + v_shaped
        self.J_transformed, A = batch_global_rigid_transformation(Rs, J, self.parents, rotate_base=True)

        weight = self.weight[:num_batch]
        W = weight.view(num_batch, -1, 24)
        T = torch.matmul(W, A.view(num_batch, 24, 16)).view(num_batch, -1, 4, 4)

        v_posed_homo = torch.cat([v_posed, torch.ones(num_batch, v_posed.shape[1], 1, device=self.cur_device)], dim=2)
        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, -1))

        verts = v_homo[:, :, :3, 0]

        if get_skin:
            return_points['verts'] = verts

        joint_x = torch.matmul(verts[:, :, 0], self.joint_regressor)
        joint_y = torch.matmul(verts[:, :, 1], self.joint_regressor)
        joint_z = torch.matmul(verts[:, :, 2], self.joint_regressor)

        joints = torch.stack([joint_x, joint_y, joint_z], dim=2)

        return_points['cocoplus'] = joints

        if add_smpl_joint:
            joints = [joints]

            smpl_joint_x = torch.matmul(verts[:, :, 0], self.J_regressor)
            smpl_joint_y = torch.matmul(verts[:, :, 1], self.J_regressor)
            smpl_joint_z = torch.matmul(verts[:, :, 2], self.J_regressor)

            smpl_joints = torch.stack([smpl_joint_x, smpl_joint_y, smpl_joint_z], dim=2)
            joints.append(smpl_joints)

            return_points['smpl'] = smpl_joints

        return return_points