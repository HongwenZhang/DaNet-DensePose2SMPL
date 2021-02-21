from functools import wraps
import torch
from models.core.config import cfg
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from models.module.GCN import GCN
from utils.graph import Graph, normalize_digraph, normalize_undigraph
from utils.iuvmap import iuv_img2map, iuv_map2img
from utils.geometry import rot6d_to_rotmat
from utils.smpl_utlis import smpl_structure
from utils.geometry import perspective_projection


from models.smpl import SMPL
from models.module.res_module import SmplResNet, LimbResLayers

import path_config


def check_inference(net_func):
    @wraps(net_func)
    def wrapper(self, *args, **kwargs):
        if not self.training:
            if cfg.PYTORCH_VERSION_LESS_THAN_040:
                return net_func(self, *args, **kwargs)
            else:
                with torch.no_grad():
                    return net_func(self, *args, **kwargs)
        else:
            raise ValueError('You should call this function only on inference.'
                             'Set the network in inference mode by net.eval().')

    return wrapper


class SMPL_Regressor(nn.Module):
    def __init__(self, options, orig_size=224, feat_in_dim=None, smpl_mean_params=None, pretrained=True):
        super(SMPL_Regressor, self).__init__()

        self.mapping_to_detectron = None
        self.orphans_in_detectron = None

        self.focal_length = 5000.
        self.options = options

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.orig_size = orig_size

        mean_params = np.load(smpl_mean_params)
        init_pose_6d = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        if cfg.DANET.USE_6D_ROT:
            init_pose = init_pose_6d
        else:
            init_pose_rotmat = rot6d_to_rotmat(init_pose_6d)
            init_pose = init_pose_rotmat.reshape(-1).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)

        init_params = (init_cam, init_shape, init_pose)

        self.smpl = SMPL(path_config.SMPL_MODEL_DIR, batch_size=self.options.batch_size, create_transl=False)

        if cfg.DANET.DECOMPOSED:
            print('using decomposed predictor.')
            self.smpl_para_Outs = DecomposedPredictor(feat_in_dim, init_params, pretrained)
        else:
            print('using global predictor.')
            self.smpl_para_Outs = GlobalPredictor(feat_in_dim, pretrained)

        # Per-vertex loss on the shape
        self.criterion_shape = nn.L1Loss().to(self.device)
        # Keypoint (2D and 3D) loss
        # No reduction because confidence weighting needs to be applied
        self.criterion_keypoints = nn.MSELoss(reduction='none').to(self.device)
        # Loss for SMPL parameter regression
        self.criterion_regr = nn.MSELoss().to(self.device)

    @check_inference
    def smpl_infer_net(self, in_dict):
        """For inference"""
        in_dict['infer_mode'] = True
        return_dict = self._forward(in_dict)

        return return_dict

    def forward(self, in_dict):
        if cfg.PYTORCH_VERSION_LESS_THAN_040:
            return self._forward(in_dict)
        else:
            with torch.set_grad_enabled(self.training):
                return self._forward(in_dict)

    def _forward(self, in_dict):
        iuv_map = in_dict['iuv_map']
        part_iuv_map = in_dict['part_iuv_map'] if 'part_iuv_map' in in_dict else None
        infer_mode = in_dict['infer_mode'] if 'infer_mode' in in_dict else False

        if cfg.DANET.INPUT_MODE in ['feat', 'iuv_feat', 'iuv_gt_feat']:
            device_id = iuv_map['feat'].get_device()
        elif cfg.DANET.INPUT_MODE == 'seg':
            device_id = iuv_map['index'].get_device()
        else:
            device_id = iuv_map.get_device()
        return_dict = {}
        return_dict['losses'] = {}
        return_dict['metrics'] = {}
        return_dict['visualization'] = {}
        return_dict['prediction'] = {}

        if cfg.DANET.DECOMPOSED:
            smpl_out_dict = self.smpl_para_Outs(iuv_map, part_iuv_map)
        else:
            smpl_out_dict = self.smpl_para_Outs(iuv_map)

        if infer_mode:
            return smpl_out_dict

        para = smpl_out_dict['para']

        for k, v in smpl_out_dict['visualization'].items():
            return_dict['visualization'][k] = v

        return_dict['prediction']['cam'] = para[:, :3]
        return_dict['prediction']['shape'] = para[:, 3:13]
        return_dict['prediction']['pose'] = para[:, 13:].reshape(-1, 24, 3, 3).contiguous()

        # losses for Training
        if self.training:
            batch_size = len(para)

            target = in_dict['target']
            target_kps = in_dict['target_kps']
            target_kps3d = in_dict['target_kps3d']
            target_vertices = in_dict['target_verts']
            has_kp3d = in_dict['has_kp3d']
            has_smpl = in_dict['has_smpl']

            if cfg.DANET.ORTHOGONAL_WEIGHTS > 0:
                loss_orth = self.orthogonal_loss(para)
                loss_orth *= cfg.DANET.ORTHOGONAL_WEIGHTS
                return_dict['losses']['Rs_orth'] = loss_orth
                return_dict['metrics']['orth'] = loss_orth.detach()

            if len(smpl_out_dict['joint_rotation']) > 0:
                for stack_i in range(len(smpl_out_dict['joint_rotation'])):
                    if torch.sum(has_smpl) > 0:
                        loss_rot = self.criterion_regr(smpl_out_dict['joint_rotation'][stack_i][has_smpl==1], target[:, 13:][has_smpl==1])
                        loss_rot *= cfg.DANET.SMPL_POSE_WEIGHTS
                    else:
                        loss_rot = torch.zeros(1).to(pred.device)

                    return_dict['losses']['joint_rotation'+str(stack_i)] = loss_rot

            if cfg.DANET.DECOMPOSED and ('joint_position' in smpl_out_dict) and cfg.DANET.JOINT_POSITION_WEIGHTS > 0:
                gt_beta = target[:, 3:13].contiguous().detach()
                gt_Rs = target[:, 13:].contiguous().view(-1, 24, 3, 3).detach()
                smpl_pts = self.smpl(betas=gt_beta, body_pose=gt_Rs[:, 1:],
                                     global_orient=gt_Rs[:, 0].unsqueeze(1), pose2rot=False)
                gt_smpl_coord = smpl_pts.smpl_joints
                for stack_i in range(len(smpl_out_dict['joint_position'])):
                    loss_pos = self.l1_losses(smpl_out_dict['joint_position'][stack_i], gt_smpl_coord, has_smpl)
                    loss_pos *= cfg.DANET.JOINT_POSITION_WEIGHTS
                    return_dict['losses']['joint_position'+str(stack_i)] = loss_pos

            pred_camera = para[:, :3]
            pred_betas = para[:, 3:13]
            pred_rotmat = para[:, 13:].reshape(-1, 24, 3, 3).contiguous()

            gt_camera = target[:, :3]
            gt_betas = target[:, 3:13]
            gt_rotmat = target[:, 13:]

            pred_output = self.smpl(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
            pred_vertices = pred_output.vertices
            pred_joints = pred_output.joints

            # Convert Weak Perspective Camera [s, tx, ty] to camera translation [tx, ty, tz] in 3D given the bounding box size
            # This camera translation can be used in a full perspective projection
            pred_cam_t = torch.stack([pred_camera[:,1],
                                        pred_camera[:,2],
                                        2*self.focal_length/(cfg.DANET.INIMG_SIZE * pred_camera[:,0] +1e-9)],dim=-1)

            camera_center = torch.zeros(batch_size, 2, device=self.device)
            pred_keypoints_2d = perspective_projection(pred_joints,
                                                        rotation=torch.eye(3, device=self.device).unsqueeze(0).expand(batch_size, -1, -1),
                                                        translation=pred_cam_t,
                                                        focal_length=self.focal_length,
                                                        camera_center=camera_center)
            # Normalize keypoints to [-1,1]
            pred_keypoints_2d = pred_keypoints_2d / (cfg.DANET.INIMG_SIZE / 2.)

            # Compute loss on predicted camera
            loss_cam = self.l1_losses(pred_camera, gt_camera, has_smpl)

            # Compute loss on SMPL parameters
            loss_regr_pose, loss_regr_betas = self.smpl_losses(pred_rotmat, pred_betas, gt_rotmat, gt_betas, has_smpl)

            # Compute 2D reprojection loss for the keypoints
            loss_keypoints = self.keypoint_loss(pred_keypoints_2d, target_kps,
                                                self.options.openpose_train_weight,
                                                self.options.gt_train_weight)

            # Compute 3D keypoint loss
            loss_keypoints_3d = self.keypoint_3d_loss(pred_joints, target_kps3d, has_kp3d)

            # Per-vertex loss for the shape
            loss_verts = self.shape_loss(pred_vertices, target_vertices, has_smpl)

            # The last component is a loss that forces the network to predict positive depth values
            return_dict['losses'].update({'keypoints_2d': loss_keypoints * cfg.DANET.PROJ_KPS_WEIGHTS,
                                    'keypoints_3d': loss_keypoints_3d * cfg.DANET.KPS3D_WEIGHTS,
                                    'smpl_pose': loss_regr_pose * cfg.DANET.SMPL_POSE_WEIGHTS,
                                    'smpl_betas': loss_regr_betas * cfg.DANET.SMPL_BETAS_WEIGHTS,
                                    'smpl_verts': loss_verts * cfg.DANET.VERTS_WEIGHTS,
                                    'cam': ((torch.exp(-pred_camera[:,0]*10)) ** 2 ).mean()})

            return_dict['prediction']['vertices'] = pred_vertices
            return_dict['prediction']['cam_t'] = pred_cam_t

        # handle bug on gathering scalar(0-dim) tensors
        for k, v in return_dict['losses'].items():
            if len(v.shape) == 0:
                return_dict['losses'][k] = v.unsqueeze(0)
        for k, v in return_dict['metrics'].items():
            if len(v.shape) == 0:
                return_dict['metrics'][k] = v.unsqueeze(0)

        return return_dict

    def l1_losses(self, pred, target, mask):
        if torch.sum(mask) > 0:
            para_loss = F.l1_loss(pred[mask==1], target[mask==1], size_average=False) / target[mask==1].size(0)
        else:
            para_loss = torch.zeros(1).to(pred.device)
        return para_loss

    def orthogonal_loss(self, para):
        device_id = para.get_device()
        Rs_pred = para[:, 13:].contiguous().view(-1, 3, 3)
        Rs_pred_transposed = torch.transpose(Rs_pred, 1, 2)
        Rs_mm = torch.bmm(Rs_pred, Rs_pred_transposed)
        tensor_eyes = torch.eye(3).expand_as(Rs_mm).cuda(device_id)
        return F.mse_loss(Rs_mm, tensor_eyes)

    def keypoint_loss(self, pred_keypoints_2d, gt_keypoints_2d, openpose_weight, gt_weight):
        """ Compute 2D reprojection loss on the keypoints.
        The loss is weighted by the confidence.
        The available keypoints are different for each dataset.
        """
        conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
        conf[:, :25] *= openpose_weight
        conf[:, 25:] *= gt_weight
        loss = (conf * self.criterion_keypoints(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).mean()
        return loss

    def keypoint_3d_loss(self, pred_keypoints_3d, gt_keypoints_3d, has_pose_3d):
        """Compute 3D keypoint loss for the examples that 3D keypoint annotations are available.
        The loss is weighted by the confidence.
        """
        pred_keypoints_3d = pred_keypoints_3d[:, 25:, :]
        conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
        gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].clone()
        gt_keypoints_3d = gt_keypoints_3d[has_pose_3d == 1]
        conf = conf[has_pose_3d == 1]
        pred_keypoints_3d = pred_keypoints_3d[has_pose_3d == 1]
        if len(gt_keypoints_3d) > 0:
            gt_pelvis = (gt_keypoints_3d[:, 2,:] + gt_keypoints_3d[:, 3,:]) / 2
            gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]
            pred_pelvis = (pred_keypoints_3d[:, 2,:] + pred_keypoints_3d[:, 3,:]) / 2
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, None, :]
            return (conf * self.criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d)).mean()
        else:
            return torch.FloatTensor(1).fill_(0.).to(self.device)

    def shape_loss(self, pred_vertices, gt_vertices, has_smpl):
        """Compute per-vertex loss on the shape for the examples that SMPL annotations are available."""
        pred_vertices_with_shape = pred_vertices[has_smpl == 1]
        gt_vertices_with_shape = gt_vertices[has_smpl == 1]
        if len(gt_vertices_with_shape) > 0:
            return self.criterion_shape(pred_vertices_with_shape, gt_vertices_with_shape)
        else:
            return torch.FloatTensor(1).fill_(0.).to(self.device)

    def smpl_losses(self, pred_rotmat, pred_betas, gt_rotmat, gt_betas, has_smpl):
        pred_rotmat_valid = pred_rotmat[has_smpl == 1]
        gt_rotmat_valid = gt_rotmat.view(-1, 24, 3, 3)[has_smpl == 1]
        pred_betas_valid = pred_betas[has_smpl == 1]
        gt_betas_valid = gt_betas[has_smpl == 1]
        if len(pred_rotmat_valid) > 0:
            loss_regr_pose = self.criterion_regr(pred_rotmat_valid, gt_rotmat_valid)
            loss_regr_betas = self.criterion_regr(pred_betas_valid, gt_betas_valid)
        else:
            loss_regr_pose = torch.FloatTensor(1).fill_(0.).to(self.device)
            loss_regr_betas = torch.FloatTensor(1).fill_(0.).to(self.device)
        return loss_regr_pose, loss_regr_betas

    # @property
    def detectron_weight_mapping(self):
        if self.mapping_to_detectron is None:
            d_wmap = {}  # detectron_weight_mapping
            d_orphan = []  # detectron orphan weight list
            for name, m_child in self.named_children():
                if list(m_child.parameters()):  # if module has any parameter
                    child_map, child_orphan = m_child.detectron_weight_mapping
                    d_orphan.extend(child_orphan)
                    for key, value in child_map.items():
                        new_key = name + '.' + key
                        d_wmap[new_key] = value
                else:
                    new_key = [key for key in self.state_dict().keys() if key.startswith(name + '.')]
                    for key in new_key:
                        d_wmap[key] = None
            self.mapping_to_detectron = d_wmap
            self.orphans_in_detectron = d_orphan

        return self.mapping_to_detectron, self.orphans_in_detectron

class GlobalPredictor(nn.Module):
    def __init__(self, feat_in_dim=None, pretrained=True):
        super(GlobalPredictor, self).__init__()

        # For cache
        self.mapping_to_detectron = None
        self.orphans_in_detectron = None

        self.add_feat_ch = feat_in_dim

        # Backbone for feature extraction
        if cfg.DANET.INPUT_MODE == 'rgb':
            print('input mode: RGB')
            in_channels = 3
        elif cfg.DANET.INPUT_MODE in ['iuv', 'iuv_gt']:
            print('input mode: IUV')
            in_channels = 3 * 25
        elif cfg.DANET.INPUT_MODE in ['iuv_feat', 'iuv_gt_feat']:
            print('input mode: IUV + feat')
            in_channels = 3 * 25 + self.add_feat_ch
        elif cfg.DANET.INPUT_MODE == 'feat':
            print('input mode: feat')
            in_channels = self.add_feat_ch
        elif cfg.DANET.INPUT_MODE == 'seg':
            print('input mode: segmentation')
            in_channels = 25

        num_layers = cfg.DANET.GLO_NUM_LAYERS
        self.Conv_Body = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=1, bias=False),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU(True),
                                       SmplResNet(resnet_nums=num_layers, num_classes=229, in_channels=64)
                                       )

        if pretrained:
            self.Conv_Body[3].init_weights(cfg.MSRES_MODEL[f'PRETRAINED_{cfg.DANET.GLO_NUM_LAYERS}'])

    def forward(self, data):
        return_dict = {}
        return_dict['visualization'] = {}
        return_dict['losses'] = {}

        if cfg.DANET.INPUT_MODE == 'rgb':
            para, _ = self.Conv_Body(data)
        elif cfg.DANET.INPUT_MODE in ['iuv', 'iuv_gt']:
            if data.size(1) == 3:
                Umap, Vmap, Imap, _ = iuv_img2map(data)
                iuv_map = torch.cat([Umap, Vmap, Imap], dim=1)
            else:
                iuv_map = data
            para, _ = self.Conv_Body(iuv_map)
        elif cfg.DANET.INPUT_MODE in ['iuv_feat', 'iuv_gt_feat']:
            para, _ = self.Conv_Body(torch.cat([data['iuv'], data['feat']], dim=1))
        elif cfg.DANET.INPUT_MODE == 'feat':
            para, _ = self.Conv_Body(data['feat'])
        elif cfg.DANET.INPUT_MODE == 'seg':
            para, _ = self.Conv_Body(data['index'])

        return_dict['para'] = para

        return return_dict

    @property
    def detectron_weight_mapping(self):
        d_wmap = {}  # detectron_weight_mapping
        d_orphan = []  # detectron orphan weight list

        new_key = [key for key in self.state_dict().keys()]
        for key in new_key:
            d_wmap[key] = True

        self.mapping_to_detectron = d_wmap
        self.orphans_in_detectron = []

        return self.mapping_to_detectron, self.orphans_in_detectron

class DecomposedPredictor(nn.Module):
    def __init__(self, feat_in_dim=None, mean_params=None, pretrained=True):
        super(DecomposedPredictor, self).__init__()

        # For cache
        self.mapping_to_detectron = None
        self.orphans_in_detectron = None

        self.add_feat_ch = feat_in_dim

        self.smpl_parents = smpl_structure('smpl_parents')
        self.smpl_children = smpl_structure('smpl_children')
        self.dp2smpl_mapping = smpl_structure('dp2smpl_mapping')

        # Backbone for shape prediction
        if cfg.DANET.INPUT_MODE == 'rgb':
            print('input mode: RGB')
            self.in_channels = 3
        elif cfg.DANET.INPUT_MODE in ['iuv', 'iuv_gt']:
            print('input mode: IUV')
            self.in_channels = 3 * (1 + 24)
        elif cfg.DANET.INPUT_MODE in ['iuv_feat', 'iuv_gt_feat']:
            print('input mode: IUV + feat')
            self.in_channels = 3 * (1 + 24) + self.add_feat_ch
        elif cfg.DANET.INPUT_MODE == 'feat':
            print('input mode: feat')
            self.in_channels = self.add_feat_ch
        elif cfg.DANET.INPUT_MODE == 'seg':
            print('input mode: segmentation')
            self.in_channels = 1 + 24

        self.register_buffer('mean_cam_shape', torch.cat(mean_params[:2], dim=1))
        self.register_buffer('mean_pose', mean_params[2])

        num_layers = cfg.DANET.GLO_NUM_LAYERS
        self.body_net = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            SmplResNet(resnet_nums=num_layers, in_channels=64, num_classes=13))

        if pretrained:
            self.body_net[3].init_weights(cfg.MSRES_MODEL[f'PRETRAINED_{num_layers}'])

        self.smpl_children_tree = [[idx for idx, val in enumerate(self.smpl_parents[0]) if val == i] for i in range(24)]

        smpl_chains = []
        for i in range(24):
            chain = [i]
            if i == 0:
                smpl_chains.append(chain)
                continue
            p_i = i
            for j in range(24):
                p_i = self.smpl_parents[0][p_i]
                chain.append(p_i)
                if p_i == 0:
                    smpl_chains.append(chain)
                    break
        self.smpl_chains = smpl_chains

        # for tree architecture
        self.limb_ind = [[0, 3, 6, 9, 12, 15],
                         [13, 16, 18, 20, 22],
                         [14, 17, 19, 21, 23],
                         [1, 4, 7, 10],
                         [2, 5, 8, 11]
                         ]

        limb_ind_flatten = sum(self.limb_ind, [])
        self.limb_ind_mapping = [limb_ind_flatten.index(i) for i in range(len(limb_ind_flatten))]

        self.limb_branch = self.limb_ind
        self.limb_branch_lstm = [self.limb_ind[0][:4],
                                 self.limb_ind[0][4:],
                                 [9] + self.limb_ind[1],
                                 [9] + self.limb_ind[2],
                                 [0] + self.limb_ind[3],
                                 [0] + self.limb_ind[4]
                                 ]

        limb_num_layers = 18
        if cfg.DANET.INPUT_MODE in ['feat']:
            self.limb_net = nn.Sequential(
                nn.Conv2d(self.add_feat_ch, 64, kernel_size=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                SmplResNet(resnet_nums=limb_num_layers, in_channels=64, num_classes=0, truncate=1)
            )
        elif cfg.DANET.INPUT_MODE in ['iuv_feat', 'iuv_gt_feat']:
            self.limb_net = nn.Sequential(
                nn.Conv2d((1 + len(self.dp2smpl_mapping[0])) * 3 + self.add_feat_ch, 64, kernel_size=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                SmplResNet(resnet_nums=limb_num_layers, in_channels=64, num_classes=0, truncate=1)
            )
        else:
            self.limb_net = nn.Sequential(
                nn.Conv2d((1 + len(self.dp2smpl_mapping[0])) * 3, 64, kernel_size=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                SmplResNet(resnet_nums=limb_num_layers, in_channels=64, num_classes=0, truncate=1)
                )

        if pretrained:
            self.limb_net[3].init_weights(cfg.MSRES_MODEL[f'PRETRAINED_{limb_num_layers}'])

        self.rot_feat_len = cfg.DANET.REFINEMENT.FEAT_DIM
        self.pos_feat_len = cfg.DANET.REFINEMENT.FEAT_DIM

        self.limb_reslayer = LimbResLayers(limb_num_layers, inplanes=256, outplanes=self.rot_feat_len, groups=24)

        if cfg.DANET.REFINE_STRATEGY == 'lstm_direct':

            self.limb_lstm = nn.ModuleList()
            for stack_i in range(cfg.DANET.REFINEMENT.STACK_NUM):
                self.limb_lstm.append(nn.ModuleList())
                for br in range(len(self.limb_branch)):
                    self.limb_lstm[stack_i].append(nn.LSTM(self.pos_feat_len, self.pos_feat_len,
                                                           num_layers=1, batch_first=True, bidirectional=True))

            self.pose_regressors = nn.ModuleList()
            self.pose_regressors.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(self.rot_feat_len * 1 * 24, 9 * 24, kernel_size=1, groups=24)
            ))
            self.pose_regressors.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(self.rot_feat_len * 2 * 24, 9 * 24, kernel_size=1, groups=24)
            ))
        elif cfg.DANET.REFINE_STRATEGY == 'lstm':

            self.limb_lstm = nn.ModuleList()
            for stack_i in range(cfg.DANET.REFINEMENT.STACK_NUM):
                self.limb_lstm.append(nn.ModuleList())
                for br in range(len(self.limb_branch)):
                    self.limb_lstm[stack_i].append(nn.LSTM(self.pos_feat_len, self.pos_feat_len,
                                                  num_layers=1, batch_first=True, bidirectional=True))

            self.rot2pos = nn.ModuleList()
            for stack_i in range(cfg.DANET.REFINEMENT.STACK_NUM):
                self.rot2pos.append(nn.ModuleList())
                for i in range(24):
                    self.rot2pos[stack_i].append(
                        nn.Sequential(nn.Conv2d(self.rot_feat_len + self.pos_feat_len, 512, kernel_size=1),
                                      nn.BatchNorm2d(512),
                                      nn.ReLU(True),
                                      nn.Conv2d(512, self.pos_feat_len, kernel_size=1),
                                      nn.BatchNorm2d(self.pos_feat_len),
                                      nn.ReLU(True))
                        )

            self.pos2rot = nn.ModuleList()
            for stack_i in range(cfg.DANET.REFINEMENT.STACK_NUM):
                self.pos2rot.append(nn.Sequential(nn.Conv2d(self.pos_feat_len * 2 * 3, 1024, kernel_size=1),
                                             nn.BatchNorm2d(1024),
                                             nn.ReLU(True),
                                             nn.Conv2d(1024, self.rot_feat_len, kernel_size=1),
                                             nn.BatchNorm2d(self.rot_feat_len),
                                             nn.ReLU(True)
                                             )
                                    )
            if cfg.DANET.REFINEMENT.POS_INTERSUPV:
                self.coord_regressors = nn.ModuleList()
                self.coord_regressors.append(nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(self.pos_feat_len * 1 * 24, 3 * 24, kernel_size=1, groups=24)
                )
                )
                for stack_i in range(cfg.DANET.REFINEMENT.STACK_NUM):
                    self.coord_regressors.append(nn.Sequential(
                        nn.AdaptiveAvgPool2d(1),
                        nn.Conv2d(self.pos_feat_len * 2 * 24, 3 * 24, kernel_size=1, groups=24)
                    )
                    )

            rot_dim = 6 if cfg.DANET.USE_6D_ROT else 9

            self.pose_regressors = nn.ModuleList()
            for stack_i in range(1+cfg.DANET.REFINEMENT.STACK_NUM):
                self.pose_regressors.append(nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(self.rot_feat_len * 24, rot_dim * 24, kernel_size=1, groups=24)
                ))
        elif cfg.DANET.REFINE_STRATEGY in ['gcn', 'gcn_direct']:

            self.rot2pos = nn.ModuleList()
            for i in range(24):
                self.rot2pos.append(
                    nn.Sequential(nn.Conv2d(self.rot_feat_len + self.pos_feat_len, 512, kernel_size=1),
                                  nn.BatchNorm2d(512),
                                  nn.ReLU(True),
                                  nn.Conv2d(512, self.pos_feat_len, kernel_size=1),
                                  nn.BatchNorm2d(self.pos_feat_len),
                                  nn.ReLU(True))
                    )

            self.pos2rot = nn.Sequential(nn.Conv2d(self.pos_feat_len * 3, 1024, kernel_size=1),
                                         nn.BatchNorm2d(1024),
                                         nn.ReLU(True),
                                         nn.Conv2d(1024, self.rot_feat_len, kernel_size=1),
                                         nn.BatchNorm2d(self.rot_feat_len),
                                         nn.ReLU(True)
                                         )

            if cfg.DANET.REFINEMENT.POS_INTERSUPV:
                self.coord_regressors = nn.ModuleList()
                for _ in range(2):
                    self.coord_regressors.append(nn.Sequential(
                        nn.AdaptiveAvgPool2d(1),
                        nn.Conv2d(self.pos_feat_len * 24, 3 * 24, kernel_size=1, groups=24)
                    )
                    )

            rot_dim = 6 if cfg.DANET.USE_6D_ROT else 9

            self.pose_regressors = nn.ModuleList()

            self.pose_regressors.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(self.rot_feat_len * 1 * 24, rot_dim * 24, kernel_size=1, groups=24)
            )
            )
            self.pose_regressors.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(self.rot_feat_len * 24, rot_dim * 24, kernel_size=1, groups=24)
            )
            )
            nn.init.xavier_uniform_(self.pose_regressors[0][1].weight, gain=0.01)
            nn.init.xavier_uniform_(self.pose_regressors[1][1].weight, gain=0.01)

            self.register_buffer('I_n', torch.from_numpy(np.eye(24)).float().unsqueeze(0))

            self.graph = Graph(layout='smpl', norm_type='none')  # uniform   distance  spatial
            A_link = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
            self.register_buffer('A_link', A_link - self.I_n)

            self.edge_act = F.relu

            self.graph_2neigh = Graph(layout='smpl_2neigh', strategy='uniform', norm_type='none')

            A_mask = torch.tensor(self.graph_2neigh.A, dtype=torch.float32, requires_grad=False)

            add_links = [(1, 2), (1, 3), (2, 3), (13, 14), (12, 13), (12, 14)]
            for lk in add_links:
                A_mask[:, lk[0], lk[1]] = 1
                A_mask[:, lk[1], lk[0]] = 1

            A = normalize_undigraph(A_mask)
            self.register_buffer('A', A)

            self.register_buffer('A_mask', A_mask - self.I_n)
            self.edge_importance = nn.Parameter(torch.ones(self.A_link.size()))

            self.refine_gcn = GCN(128, 256, 128, num_layers=int(cfg.DANET.REFINEMENT.GCN_NUM_LAYER), num_nodes=A.size(1), normalize=False)

            r2p_A = np.zeros((24, 24))
            for i in range(24):
                r2p_A[i, self.smpl_chains[i]] = 1
                r2p_A[i, i] = 0

            r2p_A = normalize_digraph(r2p_A, AD_mode=False)
            r2p_A = torch.from_numpy(r2p_A).float().unsqueeze(0)
            self.register_buffer('r2p_A', r2p_A)

            p2r_A = np.zeros((24, 24))
            for i in range(24):
                p2r_A[i, self.smpl_children_tree[i]] = 1
                p2r_A[i, self.smpl_parents[0][i]] = 1
                p2r_A[i, i] = 1

            self.r2p_gcn = GCN(128, 128, 128, num_layers=1, num_nodes=r2p_A.shape[1], normalize=False)

            p2r_A = normalize_digraph(p2r_A, AD_mode=False)
            p2r_A = torch.from_numpy(p2r_A).float().unsqueeze(0)
            self.register_buffer('p2r_A', p2r_A)

            self.p2r_gcn = GCN(128, 128, 128, num_layers=1, num_nodes=p2r_A.shape[1], normalize=False)

    def forward(self, body_iuv, limb_iuv):

        return_dict = {}
        return_dict['visualization'] = {}
        return_dict['losses'] = {}

        if cfg.DANET.INPUT_MODE == 'rgb':
            map_channels = body_iuv.size(1) / 3
            body_u, body_v, body_i = body_iuv[:, :map_channels], body_iuv[:, map_channels:2 * map_channels], body_iuv[:,
                                                                                                             2 * map_channels:3 * map_channels]
            global_para, global_feat = self.body_net(iuv_map2img(body_u, body_v, body_i))
        elif cfg.DANET.INPUT_MODE in ['iuv', 'iuv_gt']:
            global_para, global_feat = self.body_net(body_iuv)
        elif cfg.DANET.INPUT_MODE in ['iuv_feat', 'iuv_gt_feat']:
            global_para, global_feat = self.body_net(torch.cat([body_iuv['iuv'], body_iuv['feat']], dim=1))
        elif cfg.DANET.INPUT_MODE == 'feat':
            global_para, global_feat = self.body_net(body_iuv['feat'])
        elif cfg.DANET.INPUT_MODE == 'seg':
            global_para, global_feat = self.body_net(body_iuv['index'])

        global_para += self.mean_cam_shape

        if cfg.DANET.INPUT_MODE in ['iuv_feat', 'feat', 'iuv_gt_feat']:
            nbs = limb_iuv['pfeat'].size(0)
            limb_mapsize = limb_iuv['pfeat'].size(-1)
        elif cfg.DANET.INPUT_MODE in ['seg']:
            nbs = limb_iuv['pindex'].size(0)
            limb_mapsize = limb_iuv['pindex'].size(-1)
        else:
            nbs = limb_iuv.size(0)
            limb_mapsize = limb_iuv.size(-1)

        if cfg.DANET.INPUT_MODE in ['iuv_feat', 'iuv_gt_feat']:
            limb_iuv_stacked = limb_iuv['piuv'].view(nbs * 24, -1, limb_mapsize, limb_mapsize)
            limb_feat_stacked = limb_iuv['pfeat'].view(nbs * 24, -1, limb_mapsize, limb_mapsize)
            _, limb_feat = self.limb_net(torch.cat([limb_iuv_stacked, limb_feat_stacked], dim=1))
        elif cfg.DANET.INPUT_MODE in ['iuv', 'iuv_gt']:
            limb_iuv_stacked = limb_iuv.view(nbs * 24, -1, limb_mapsize, limb_mapsize)
            _, limb_feat = self.limb_net(limb_iuv_stacked)
        if cfg.DANET.INPUT_MODE == 'feat':
            limb_feat_stacked = limb_iuv['pfeat'].view(nbs * 24, -1, limb_mapsize, limb_mapsize)
            _, limb_feat = self.limb_net(limb_feat_stacked)
        if cfg.DANET.INPUT_MODE == 'seg':
            limb_feat_stacked = limb_iuv['pindex'].view(nbs * 24, -1, limb_mapsize, limb_mapsize)
            _, limb_feat = self.limb_net(limb_feat_stacked)

        limb_feat = limb_feat['x4']
        limb_feat = self.limb_reslayer(limb_feat.view(nbs, -1, limb_feat.size(-2), limb_feat.size(-1)))

        rot_feats = limb_feat.view(nbs, 24, -1, limb_feat.size(-2), limb_feat.size(-1))

        if cfg.DANET.REFINE_STRATEGY == 'lstm_direct':

            return_dict['joint_rotation'] = []

            local_para = self.pose_regressors[0](rot_feats.view(rot_feats.size(0), 24 * rot_feats.size(2), 1, 1)).view(
                nbs, 24, -1)
            smpl_pose = local_para.view(local_para.size(0), -1)
            return_dict['joint_rotation'].append(smpl_pose)

            for s_i in range(cfg.DANET.REFINEMENT.STACK_NUM):
                pos_feats = {}
                for i in range(24):
                    pos_feats[i] = rot_feats[:, i]

                pos_feats_refined = {}
                for br in range(len(self.limb_branch_lstm)):
                    pos_feat_in = torch.stack([pos_feats[ind] for ind in self.limb_branch_lstm[br]], dim=1)
                    pos_feat_in = pos_feat_in.squeeze(-1).squeeze(-1)
                    if br == 0:
                        lstm_out, hidden_feat = self.limb_lstm[s_i][0](pos_feat_in)
                    elif br == 1:
                        lstm_out, _ = self.limb_lstm[s_i][0](pos_feat_in, hidden_feat)
                    elif br in [2, 3]:
                        lstm_out, _ = self.limb_lstm[s_i][br - 1](pos_feat_in, hidden_feat)
                    else:
                        lstm_out, _ = self.limb_lstm[s_i][br - 1](pos_feat_in)
                    for i, ind in enumerate(self.limb_branch_lstm[br]):
                        if ind == 0 and br != 0:
                            continue
                        pos_feats_refined[ind] = lstm_out[:, i].unsqueeze(-1).unsqueeze(-1)

                # update
                for i in range(24):
                    pos_feats[i] = pos_feats[i].repeat(1, 2, 1, 1) + pos_feats_refined[i]

                refined_feat = torch.stack([pos_feats[i] for i in range(24)], dim=1)
                part_feats = refined_feat.view(refined_feat.size(0), 24 * refined_feat.size(2), 1, 1)

                local_para = self.pose_regressors[s_i + 1](part_feats).view(nbs, 24, -1)
                smpl_pose = local_para.view(local_para.size(0), -1)


        elif cfg.DANET.REFINE_STRATEGY == 'lstm':

            return_dict['joint_position'] = []
            return_dict['joint_rotation'] = []

            if self.training:
                local_para = self.pose_regressors[0](rot_feats.view(rot_feats.size(0), 24*rot_feats.size(2), 1, 1)).view(nbs, 24, -1)
                smpl_pose = local_para.view(local_para.size(0), -1)
                smpl_pose += self.mean_pose
                if cfg.DANET.USE_6D_ROT:
                    smpl_pose = rot6d_to_rotmat(smpl_pose).view(local_para.size(0), -1)
                return_dict['joint_rotation'].append(smpl_pose)

            rot_feats_before = rot_feats

            for s_i in range(cfg.DANET.REFINEMENT.STACK_NUM):
                pos_feats = {}

                pos_feats[0] = rot_feats_before[:, 0]
                for br in range(len(self.limb_branch)):
                    for ind in self.limb_branch[br]:
                        p_ind = self.smpl_parents[0][ind]
                        pos_rot_feat_cat = torch.cat([pos_feats[p_ind], rot_feats_before[:, p_ind]], dim=1)
                        pos_feats[ind] = self.rot2pos[s_i][ind](pos_rot_feat_cat)

                if self.training:
                    if cfg.DANET.JOINT_POSITION_WEIGHTS > 0 and cfg.DANET.REFINEMENT.POS_INTERSUPV:
                        coord_feats = torch.cat([pos_feats[i] for i in range(24)], dim=1)
                        smpl_coord = self.coord_regressors[s_i](coord_feats).view(nbs, 24, -1)
                        return_dict['joint_position'].append(smpl_coord)

                pos_feats_refined = {}
                for br in range(len(self.limb_branch_lstm)):
                    pos_feat_in = torch.stack([pos_feats[ind] for ind in self.limb_branch_lstm[br]], dim=1)
                    pos_feat_in = pos_feat_in.squeeze(-1).squeeze(-1)
                    if br == 0:
                        lstm_out, hidden_feat = self.limb_lstm[s_i][0](pos_feat_in)
                    elif br == 1:
                        lstm_out, _ = self.limb_lstm[s_i][0](pos_feat_in, hidden_feat)
                    elif br in [2, 3]:
                        lstm_out, _ = self.limb_lstm[s_i][br - 1](pos_feat_in, hidden_feat)
                    else:
                        lstm_out, _ = self.limb_lstm[s_i][br - 1](pos_feat_in)
                    for i, ind in enumerate(self.limb_branch_lstm[br]):
                        if ind == 0 and br != 0:
                            continue
                        pos_feats_refined[ind] = lstm_out[:, i].unsqueeze(-1).unsqueeze(-1)

                # update
                for i in range(24):
                    pos_feats[i] = pos_feats[i].repeat(1, 2, 1, 1) + pos_feats_refined[i]

                if self.training:
                    if cfg.DANET.JOINT_POSITION_WEIGHTS > 0 and cfg.DANET.REFINEMENT.POS_INTERSUPV:
                        coord_feats = torch.cat([pos_feats[i] for i in range(24)], dim=1)
                        smpl_coord = self.coord_regressors[s_i+1](coord_feats).view(nbs, 24, -1)
                        return_dict['joint_position'].append(smpl_coord)

                tri_pos_feats = [
                    torch.cat([pos_feats[self.smpl_parents[0][i]], pos_feats[i], pos_feats[self.smpl_children[1][i]]], dim=1)
                    for i in range(24)]
                tri_pos_feats = torch.cat(tri_pos_feats, dim=0)
                tran_rot_feats = self.pos2rot[s_i](tri_pos_feats)
                tran_rot_feats = tran_rot_feats.view(24, nbs, -1, tran_rot_feats.size(-2),
                                                     tran_rot_feats.size(-1))
                tran_rot_feats = tran_rot_feats.transpose(0, 1)

                part_feats = tran_rot_feats.contiguous().view(tran_rot_feats.size(0), 24 * tran_rot_feats.size(2), 1, 1)

                local_para = self.pose_regressors[s_i+1](part_feats).view(nbs, 24, -1)
                smpl_pose = local_para.view(local_para.size(0), -1)
                smpl_pose += self.mean_pose
                if cfg.DANET.USE_6D_ROT:
                    smpl_pose = rot6d_to_rotmat(smpl_pose).view(local_para.size(0), -1)

        elif cfg.DANET.REFINE_STRATEGY == 'gcn':

            return_dict['joint_position'] = []
            return_dict['joint_rotation'] = []

            if self.training:
                local_para = self.pose_regressors[0](rot_feats.view(rot_feats.size(0), 24 * rot_feats.size(2), 1, 1)).view(
                    nbs, 24, -1)
                smpl_pose = local_para.view(local_para.size(0), -1)
                smpl_pose += self.mean_pose
                if cfg.DANET.USE_6D_ROT:
                    smpl_pose = rot6d_to_rotmat(smpl_pose).view(local_para.size(0), -1)
                return_dict['joint_rotation'].append(smpl_pose)

            rot_feats_before = rot_feats

            rot_feats_init = rot_feats_before.squeeze(-1).squeeze(-1)
            pos_feats_init = self.r2p_gcn(rot_feats_init, self.r2p_A[0])

            if self.training:
                if cfg.DANET.JOINT_POSITION_WEIGHTS > 0 and cfg.DANET.REFINEMENT.POS_INTERSUPV:
                    coord_feats0 = pos_feats_init.unsqueeze(2).view(pos_feats_init.size(0), pos_feats_init.size(-1) * 24, 1, 1)
                    smpl_coord0 = self.coord_regressors[0](coord_feats0).view(nbs, 24, -1)
                    return_dict['joint_position'].append(smpl_coord0)

            if cfg.DANET.REFINEMENT.REFINE_ON:
                graph_A = self.A_mask * self.edge_act(self.edge_importance)
                norm_graph_A = normalize_undigraph(self.I_n[0] + graph_A)[0]
                l_pos_feat = self.refine_gcn(pos_feats_init, norm_graph_A)
                l_pos_feat = pos_feats_init + l_pos_feat

                pos_feats_refined = l_pos_feat

                if self.training:
                    if cfg.DANET.JOINT_POSITION_WEIGHTS > 0 and cfg.DANET.REFINEMENT.POS_INTERSUPV:
                        coord_feats1 = pos_feats_refined.unsqueeze(2).view(pos_feats_refined.size(0),
                                                                           pos_feats_refined.size(-1) * 24, 1, 1)
                        smpl_coord1 = self.coord_regressors[1](coord_feats1).view(nbs, 24, -1)
                        return_dict['joint_position'].append(smpl_coord1)
            else:
                pos_feats_refined = pos_feats_init

            rot_feats_refined = self.p2r_gcn(pos_feats_refined, self.p2r_A[0])

            tran_rot_feats = rot_feats_refined.unsqueeze(-1).unsqueeze(-1)
            part_feats = tran_rot_feats.view(tran_rot_feats.size(0), 24 * tran_rot_feats.size(2), 1, 1)

            local_para = self.pose_regressors[-1](part_feats).view(nbs, 24, -1)
            smpl_pose = local_para.view(local_para.size(0), -1)
            smpl_pose += self.mean_pose
            if cfg.DANET.USE_6D_ROT:
                smpl_pose = rot6d_to_rotmat(smpl_pose).view(local_para.size(0), -1)

        elif cfg.DANET.REFINE_STRATEGY == 'gcn_direct':

            return_dict['joint_position'] = []
            return_dict['joint_rotation'] = []

            local_para = self.pose_regressors[0](rot_feats.view(rot_feats.size(0), 24 * rot_feats.size(2), 1, 1)).view(
                nbs, 24, -1)
            smpl_pose = local_para.view(local_para.size(0), -1)

            if cfg.DANET.REFINEMENT.REFINE_ON:
                return_dict['joint_rotation'].append(smpl_pose)

                pos_feats_init = rot_feats.squeeze(-1).squeeze(-1)

                graph_A = self.A_mask * self.edge_act(self.edge_importance)
                norm_graph_A = normalize_undigraph(self.I_n[0] + graph_A)[0]
                l_pos_feat = self.refine_gcn(pos_feats_init, norm_graph_A)
                l_pos_feat = pos_feats_init + l_pos_feat

                pos_feats_refined = l_pos_feat
                tran_rot_feats = pos_feats_refined.unsqueeze(-1).unsqueeze(-1)

                part_feats = tran_rot_feats.view(tran_rot_feats.size(0), 24 * tran_rot_feats.size(2), 1, 1)

                local_para = self.pose_regressors[-1](part_feats).view(nbs, 24, -1)
                smpl_pose = local_para.view(local_para.size(0), -1)

        para = torch.cat([global_para, smpl_pose], dim=1)

        return_dict['para'] = para

        return return_dict

    @property
    def detectron_weight_mapping(self):
        d_wmap = {}  # detectron_weight_mapping
        d_orphan = []  # detectron orphan weight list

        new_key = [key for key in self.state_dict().keys()]
        for key in new_key:
            d_wmap[key] = True

        self.mapping_to_detectron = d_wmap
        self.orphans_in_detectron = []

        return self.mapping_to_detectron, self.orphans_in_detectron
