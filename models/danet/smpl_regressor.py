from functools import wraps
import torch
from models.core.config import cfg
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import neural_renderer as nr
from neural_renderer.projection import projection as nr_projection

from utils.densepose_methods import DensePoseMethods
from models.module.GCN import GCN
from utils.graph import Graph, normalize_digraph, normalize_undigraph
from utils.iuvmap import iuv_img2map, iuv_map2img
from utils.geometry import rot6d_to_rotmat
from utils.smpl_utlis import smpl_structure

from models.smpl import SMPL
from models.module.res_module import SmplResNet, LimbResLayers

import path_config


def check_inference(net_func):
    """
    Decorator to check if the wrapped wrapped.

    Args:
        net_func: (todo): write your description
    """
    @wraps(net_func)
    def wrapper(self, *args, **kwargs):
        """
        Decorator to wrap a function.

        Args:
            self: (todo): write your description
        """
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
    def __init__(self, options, as_renderer_only=False, orig_size=224, feat_in_dim=None, smpl_mean_params=None):
        """
        Initialize the class

        Args:
            self: (todo): write your description
            options: (dict): write your description
            as_renderer_only: (bool): write your description
            orig_size: (int): write your description
            feat_in_dim: (int): write your description
            smpl_mean_params: (dict): write your description
        """
        super(SMPL_Regressor, self).__init__()

        self.mapping_to_detectron = None
        self.orphans_in_detectron = None
        self.orig_size = orig_size
        self.focal_length = 5000.

        self.options = options

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

        K = np.array([[self.focal_length, 0., self.orig_size / 2.],
                      [0., self.focal_length, self.orig_size / 2.],
                      [0., 0., 1.]])

        R = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])

        t = np.array([0, 0, 5])

        if self.orig_size != 224:
            rander_scale = self.orig_size / float(224)
            K[0, 0] *= rander_scale
            K[1, 1] *= rander_scale
            K[0, 2] *= rander_scale
            K[1, 2] *= rander_scale

        self.K = torch.FloatTensor(K[None, :, :])
        self.R = torch.FloatTensor(R[None, :, :])
        self.t = torch.FloatTensor(t[None, None, :])

        self.smpl = SMPL(path_config.SMPL_MODEL_DIR, batch_size=self.options.batch_size, create_transl=False)

        self.coco_plus2coco = [14, 15, 16, 17, 18, 9, 8, 10, 7, 11, 6, 3, 2, 4, 1, 5, 0]

        DP = DensePoseMethods()

        vert_mapping = DP.All_vertices.astype('int64') - 1
        self.vert_mapping = torch.from_numpy(vert_mapping)

        faces = DP.FacesDensePose
        faces = faces[None, :, :]
        self.faces = torch.from_numpy(faces.astype(np.int32))

        num_part = float(np.max(DP.FaceIndices))
        textures = np.array(
            [(DP.FaceIndices[i] / num_part, np.mean(DP.U_norm[v]), np.mean(DP.V_norm[v])) for i, v in
             enumerate(DP.FacesDensePose)])

        self.VertIndices = {}
        self.VertU = {}
        self.VertV = {}
        self.VertUV = {}
        for i in range(24):
            self.VertIndices[i] = np.int32(np.unique(DP.FacesDensePose[DP.FaceIndices == i + 1]))
            self.VertU[i] = DP.U_norm[self.VertIndices[i]]
            self.VertV[i] = DP.V_norm[self.VertIndices[i]]
            self.VertUV[i] = np.vstack((self.VertU[i], self.VertV[i])).T

        textures = textures[None, :, None, None, None, :]
        self.textures = torch.from_numpy(textures.astype(np.float32))

        self.renderer = nr.Renderer(camera_mode='projection', image_size=cfg.DANET.HEATMAP_SIZE, fill_back=False, anti_aliasing=False,
                                    dist_coeffs=torch.FloatTensor([[0.] * 5]), orig_size=self.orig_size)
        self.renderer.light_intensity_directional = 0.0
        self.renderer.light_intensity_ambient = 1.0

        if not as_renderer_only:
            if cfg.DANET.DECOMPOSED:
                print('using decomposed predictor.')
                self.smpl_para_Outs = DecomposedPredictor(feat_in_dim, init_params)
            else:
                print('using global predictor.')
                self.smpl_para_Outs = GlobalPredictor(feat_in_dim)

    @check_inference
    def smpl_infer_net(self, in_dict):
        """For inference"""
        in_dict['infer_mode'] = True
        return_dict = self._forward(in_dict)

        return return_dict

    def forward(self, in_dict):
        """
        Forward forward forward forward

        Args:
            self: (todo): write your description
            in_dict: (dict): write your description
        """
        if cfg.PYTORCH_VERSION_LESS_THAN_040:
            return self._forward(in_dict)
        else:
            with torch.set_grad_enabled(self.training):
                return self._forward(in_dict)

    def _forward(self, in_dict):
        """
        Returns a dictionary for a dictionary.

        Args:
            self: (todo): write your description
            in_dict: (dict): write your description
        """
        iuv_map = in_dict['iuv_map']
        part_iuv_map = in_dict['part_iuv_map'] if 'part_iuv_map' in in_dict else None
        target = in_dict['target'] if 'target' in in_dict else None
        target_kps = in_dict['target_kps'] if 'target_kps' in in_dict else None
        target_kps3d = in_dict['target_kps3d'] if 'target_kps3d' in in_dict else None
        has_kp3d = in_dict['has_kp3d'] if 'has_kp3d' in in_dict else None
        target_verts = in_dict['target_verts'] if 'target_verts' in in_dict else None
        infer_mode = in_dict['infer_mode'] if 'infer_mode' in in_dict else False
        has_smpl = in_dict['has_smpl'] if 'has_smpl' in in_dict else None

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

        # Training
        para = smpl_out_dict['para']

        for k, v in smpl_out_dict['visualization'].items():
            return_dict['visualization'][k] = v

        if cfg.DANET.ORTHOGONAL_WEIGHTS > 0:
            loss_orth = self.orthogonal_loss(para)
            loss_orth *= cfg.DANET.ORTHOGONAL_WEIGHTS
            return_dict['losses']['Rs_orth'] = loss_orth

        return_dict['prediction']['cam'] = para[:, :3]
        return_dict['prediction']['shape'] = para[:, 3:13]
        return_dict['prediction']['pose'] = para[:, 13:].reshape(-1, 24, 3, 3).contiguous()

        if target is not None:
            if cfg.DANET.POSE_PARA_WEIGHTS > 0:
                loss_para_global = self.smpl_para_losses(para[:, :13], target[:, :13], has_smpl)
                loss_para_global *= cfg.DANET.SHAPE_PARA_WEIGHTS
                return_dict['losses']['smpl_shape'] = loss_para_global

                loss_para_pose = self.smpl_para_losses(para[:, 13:], target[:, 13:], has_smpl)
                loss_para_pose *= cfg.DANET.POSE_PARA_WEIGHTS
                return_dict['losses']['smpl_pose'] = loss_para_pose

                if 'smpl_pose' in smpl_out_dict:
                    for stack_i in range(len(smpl_out_dict['smpl_pose'])):
                        loss_rot = self.smpl_para_losses(smpl_out_dict['smpl_pose'][stack_i], target[:, 13:], has_smpl)
                        loss_rot *= cfg.DANET.POSE_PARA_WEIGHTS
                        return_dict['losses']['smpl_rotation'+str(stack_i)] = loss_rot

            if cfg.DANET.DECOMPOSED and ('smpl_coord' in smpl_out_dict) and cfg.DANET.SMPL_KPS_WEIGHTS > 0:
                gt_beta = target[:, 3:13].contiguous().detach()
                gt_Rs = target[:, 13:].contiguous().view(-1, 24, 3, 3).detach()
                smpl_pts = self.smpl(betas=gt_beta, body_pose=gt_Rs[:, 1:],
                                     global_orient=gt_Rs[:, 0].unsqueeze(1), pose2rot=False)
                # smpl_pts = self.smpl(gt_beta, Rs=gt_Rs, get_skin=False, add_smpl_joint=True)
                gt_smpl_coord = smpl_pts.smpl_joints
                for stack_i in range(len(smpl_out_dict['smpl_coord'])):
                    if torch.sum(has_smpl) > 0:
                        loss_smpl_coord = F.l1_loss(smpl_out_dict['smpl_coord'][stack_i][has_smpl==1], gt_smpl_coord[has_smpl==1],
                                                    size_average=False) / gt_smpl_coord[has_smpl==1].size(0)
                        loss_smpl_coord *= cfg.DANET.SMPL_KPS_WEIGHTS
                    else:
                        loss_smpl_coord = torch.zeros(1).to(para.device)
                    return_dict['losses']['smpl_position'+str(stack_i)] = loss_smpl_coord

        if target_kps is not None:
            if isinstance(target_kps, np.ndarray):
                target_kps = torch.from_numpy(target_kps).cuda(device_id)
            target_kps_vis = target_kps[:, :, -1].unsqueeze(-1).expand(-1, -1, 2)

            cam_gt = target[:, :3].detach() if target is not None else None
            shape_gt = target[:, 3:13].detach() if target is not None else None
            pose_gt = target[:, 13:].detach() if target is not None else None
            loss_proj_kps, loss_kps3d, loss_verts, proj_kps = self.projection_losses(para, target_kps, target_kps_vis, target_kps3d[:, :, :3], has_kp3d, target_verts, cam_gt=cam_gt, shape_gt=shape_gt, pose_gt=pose_gt)

            if cfg.DANET.PROJ_KPS_WEIGHTS > 0:
                loss_proj_kps *= cfg.DANET.PROJ_KPS_WEIGHTS
                return_dict['losses']['proj_kps'] = loss_proj_kps

            if cfg.DANET.KPS3D_WEIGHTS > 0 and loss_kps3d is not None:
                loss_kps3d *= cfg.DANET.KPS3D_WEIGHTS
                return_dict['losses']['kps_3d'] = loss_kps3d

            if loss_verts is not None and cfg.DANET.VERTS_WEIGHTS > 0:
                loss_verts *= cfg.DANET.VERTS_WEIGHTS
                return_dict['losses']['smpl_verts'] = loss_verts

        if cfg.DANET.ORTHOGONAL_WEIGHTS > 0:
            return_dict['metrics']['none'] = loss_orth.detach()

        # pytorch0.4 bug on gathering scalar(0-dim) tensors
        for k, v in return_dict['losses'].items():
            if len(v.shape) == 0:
                return_dict['losses'][k] = v.unsqueeze(0)
        for k, v in return_dict['metrics'].items():
            if len(v.shape) == 0:
                return_dict['metrics'][k] = v.unsqueeze(0)

        return return_dict

    def smpl_para_losses(self, pred, target, has_smpl):
        """
        R compute the sum of the loss.

        Args:
            self: (todo): write your description
            pred: (todo): write your description
            target: (todo): write your description
            has_smpl: (todo): write your description
        """

        if torch.sum(has_smpl) > 0:
            para_loss = F.l1_loss(pred[has_smpl==1], target[has_smpl==1])
        else:
            para_loss = torch.zeros(1).to(pred.device)

        # return F.smooth_l1_loss(pred, target)
        return para_loss

    def orthogonal_loss(self, para):
        """
        Return orthogonal orthogonal orthogonal tensor.

        Args:
            self: (todo): write your description
            para: (todo): write your description
        """
        device_id = para.get_device()
        Rs_pred = para[:, 13:].contiguous().view(-1, 3, 3)
        Rs_pred_transposed = torch.transpose(Rs_pred, 1, 2)
        Rs_mm = torch.bmm(Rs_pred, Rs_pred_transposed)
        tensor_eyes = torch.eye(3).expand_as(Rs_mm).cuda(device_id)
        return F.mse_loss(Rs_mm, tensor_eyes)

    def projection_losses(self, para, target_kps, target_kps_vis=None, target_kps3d=None, has_kp3d=None, target_verts=None, cam_gt=None, cam_t_gt=None, shape_gt=None, pose_gt=None):
        """
        Projection loss loss.

        Args:
            self: (todo): write your description
            para: (todo): write your description
            target_kps: (todo): write your description
            target_kps_vis: (todo): write your description
            target_kps3d: (todo): write your description
            has_kp3d: (array): write your description
            target_verts: (todo): write your description
            cam_gt: (int): write your description
            cam_t_gt: (todo): write your description
            shape_gt: (int): write your description
            pose_gt: (bool): write your description
        """
        device_id = para.get_device()

        batch_size = para.size(0)

        def weighted_l1_loss(input, target, weights=1, size_average=True):
            """
            Mean loss.

            Args:
                input: (array): write your description
                target: (todo): write your description
                weights: (array): write your description
                size_average: (int): write your description
            """
            out = torch.abs(input - target)
            out = out * weights
            if size_average:
                return out.sum() / len(input)
            else:
                return out.sum()

        if cfg.DANET.GTCAM_FOR_REPJ and cam_gt is not None:
            cam = cam_gt
        else:
            cam = para[:, 0:3].contiguous()

        if cfg.DANET.GTSHAPE_FOR_REPJ and shape_gt is not None:
            beta_gt = shape_gt
            Rs_gt = pose_gt.view(-1, 24, 3, 3)

        beta = para[:, 3:13].contiguous()
        Rs = para[:, 13:].contiguous().view(-1, 24, 3, 3)

        M = cfg.DANET.HEATMAP_SIZE

        K, R, t = self.camera_matrix(cam)
        dist_coeffs = torch.FloatTensor([[0.] * 5]).cuda(device_id)

        if cfg.DANET.GTSHAPE_FOR_REPJ and shape_gt is not None and pose_gt is not None:
            ps_dict = self.smpl2kps(beta_gt, Rs, K, R, t, dist_coeffs)
            verts_pose, proj_kps = ps_dict['vertices'], ps_dict['proj_kps']
            ps_dict_shape = self.smpl2kps(beta, Rs_gt, K, R, t, dist_coeffs)
            verts_shape = ps_dict_shape['vertices']
        else:
            ps_dict = self.smpl2kps(beta, Rs, K, R, t, dist_coeffs)
            verts, proj_kps = ps_dict['vertices'], ps_dict['proj_kps']

        if target_kps.size(1) == 14:
            proj_kps_pred = proj_kps[:, :14, :]
        elif target_kps.size(1) == 17:
            proj_kps_pred = proj_kps[:, self.coco_plus2coco, :]

        if target_kps3d is not None and cfg.DANET.KPS3D_WEIGHTS > 0 and torch.sum(has_kp3d==1) > 0:
            kps3d_from_smpl = ps_dict['cocoplus_kps'][:, :14]
            target_kps3d -= torch.mean(target_kps3d[:, [2, 3]], dim=1).unsqueeze(1)
            if has_kp3d is None:
                loss_kps3d = weighted_l1_loss(kps3d_from_smpl, target_kps3d)
            else:
                loss_kps3d = weighted_l1_loss(kps3d_from_smpl[has_kp3d==1], target_kps3d[has_kp3d==1])
        else:
            loss_kps3d = None

        if target_verts is not None and cfg.DANET.VERTS_WEIGHTS > 0:
            if cfg.DANET.GTSHAPE_FOR_REPJ:
                loss_verts = F.mse_loss(verts_shape, target_verts) + F.mse_loss(verts_pose, target_verts)
            else:
                loss_verts = F.mse_loss(verts, target_verts)
        else:
            loss_verts = None

        if target_kps_vis is None:
            target_kps_vis = torch.ones(1).cuda(device_id).expand_as(target_kps[:, :, :2])

        loss_proj_kps = weighted_l1_loss(proj_kps_pred[:, :, :2] / float(M),
                                         target_kps[:, :, :2] / float(cfg.DANET.INIMG_SIZE), target_kps_vis)

        return loss_proj_kps, loss_kps3d, loss_verts, proj_kps

    def smpl2kps(self, beta, Rs, K, R, t, dist_coeffs, add_smpl_joint=False, theta=None, orig_size=None, selected_ind=None):
        """
        Smpl2kps : paramater : param beta : : param beta : : param beta : : param beta : : param beta : : param

        Args:
            self: (todo): write your description
            beta: (float): write your description
            Rs: (array): write your description
            K: (array): write your description
            R: (array): write your description
            t: (array): write your description
            dist_coeffs: (str): write your description
            add_smpl_joint: (array): write your description
            theta: (float): write your description
            orig_size: (int): write your description
            selected_ind: (todo): write your description
        """
        return_dict = {}
        if orig_size is None:
            orig_size = self.orig_size
        if theta is None:
            smpl_pts = self.smpl(betas=beta, body_pose=Rs[:, 1:],
                                 global_orient=Rs[:, 0].unsqueeze(1), pose2rot=False)

        verts = smpl_pts.vertices
        # kps = smpl_pts.cocoplus
        kps = smpl_pts.joints_J19
        return_dict['cocoplus_kps'] = kps
        if add_smpl_joint:
            joint3d_smpl = smpl_pts.smpl_joints
        else:
            joint3d_smpl = None
        if selected_ind is not None:
            kps = kps[:, selected_ind]
        proj_kps = nr_projection(kps, K=K, R=R, t=t, dist_coeffs=dist_coeffs, orig_size=orig_size)
        proj_kps[:, :, 1] *= -1
        proj_kps[:, :, :2] *= cfg.DANET.HEATMAP_SIZE / 2.
        proj_kps[:, :, :2] += cfg.DANET.HEATMAP_SIZE / 2.

        return_dict['vertices'] = verts
        return_dict['proj_kps'] = proj_kps
        return_dict['joint3d_smpl'] = joint3d_smpl

        return return_dict

    def verts2uvimg(self, verts, cam, f=None, tran=None):
        """
        Convert image touv image

        Args:
            self: (todo): write your description
            verts: (str): write your description
            cam: (todo): write your description
            f: (todo): write your description
            tran: (todo): write your description
        """
        batch_size = verts.size(0)

        K, R, t = self.camera_matrix(cam)

        if self.vert_mapping is None:
            vertices = verts
        else:
            vertices = verts[:, self.vert_mapping, :]

        iuv_image = self.renderer(vertices, self.faces.to(verts.device).expand(batch_size, -1, -1),
                               self.textures.to(verts.device).expand(batch_size, -1, -1, -1, -1, -1).clone(),
                               K=K, R=R, t=t,
                               mode='rgb',
                               dist_coeffs=torch.FloatTensor([[0.] * 5]).to(verts.device))

        return iuv_image

    def camera_matrix(self, cam):
        """
        Build a camera matrix.

        Args:
            self: (todo): write your description
            cam: (todo): write your description
        """
        batch_size = cam.size(0)

        K = self.K.repeat(batch_size, 1, 1)
        R = self.R.repeat(batch_size, 1, 1)
        # t = self.t.repeat(batch_size, 1, 1)
        t = torch.stack([cam[:, 1], cam[:, 2], 2 * self.focal_length/(self.orig_size * cam[:, 0] + 1e-9)], dim=-1)
        t = t.unsqueeze(1)

        if cam.is_cuda:
            device_id = cam.get_device()
            K = K.cuda(device_id)
            R = R.cuda(device_id)
            t = t.cuda(device_id)

        return K, R, t

    # @property
    def detectron_weight_mapping(self):
        """
        R find the weight of the weight of the weight states.

        Args:
            self: (todo): write your description
        """
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
    def __init__(self, feat_in_dim=None):
        """
        Initialize the neural network.

        Args:
            self: (todo): write your description
            feat_in_dim: (int): write your description
        """
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
            # self.add_feat_ch = 48
            in_channels = self.add_feat_ch
        elif cfg.DANET.INPUT_MODE == 'seg':
            print('input mode: segmentation')
            in_channels = 25

        num_layers = cfg.DANET.GLO_NUM_LAYERS
        # num_layers = 50
        self.Conv_Body = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=1, bias=False),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU(True),
                                       SmplResNet(resnet_nums=num_layers, num_classes=229, in_channels=64)
                                       )

        if not cfg.DANET.EVAL_MODE:
            if cfg.DANET.GLO_NUM_LAYERS == 18:
                self.Conv_Body[3].init_weights('data/pretrained_model/resnet18-5c106cde.pth')
            elif cfg.DANET.GLO_NUM_LAYERS == 50:
                self.Conv_Body[3].init_weights('data/pretrained_model/resnet50-19c8e357.pth')
            elif cfg.DANET.GLO_NUM_LAYERS == 101:
                self.Conv_Body[3].init_weights('data/pretrained_model/resnet101-5d3b4d8f.pth')

    def forward(self, data):
        """
        Forward forward forward

        Args:
            self: (todo): write your description
            data: (array): write your description
        """
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
        """
        Return the weight mapping of the weight of the weight matrix.

        Args:
            self: (todo): write your description
        """
        d_wmap = {}  # detectron_weight_mapping
        d_orphan = []  # detectron orphan weight list

        new_key = [key for key in self.state_dict().keys()]
        for key in new_key:
            d_wmap[key] = True

        self.mapping_to_detectron = d_wmap
        self.orphans_in_detectron = []

        return self.mapping_to_detectron, self.orphans_in_detectron

class DecomposedPredictor(nn.Module):
    def __init__(self, feat_in_dim=None, mean_params=None):
        """
        Initialize the network.

        Args:
            self: (todo): write your description
            feat_in_dim: (int): write your description
            mean_params: (dict): write your description
        """
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

        if not cfg.DANET.EVAL_MODE:
            if num_layers == 18:
                self.body_net[3].init_weights('data/pretrained_model/resnet18-5c106cde.pth')
            elif num_layers == 50:
                self.body_net[3].init_weights('data/pretrained_model/resnet50-19c8e357.pth')

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

        if not cfg.DANET.EVAL_MODE:
            if limb_num_layers == 18:
                self.limb_net[3].init_weights('data/pretrained_model/resnet18-5c106cde.pth')
            elif limb_num_layers == 50:
                self.limb_net[3].init_weights('data/pretrained_model/resnet50-19c8e357.pth')

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
        """
        Forward forward forward

        Args:
            self: (todo): write your description
            body_iuv: (todo): write your description
            limb_iuv: (todo): write your description
        """

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

            return_dict['smpl_pose'] = []

            local_para = self.pose_regressors[0](rot_feats.view(rot_feats.size(0), 24 * rot_feats.size(2), 1, 1)).view(
                nbs, 24, -1)
            smpl_pose = local_para.view(local_para.size(0), -1)
            return_dict['smpl_pose'].append(smpl_pose)

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

            return_dict['smpl_coord'] = []
            return_dict['smpl_pose'] = []

            if self.training:
                local_para = self.pose_regressors[0](rot_feats.view(rot_feats.size(0), 24*rot_feats.size(2), 1, 1)).view(nbs, 24, -1)
                smpl_pose = local_para.view(local_para.size(0), -1)
                smpl_pose += self.mean_pose
                if cfg.DANET.USE_6D_ROT:
                    smpl_pose = rot6d_to_rotmat(smpl_pose).view(local_para.size(0), -1)
                return_dict['smpl_pose'].append(smpl_pose)

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
                    if cfg.DANET.SMPL_KPS_WEIGHTS > 0 and cfg.DANET.REFINEMENT.POS_INTERSUPV:
                        coord_feats = torch.cat([pos_feats[i] for i in range(24)], dim=1)
                        smpl_coord = self.coord_regressors[s_i](coord_feats).view(nbs, 24, -1)
                        return_dict['smpl_coord'].append(smpl_coord)

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
                    if cfg.DANET.SMPL_KPS_WEIGHTS > 0 and cfg.DANET.REFINEMENT.POS_INTERSUPV:
                        coord_feats = torch.cat([pos_feats[i] for i in range(24)], dim=1)
                        smpl_coord = self.coord_regressors[s_i+1](coord_feats).view(nbs, 24, -1)
                        return_dict['smpl_coord'].append(smpl_coord)

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

            return_dict['smpl_coord'] = []
            return_dict['smpl_pose'] = []

            if self.training:
                local_para = self.pose_regressors[0](rot_feats.view(rot_feats.size(0), 24 * rot_feats.size(2), 1, 1)).view(
                    nbs, 24, -1)
                smpl_pose = local_para.view(local_para.size(0), -1)
                smpl_pose += self.mean_pose
                if cfg.DANET.USE_6D_ROT:
                    smpl_pose = rot6d_to_rotmat(smpl_pose).view(local_para.size(0), -1)
                # smpl_pose = local_para[:, self.limb_ind_mapping].view(local_para.size(0), -1)
                return_dict['smpl_pose'].append(smpl_pose)

            rot_feats_before = rot_feats

            rot_feats_init = rot_feats_before.squeeze(-1).squeeze(-1)
            pos_feats_init = self.r2p_gcn(rot_feats_init, self.r2p_A[0])

            if self.training:
                if cfg.DANET.SMPL_KPS_WEIGHTS > 0 and cfg.DANET.REFINEMENT.POS_INTERSUPV:
                    coord_feats0 = pos_feats_init.unsqueeze(2).view(pos_feats_init.size(0), pos_feats_init.size(-1) * 24, 1, 1)
                    smpl_coord0 = self.coord_regressors[0](coord_feats0).view(nbs, 24, -1)
                    return_dict['smpl_coord'].append(smpl_coord0)

            if cfg.DANET.REFINEMENT.REFINE_ON:
                graph_A = self.A_mask * self.edge_act(self.edge_importance)
                norm_graph_A = normalize_undigraph(self.I_n[0] + graph_A)[0]
                l_pos_feat = self.refine_gcn(pos_feats_init, norm_graph_A)
                l_pos_feat = pos_feats_init + l_pos_feat

                pos_feats_refined = l_pos_feat

                if self.training:
                    if cfg.DANET.SMPL_KPS_WEIGHTS > 0 and cfg.DANET.REFINEMENT.POS_INTERSUPV:
                        coord_feats1 = pos_feats_refined.unsqueeze(2).view(pos_feats_refined.size(0),
                                                                           pos_feats_refined.size(-1) * 24, 1, 1)
                        smpl_coord1 = self.coord_regressors[1](coord_feats1).view(nbs, 24, -1)
                        return_dict['smpl_coord'].append(smpl_coord1)
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

            return_dict['smpl_coord'] = []
            return_dict['smpl_pose'] = []

            local_para = self.pose_regressors[0](rot_feats.view(rot_feats.size(0), 24 * rot_feats.size(2), 1, 1)).view(
                nbs, 24, -1)
            smpl_pose = local_para.view(local_para.size(0), -1)

            if cfg.DANET.REFINEMENT.REFINE_ON:
                return_dict['smpl_pose'].append(smpl_pose)

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
        """
        Return the weight mapping of the weight of the weight matrix.

        Args:
            self: (todo): write your description
        """
        d_wmap = {}  # detectron_weight_mapping
        d_orphan = []  # detectron orphan weight list

        new_key = [key for key in self.state_dict().keys()]
        for key in new_key:
            d_wmap[key] = True

        self.mapping_to_detectron = d_wmap
        self.orphans_in_detectron = []

        return self.mapping_to_detectron, self.orphans_in_detectron
