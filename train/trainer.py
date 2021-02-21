# This script is borrowed and extended from https://github.com/nkolot/SPIN/blob/master/train/trainer.py
import torch
import torch.nn as nn
import numpy as np
from torchgeometry import angle_axis_to_rotation_matrix, rotation_matrix_to_angle_axis
import cv2
from torchvision.utils import make_grid

from datasets import MixedDataset
from datasets import BaseDataset
from models import hmr, SMPL
from models.danet import DaNet
from utils.geometry import batch_rodrigues, perspective_projection, estimate_translation

try:
    from utils.renderer import Renderer
except:
    print('fail to import Renderer.')
    pass

import path_config
import constants
from .fits_dict import FitsDict
from .base_trainer import BaseTrainer

from models.core.config import cfg
import utils.vis as vis_utils

import torch.nn.functional as F


class Trainer(BaseTrainer):
    
    def init_fn(self):
        self.options.img_res = cfg.DANET.INIMG_SIZE
        self.options.heatmap_size = cfg.DANET.HEATMAP_SIZE
        self.train_ds = MixedDataset(self.options, ignore_3d=self.options.ignore_3d, is_train=True)

        self.model = DaNet(options=self.options, smpl_mean_params=path_config.SMPL_MEAN_PARAMS).to(self.device)
        self.smpl = self.model.iuv2smpl.smpl

        self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                          lr=cfg.SOLVER.BASE_LR,
                                          weight_decay=0)

        self.models_dict = {'model': self.model}
        self.optimizers_dict = {'optimizer': self.optimizer}
        self.focal_length = constants.FOCAL_LENGTH

        if self.options.pretrained_checkpoint is not None:
            self.load_pretrained(checkpoint_file=self.options.pretrained_checkpoint)

        # Load dictionary of fits of SPIN
        self.fits_dict = FitsDict(self.options, self.train_ds)

        # Create renderer
        try:
            self.renderer = Renderer(focal_length=self.focal_length, img_res=self.options.img_res, faces=self.smpl.faces)
        except:
            Warning('No renderer for visualization.')
            self.renderer = None

        self.decay_steps_ind = 1

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

    def smpl_losses(self, pred_rotmat, pred_betas, gt_pose, gt_betas, has_smpl):
        pred_rotmat_valid = pred_rotmat[has_smpl == 1]
        gt_rotmat_valid = batch_rodrigues(gt_pose.view(-1,3)).view(-1, 24, 3, 3)[has_smpl == 1]
        pred_betas_valid = pred_betas[has_smpl == 1]
        gt_betas_valid = gt_betas[has_smpl == 1]
        if len(pred_rotmat_valid) > 0:
            loss_regr_pose = self.criterion_regr(pred_rotmat_valid, gt_rotmat_valid)
            loss_regr_betas = self.criterion_regr(pred_betas_valid, gt_betas_valid)
        else:
            loss_regr_pose = torch.FloatTensor(1).fill_(0.).to(self.device)
            loss_regr_betas = torch.FloatTensor(1).fill_(0.).to(self.device)
        return loss_regr_pose, loss_regr_betas

    def train_step(self, input_batch):

        # Learning rate decay
        if self.decay_steps_ind < len(cfg.SOLVER.STEPS) and input_batch['step_count'] == cfg.SOLVER.STEPS[self.decay_steps_ind]:
            lr = self.optimizer.param_groups[0]['lr']
            lr_new = lr * cfg.SOLVER.GAMMA
            print('Decay the learning on step {} from {} to {}'.format(input_batch['step_count'], lr, lr_new))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr_new
            lr = self.optimizer.param_groups[0]['lr']
            assert lr == lr_new
            self.decay_steps_ind += 1

        self.model.train()

        # Get data from the batch
        images = input_batch['img'] # input image
        gt_keypoints_2d = input_batch['keypoints'] # 2D keypoints
        gt_pose = input_batch['pose'] # SMPL pose parameters
        gt_betas = input_batch['betas'] # SMPL beta parameters
        gt_joints = input_batch['pose_3d'] # 3D pose
        has_smpl = input_batch['has_smpl'].byte() # flag that indicates whether SMPL parameters are valid
        has_pose_3d = input_batch['has_pose_3d'].byte() # flag that indicates whether 3D pose is valid
        is_flipped = input_batch['is_flipped'] # flag that indicates whether image was flipped during data augmentation
        rot_angle = input_batch['rot_angle'] # rotation angle used for data augmentation
        dataset_name = input_batch['dataset_name'] # name of the dataset the image comes from
        indices = input_batch['sample_index'] # index of example inside its dataset
        batch_size = images.shape[0]

        # Get GT vertices and model joints
        # Note that gt_model_joints is different from gt_joints as it comes from SMPL
        gt_out = self.smpl(betas=gt_betas, body_pose=gt_pose[:,3:], global_orient=gt_pose[:,:3])
        gt_model_joints = gt_out.joints
        gt_vertices = gt_out.vertices

        # Get current pseudo labels (final fits of SPIN) from the dictionary
        opt_pose, opt_betas = self.fits_dict[(dataset_name, indices.cpu(), rot_angle.cpu(), is_flipped.cpu())]
        opt_pose = opt_pose.to(self.device)
        opt_betas = opt_betas.to(self.device)

        # Replace extreme betas with zero betas
        opt_betas[(opt_betas.abs() > 3).any(dim=-1)] = 0.
        # Replace the optimized parameters with the ground truth parameters, if available
        opt_pose[has_smpl, :] = gt_pose[has_smpl, :]
        opt_betas[has_smpl, :] = gt_betas[has_smpl, :]

        opt_output = self.smpl(betas=opt_betas, body_pose=opt_pose[:,3:], global_orient=opt_pose[:,:3])
        opt_vertices = opt_output.vertices
        opt_joints = opt_output.joints

        # De-normalize 2D keypoints from [-1,1] to pixel space
        gt_keypoints_2d_orig = gt_keypoints_2d.clone()
        gt_keypoints_2d_orig[:, :, :-1] = 0.5 * self.options.img_res * (gt_keypoints_2d_orig[:, :, :-1] + 1)

        # Estimate camera translation given the model joints and 2D keypoints
        # by minimizing a weighted least squares loss
        gt_cam_t = estimate_translation(gt_model_joints, gt_keypoints_2d_orig, focal_length=self.focal_length, img_size=self.options.img_res)

        opt_cam_t = estimate_translation(opt_joints, gt_keypoints_2d_orig, focal_length=self.focal_length, img_size=self.options.img_res)

        if self.options.train_data in ['h36m_coco_itw']:
            valid_fit = self.fits_dict.get_vaild_state(dataset_name, indices.cpu()).to(self.device)
            valid_fit = valid_fit | has_smpl
        else:
            valid_fit = has_smpl

        # Feed images in the network to predict camera and SMPL parameters
        input_batch['opt_pose'] = opt_pose
        input_batch['opt_betas'] = opt_betas
        input_batch['valid_fit'] = valid_fit

        input_batch['dp_dict'] = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in
                                    input_batch['dp_dict'].items()}
        has_iuv = torch.tensor([dn not in ['dp_coco'] for dn in dataset_name], dtype=torch.uint8).to(self.device)
        has_iuv = has_iuv & valid_fit
        input_batch['has_iuv'] = has_iuv
        has_dp = input_batch['has_dp']
        target_smpl_kps = torch.zeros((batch_size, 24, 3)).to(opt_output.smpl_joints.device)
        target_smpl_kps[:, :, :2] = perspective_projection(opt_output.smpl_joints.detach().clone(),
                                                    rotation=torch.eye(3, device=self.device).unsqueeze(0).expand(
                                                        batch_size, -1, -1),
                                                    translation=opt_cam_t,
                                                    focal_length=self.focal_length,
                                                    camera_center=torch.zeros(batch_size, 2, device=self.device) + (0.5 * self.options.img_res))
        target_smpl_kps[:, :, :2] = target_smpl_kps[:, :, :2] / (0.5 * self.options.img_res) - 1
        target_smpl_kps[has_iuv == 1, :, 2] = 1
        target_smpl_kps[has_dp == 1] = input_batch['smpl_2dkps'][has_dp == 1]
        input_batch['target_smpl_kps'] = target_smpl_kps        # [B, 24, 3]
        input_batch['target_verts'] = opt_vertices.detach().clone()      # [B, 6890, 3]

        # camera translation for neural renderer
        gt_cam_t_nr = opt_cam_t.detach().clone()
        gt_camera = torch.zeros(gt_cam_t_nr.shape).to(gt_cam_t_nr.device)
        gt_camera[:, 1:] = gt_cam_t_nr[:, :2]
        gt_camera[:, 0] = (2. * self.focal_length / self.options.img_res) / gt_cam_t_nr[:, 2]
        input_batch['target_cam'] = gt_camera

        # Do forward
        danet_return_dict = self.model(input_batch)

        loss_tatal = 0
        losses_dict = {}
        for loss_key in danet_return_dict['losses']:
            loss_tatal += danet_return_dict['losses'][loss_key]
            losses_dict['loss_{}'.format(loss_key)] = danet_return_dict['losses'][loss_key].detach().item()

        # Do backprop
        self.optimizer.zero_grad()
        loss_tatal.backward()
        self.optimizer.step()

        if input_batch['pretrain_mode']:
            pred_vertices = None
            pred_cam_t = None
        else:
            pred_vertices = danet_return_dict['prediction']['vertices'].detach()
            pred_cam_t = danet_return_dict['prediction']['cam_t'].detach()

        # Pack output arguments for tensorboard logging
        output = {'pred_vertices': pred_vertices,
                'opt_vertices': opt_vertices,
                'pred_cam_t': pred_cam_t,
                'opt_cam_t': opt_cam_t,
                'visualization': danet_return_dict['visualization']}

        losses_dict.update({'loss_tatal': loss_tatal.detach().item()})

        return output, losses_dict

    def train_summaries(self, input_batch, output, losses):
        for loss_name, val in losses.items():
            self.summary_writer.add_scalar(loss_name, val, self.step_count)

    def visualize(self, input_batch, output, losses):
        images = input_batch['img']
        images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1,3,1,1)
        images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1,3,1,1)

        pred_vertices = output['pred_vertices']
        opt_vertices = output['opt_vertices']
        pred_cam_t = output['pred_cam_t']
        opt_cam_t = output['opt_cam_t']
        if self.renderer is not None:
            images_opt = self.renderer.visualize_tb(opt_vertices, opt_cam_t, images)
            self.summary_writer.add_image('opt_shape', images_opt, self.step_count)
            if pred_vertices is not None:
                images_pred = self.renderer.visualize_tb(pred_vertices, pred_cam_t, images)
                self.summary_writer.add_image('pred_shape', images_pred, self.step_count)

        for key_name in ['pred_uv', 'gt_uv', 'part_uvi_pred', 'part_uvi_gt', 'skps_hm_pred', 'skps_hm_pred_soft',
                            'skps_hm_gt', 'skps_hm_gt_soft']:
            if key_name in output['visualization']:
                vis_uv_raw = output['visualization'][key_name]
                if key_name in ['pred_uv', 'gt_uv']:
                    iuv = F.interpolate(vis_uv_raw, scale_factor=4., mode='nearest')
                    img_iuv = images.clone()
                    img_iuv[iuv > 0] = iuv[iuv > 0]
                    vis_uv = make_grid(img_iuv, padding=1, pad_value=1)
                else:
                    vis_uv = make_grid(vis_uv_raw, padding=1, pad_value=1)
                self.summary_writer.add_image(key_name, vis_uv, self.step_count)

        if 'target_smpl_kps' in input_batch:
            smpl_kps = input_batch['target_smpl_kps'].detach()
            smpl_kps[:, :, :2] *= images.size(-1) / 2.
            smpl_kps[:, :, :2] += images.size(-1) / 2.
            img_smpl_hm = images.detach().clone()
            img_with_smpljoints = vis_utils.vis_batch_image_with_joints(img_smpl_hm.data,
                                                                        smpl_kps.cpu().numpy(),
                                                                        np.ones((smpl_kps.shape[0],
                                                                                    smpl_kps.shape[1], 1)))
            img_with_smpljoints = np.transpose(img_with_smpljoints, (2, 0, 1))
            self.summary_writer.add_image('stn_centers_gt', img_with_smpljoints, self.step_count)

        if 'stn_kps_pred' in output['visualization']:
            smpl_kps = output['visualization']['stn_kps_pred']
            smpl_kps[:, :, :2] *= images.size(-1) / 2.
            smpl_kps[:, :, :2] += images.size(-1) / 2.
            img_smpl_hm = images.detach().clone()
            if 'skps_hm_gt' in output['visualization']:
                smpl_hm = output['visualization']['skps_hm_gt'].expand(-1, 3, -1, -1)
                smpl_hm = F.interpolate(smpl_hm, scale_factor=output.size(-1) / smpl_hm.size(-1))
                img_smpl_hm[smpl_hm > 0.1] = smpl_hm[smpl_hm > 0.1]
            img_with_smpljoints = vis_utils.vis_batch_image_with_joints(img_smpl_hm.data,
                                                                        smpl_kps.cpu().numpy(),
                                                                        np.ones((smpl_kps.shape[0],
                                                                                    smpl_kps.shape[1], 1)))
            img_with_smpljoints = np.transpose(img_with_smpljoints, (2, 0, 1))
            self.summary_writer.add_image('stn_centers_pred', img_with_smpljoints, self.step_count)
