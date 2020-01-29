from functools import wraps
import os
import torch
from lib.core.config import cfg
import numpy as np
import pickle
from lib.utils.smpl import smpl_structure
import torch.nn as nn
import torch.nn.functional as F
import neural_renderer as nr
from neural_renderer.projection import projection as nr_projection

import lib.utils.densepose_methods as dp_utils
from lib.modeling.GCN import GCN
from lib.utils.graph import Graph, normalize_digraph, normalize_undigraph
from lib.utils.imutils import softmax_integral_tensor
from lib.utils.dropblock import getDropBlockMask
import lib.utils.net as net_utils
from lib.utils.iuvmap import iuv_img2map, iuv_map2img, iuvmap_clean

from lib.utils.collections import AttrDict

from .smpl_model import SMPL
from .hr_module import PoseHighResolutionNet
from .res_module import PoseResNet, SmplResNet, LimbResLayers


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


class DaNet(nn.Module):
    '''
    Ref.
    H. Zhang et al. DaNet: Decompose-and-aggregate Network for 3D Human Shape and Pose Estimation
    H. Zhang et al. Learning 3D Human Shape and Pose from Dense Body Parts
    '''
    def __init__(self):
        super(DaNet, self).__init__()

        # For cache
        self.mapping_to_detectron = None
        self.orphans_in_detectron = None

        if cfg.DANET.INPUT_MODE not in ['iuv_gt'] or cfg.DANET.DECOMPOSED:
            self.img2iuv = IUV_Estimator()
            if cfg.DANET.INPUT_MODE not in ['iuv_gt']:
                try:
                    final_feat_dim = self.img2iuv.iuv_est.final_feat_dim
                except:
                    final_feat_dim = None
            else:
                final_feat_dim = None
        else:
            final_feat_dim = None
        self.iuv2smpl = SMPL_Head(feat_in_dim=final_feat_dim)

    @check_inference
    def infer_net(self, image):
        """For inference"""
        return_dict = {}
        return_dict['visualization'] = {}

        if cfg.DANET.INPUT_MODE in ['iuv_gt']:
            if cfg.DANET.DECOMPOSED:
                uv_return_dict = self.img2iuv(image[0], iuv_image_gt=image[1], smpl_kps_gt=image[2])
                u_pred, v_pred, index_pred, ann_pred = iuv_img2map(image[1])
            else:
                uv_return_dict = {}
                u_pred, v_pred, index_pred, ann_pred = iuv_img2map(image)
        elif cfg.DANET.INPUT_MODE in ['iuv_gt_feat']:
            uv_return_dict = self.img2iuv(image[0])
            u_pred, v_pred, index_pred, ann_pred = iuv_img2map(image[1])
        else:
            uv_return_dict = self.img2iuv(image)
            u_pred, v_pred, index_pred, ann_pred = iuvmap_clean(*uv_return_dict['uvia_pred'])

        return_dict['visualization']['iuv_pred'] = [u_pred, v_pred, index_pred, ann_pred]
        return_dict['visualization']['part_iuv_pred'] = uv_return_dict['part_iuv_pred']

        iuv_map = torch.cat([u_pred, v_pred, index_pred], dim=1)

        if cfg.DANET.INPUT_MODE in ['iuv_gt', 'iuv_gt_feat'] and 'part_iuv_gt' in uv_return_dict:
            part_iuv_map = uv_return_dict['part_iuv_gt']
            part_index_map = part_iuv_map[:, :, 2]
        elif 'part_iuv_pred' in uv_return_dict:
            part_iuv_pred = uv_return_dict['part_iuv_pred']
            part_iuv_map = []
            for p_ind in range(part_iuv_pred.size(1)):
                p_u_pred, p_v_pred, p_index_pred = [part_iuv_pred[:, p_ind, iuv] for iuv in range(3)]
                p_u_map, p_v_map, p_i_map, _ = iuvmap_clean(p_u_pred, p_v_pred, p_index_pred)
                p_iuv_map = torch.stack([p_u_map, p_v_map, p_i_map], dim=1)
                part_iuv_map.append(p_iuv_map)
            part_iuv_map = torch.stack(part_iuv_map, dim=1)
            part_index_map = part_iuv_map[:, :, 2].detach()
        else:
            part_iuv_map = None
            part_index_map = None

        if uv_return_dict.has_key('part_featmaps'):
            part_feat_map = uv_return_dict['part_featmaps']
        else:
            part_feat_map = None

        if cfg.DANET.INPUT_MODE == 'feat':
            smpl_return_dict = self.iuv2smpl.smpl_infer_net({'iuv_map': {'feat': uv_return_dict['global_featmaps']},
                                             'part_iuv_map': {'pfeat': part_feat_map}
                                                             })
        elif cfg.DANET.INPUT_MODE in ['iuv_feat', 'iuv_gt_feat']:
            smpl_return_dict = self.iuv2smpl.smpl_infer_net({'iuv_map': {'iuv': iuv_map, 'feat': uv_return_dict['global_featmaps']},
                                             'part_iuv_map': {'piuv': part_iuv_map, 'pfeat': part_feat_map}
                                                             })
        elif cfg.DANET.INPUT_MODE in ['iuv', 'iuv_gt']:
            smpl_return_dict = self.iuv2smpl.smpl_infer_net({'iuv_map': iuv_map,
                                                            'part_iuv_map': part_iuv_map
                                                             })
        elif cfg.DANET.INPUT_MODE == 'seg':
            smpl_return_dict = self.iuv2smpl.smpl_infer_net({'iuv_map': {'index': index_pred.detach()},
                                             'part_iuv_map': {'pindex': part_index_map}
                                                             })

        return_dict['para'] = smpl_return_dict['para']

        for k, v in smpl_return_dict['visualization'].items():
            return_dict['visualization'][k] = v

        return return_dict

    def forward(self, in_dict):
        if cfg.PYTORCH_VERSION_LESS_THAN_040:
            return self._forward(in_dict)
        else:
            with torch.set_grad_enabled(self.training):
                return self._forward(in_dict)

    def _forward(self, in_dict):

        if type(in_dict) is not dict:
            in_dict = {'data': in_dict, 'pretrain_mode': False, 'vis_on': False, 'data_name': ''}

        image = in_dict['data']
        target = in_dict['target'] if 'target' in in_dict else None
        target_kps = in_dict['target_kps'] if 'target_kps' in in_dict else None
        target_kps3d = in_dict['target_kps3d'] if 'target_kps3d' in in_dict else None
        kp3d_type = in_dict['kp3d_type'] if 'kp3d_type' in in_dict else None
        target_smpl_kps = in_dict['target_smpl_kps'] if 'target_smpl_kps' in in_dict else None
        target_verts = in_dict['target_verts'] if 'target_verts' in in_dict else None
        uv_image_gt = in_dict['iuv_image'] if 'iuv_image' in in_dict else None

        target_iuv_dp = in_dict['target_iuv_dp'] if 'target_iuv_dp' in in_dict else None

        if 'target_kps_coco' in in_dict:
            target_kps = in_dict['target_kps_coco']

        return_dict = {}
        return_dict['losses'] = {}
        return_dict['metrics'] = {}
        return_dict['visualization'] = {}

        if cfg.DANET.INPUT_MODE in ['iuv_gt']:
            if cfg.DANET.DECOMPOSED:
                uv_return_dict = self.img2iuv(image, uv_image_gt, target_smpl_kps, pretrained=in_dict['pretrain_mode'], uvia_dp_gt=target_iuv_dp)
                uv_return_dict['uvia_pred'] = iuv_img2map(uv_image_gt)
            else:
                uv_return_dict = {}
                uv_return_dict['uvia_pred'] = iuv_img2map(uv_image_gt)
        elif cfg.DANET.INPUT_MODE in ['iuv_gt_feat']:
            uv_return_dict = self.img2iuv(image, uv_image_gt, target_smpl_kps, pretrained=in_dict['pretrain_mode'], uvia_dp_gt=target_iuv_dp)
            uv_return_dict['uvia_pred'] = iuv_img2map(uv_image_gt)
        elif cfg.DANET.INPUT_MODE in ['feat']:
            uv_return_dict = self.img2iuv(image, None, target_smpl_kps, pretrained=in_dict['pretrain_mode'], uvia_dp_gt=target_iuv_dp)
        else:
            uv_return_dict = self.img2iuv(image, uv_image_gt, target_smpl_kps, pretrained=in_dict['pretrain_mode'], uvia_dp_gt=target_iuv_dp)

        u_pred, v_pred, index_pred, ann_pred = uv_return_dict['uvia_pred']
        if self.training and cfg.DANET.PART_IUV_ZERO > 0 and in_dict['data_name'] not in ['coco_dp']:
            if cfg.DANET.DROP_STRATEGY in ['partdrop']:
                zero_idxs = []
                for bs in range(u_pred.shape[0]):
                    zero_idxs.append([int(i) + 1 for i in torch.nonzero(torch.rand(24) < cfg.DANET.PART_IUV_ZERO)])
            elif cfg.DANET.DROP_STRATEGY in ['dropblock']:
                if cfg.DANET.CUTOUT_BS == 1:
                    drop_prob = cfg.DANET.PART_IUV_ZERO
                else:
                    if cfg.DANET.CUTOUT_BS == 7:
                        drop_prob_mapping = [0.0, 0.092, 0.212, 0.348, 0.521, 0.686, 0.900]
                        drop_prob = drop_prob_mapping[int(cfg.DANET.PART_IUV_ZERO * 10)]
                zero_mask = getDropBlockMask((u_pred.shape[0] * u_pred.shape[1], u_pred.shape[-2], u_pred.shape[-1]),
                                             drop_prob=drop_prob,
                                             block_size=int(cfg.DANET.CUTOUT_BS), device=u_pred.device)
                zero_mask = zero_mask.reshape(u_pred.shape[0], -1, u_pred.shape[-2], u_pred.shape[-1])

        if self.training and cfg.DANET.PART_IUV_ZERO > 0 and in_dict['data_name'] not in ['coco_dp']:
            if cfg.DANET.DROP_STRATEGY in ['partdrop']:
                for bs in range(len(zero_idxs)):
                    u_pred[bs, zero_idxs[bs]] *= 0
                    v_pred[bs, zero_idxs[bs]] *= 0
                    index_pred[bs, zero_idxs[bs]] *= 0
            elif cfg.DANET.DROP_STRATEGY in ['dropblock']:
                u_pred *= zero_mask
                v_pred *= zero_mask
                index_pred *= zero_mask

        u_pred_cl, v_pred_cl, index_pred_cl, ann_pred_cl = iuvmap_clean(u_pred, v_pred, index_pred, ann_pred)

        iuv_pred_clean = [u_pred_cl.detach(), v_pred_cl.detach(), index_pred_cl.detach(), ann_pred_cl.detach()]
        return_dict['visualization']['iuv_pred'] = iuv_pred_clean

        if not in_dict['pretrain_mode']:

            iuv_map = torch.cat([u_pred_cl, v_pred_cl, index_pred_cl], dim=1)

            if cfg.DANET.INPUT_MODE in ['iuv_gt', 'iuv_gt_feat'] and 'part_iuv_gt' in uv_return_dict:
                part_iuv_map = uv_return_dict['part_iuv_gt']
                if self.training and cfg.DANET.PART_IUV_ZERO > 0:
                    if cfg.DANET.DROP_STRATEGY in ['partdrop']:
                        for bs in range(len(zero_idxs)):
                            zero_channel = []
                            for zero_i in zero_idxs[bs]:
                                zero_channel.extend(
                                    [(i, m_i + 1) for i, mapping in enumerate(self.img2iuv.dp2smpl_mapping) for m_i, map_idx in
                                     enumerate(mapping) if map_idx == zero_i])
                            zero_dp_i = [iterm[0] for iterm in zero_channel]
                            zero_p_i = [iterm[1] for iterm in zero_channel]
                            part_iuv_map[bs, zero_dp_i, :, zero_p_i] *= 0
                    elif cfg.DANET.DROP_STRATEGY in ['dropblock']:
                        part_iuv_map *= zero_mask[:, 1:, None, None, :, :]

                part_index_map = part_iuv_map[:, :, 2]
            elif 'part_iuv_pred' in uv_return_dict:
                part_iuv_pred = uv_return_dict['part_iuv_pred']
                if self.training and cfg.DANET.PART_IUV_ZERO > 0 and in_dict['data_name'] not in ['coco_dp']:
                    if cfg.DANET.DROP_STRATEGY in ['partdrop']:
                        for bs in range(len(zero_idxs)):
                            zero_channel = []
                            for zero_i in zero_idxs[bs]:
                                zero_channel.extend(
                                    [(i, m_i + 1) for i, mapping in enumerate(self.img2iuv.dp2smpl_mapping) for m_i, map_idx in
                                     enumerate(mapping) if map_idx == zero_i])
                            zero_dp_i = [iterm[0] for iterm in zero_channel]
                            zero_p_i = [iterm[1] for iterm in zero_channel]
                            part_iuv_pred[bs, zero_dp_i, :, zero_p_i] *= 0
                    elif cfg.DANET.DROP_STRATEGY in ['dropblock']:
                        part_iuv_pred *= zero_mask[:, 1:, None, None, :, :]

                part_iuv_map = []
                for p_ind in range(part_iuv_pred.size(1)):
                    p_u_pred, p_v_pred, p_index_pred = [part_iuv_pred[:, p_ind, iuv] for iuv in range(3)]
                    p_u_map, p_v_map, p_i_map, _ = iuvmap_clean(p_u_pred, p_v_pred, p_index_pred)
                    p_iuv_map = torch.stack([p_u_map, p_v_map, p_i_map], dim=1)
                    part_iuv_map.append(p_iuv_map)
                part_iuv_map = torch.stack(part_iuv_map, dim=1)
                part_index_map = part_iuv_map[:, :, 2]

            else:
                part_iuv_map = None
                part_index_map = None

            return_dict['visualization']['part_iuv_pred'] = part_iuv_map

            if uv_return_dict.has_key('part_featmaps'):
                part_feat_map = uv_return_dict['part_featmaps']
            else:
                part_feat_map = None

            if cfg.DANET.INPUT_MODE == 'feat':
                smpl_return_dict = self.iuv2smpl({'iuv_map': {'feat': uv_return_dict['global_featmaps']},
                                                 'part_iuv_map': {'pfeat': part_feat_map},
                                                 'target': target,
                                                 'target_kps': target_kps,
                                                 'target_verts': target_verts,
                                                 'target_kps3d': target_kps3d,
                                                 'kp3d_type': kp3d_type
                                                  })
            elif cfg.DANET.INPUT_MODE in ['iuv_feat', 'iuv_gt_feat']:
                smpl_return_dict = self.iuv2smpl({'iuv_map': {'iuv': iuv_map, 'feat': uv_return_dict['global_featmaps']},
                                                 'part_iuv_map': {'piuv': part_iuv_map, 'pfeat': part_feat_map},
                                                 'target': target,
                                                 'target_kps': target_kps,
                                                 'target_verts': target_verts,
                                                 'target_kps3d': target_kps3d,
                                                 'kp3d_type': kp3d_type
                                                  })
            elif cfg.DANET.INPUT_MODE in ['iuv', 'iuv_gt']:
                smpl_return_dict = self.iuv2smpl({'iuv_map': iuv_map,
                                                 'part_iuv_map': part_iuv_map,
                                                 'target': target,
                                                 'target_kps': target_kps,
                                                 'target_verts': target_verts,
                                                 'target_kps3d': target_kps3d,
                                                 'kp3d_type': kp3d_type
                                                  })
            elif cfg.DANET.INPUT_MODE == 'seg':
                # REMOVE _.detach
                smpl_return_dict = self.iuv2smpl({'iuv_map': {'index': index_pred_cl},
                                                 'part_iuv_map': {'pindex': part_index_map},
                                                 'target': target,
                                                 'target_kps': target_kps,
                                                 'target_verts': target_verts,
                                                 'target_kps3d': target_kps3d,
                                                 'kp3d_type': kp3d_type
                                                  })

        for key_name in ['losses', 'metrics', 'visualization']:
            if uv_return_dict.has_key(key_name):
                return_dict[key_name].update(uv_return_dict[key_name])
            if not in_dict['pretrain_mode']:
                return_dict[key_name].update(smpl_return_dict[key_name])

        # pytorch0.4 bug on gathering scalar(0-dim) tensors
        for k, v in return_dict['losses'].items():
            if len(v.shape) == 0:
                return_dict['losses'][k] = v.unsqueeze(0)
        for k, v in return_dict['metrics'].items():
            if len(v.shape) == 0:
                return_dict['metrics'][k] = v.unsqueeze(0)

        return return_dict

    # @property
    def detectron_weight_mapping(self):
        if self.mapping_to_detectron is None:
            d_wmap = {}  # detectron_weight_mapping
            d_orphan = []  # detectron orphan weight list
            for name, m_child in self.named_children():
                if list(m_child.parameters()):  # if module has any parameter
                    if callable(m_child.detectron_weight_mapping):
                        child_map, child_orphan = m_child.detectron_weight_mapping()
                    else:
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


class IUV_Estimator(nn.Module):
    def __init__(self):
        super(IUV_Estimator, self).__init__()

        if cfg.DANET.USE_LEARNED_RATIO:
            with open(os.path.join('./data/pretrained_model', 'learned_ratio.pkl'), 'rb') as f:
                pretrain_ratio = pickle.load(f)

            if cfg.DANET.INPUT_MODE in ['iuv_gt']:
                self.learned_ratio = nn.Parameter(torch.FloatTensor(pretrain_ratio['ratio']))
                self.learned_offset = nn.Parameter(torch.FloatTensor(pretrain_ratio['offset']))
            else:
                self.register_buffer('learned_ratio', torch.FloatTensor(pretrain_ratio['ratio']))
                self.register_buffer('learned_offset', torch.FloatTensor(pretrain_ratio['offset']))
        else:
            self.learned_ratio = nn.Parameter(cfg.DANET.PART_UVI_SCALE * torch.ones(24))
            self.learned_offset = nn.Parameter(cfg.DANET.PART_UVI_LR_OFFSET * torch.ones(24))

        self.smpl_parents = smpl_structure('smpl_parents')
        self.smpl_children = smpl_structure('smpl_children')
        self.smpl2dp_part = smpl_structure('smpl2dp_part')
        if cfg.DANET.PART_IUV_SIMP:
            self.dp2smpl_mapping = smpl_structure('dp2smpl_mapping')
        else:
            self.dp2smpl_mapping = [range(1, 25)] * 24

        if cfg.DANET.INPUT_MODE not in ['iuv_gt']:
            part_out_dim = 1 + len(self.dp2smpl_mapping[0])
            if cfg.DANET.IUV_REGRESSOR in ['resnet']:
                self.iuv_est = PoseResNet(part_out_dim=part_out_dim)
                if len(cfg.MSRES_MODEL.PRETRAINED) > 0:
                    self.iuv_est.init_weights(cfg.MSRES_MODEL.PRETRAINED)
            elif cfg.DANET.IUV_REGRESSOR in ['hrnet']:
                self.iuv_est = PoseHighResolutionNet(part_out_dim=part_out_dim)
                if cfg.HR_MODEL.PRETR_SET in ['imagenet']:
                    self.iuv_est.init_weights(cfg.HR_MODEL.PRETRAINED_IM)
                elif cfg.HR_MODEL.PRETR_SET in ['coco']:
                    self.iuv_est.init_weights(cfg.HR_MODEL.PRETRAINED_COCO)

        self.bodyfeat_channels = 1024
        self.part_channels = 256

    def forward(self, data, iuv_image_gt=None, smpl_kps_gt=None, kps3d_gt=None, pretrained=False, uvia_dp_gt=None):
        return_dict = {}
        return_dict['losses'] = {}
        return_dict['metrics'] = {}
        return_dict['visualization'] = {}

        if cfg.DANET.INPUT_MODE in ['iuv_gt']:
            uvia_list = iuv_img2map(iuv_image_gt)
            stn_centers_target = smpl_kps_gt[:, :, :2].contiguous()
            if self.training and cfg.DANET.STN_CENTER_JITTER > 0:
                stn_centers_target = stn_centers_target + cfg.DANET.STN_CENTER_JITTER * (
                            torch.rand(stn_centers_target.size()).cuda(stn_centers_target.device) - 0.5)

            thetas, scales = self.affine_para(stn_centers_target)

            part_map_size = iuv_image_gt.size(-1)
            pred_gt_ratio = float(part_map_size) / uvia_list[0].size(-1)
            iuv_resized = [F.interpolate(uvia_list[i], scale_factor=pred_gt_ratio, mode='nearest') for i in
                           range(3)]
            iuv_simplified = self.part_iuv_simp(iuv_resized)
            part_iuv_gt = []
            for i in range(len(iuv_simplified)):
                part_iuv_i = iuv_simplified[i]
                grid = F.affine_grid(thetas[i], part_iuv_i.size())
                part_iuv_i = F.grid_sample(part_iuv_i, grid)
                part_iuv_i = part_iuv_i.view(-1, 3, len(self.dp2smpl_mapping[i]) + 1, part_map_size, part_map_size)
                part_iuv_gt.append(part_iuv_i)

            # (bs, 24, 3, 7, 56, 56)
            return_dict['part_iuv_gt'] = torch.stack(part_iuv_gt, dim=1)

            return return_dict

        uv_est_dic = self.iuv_est(data)
        u_pred, v_pred, index_pred, ann_pred = uv_est_dic['predict_u'], uv_est_dic['predict_v'], uv_est_dic['predict_uv_index'], uv_est_dic['predict_ann_index']

        if cfg.DANET.INPUT_MODE in ['iuv_feat', 'feat', 'iuv_gt_feat']:
            return_dict['global_featmaps'] = uv_est_dic['xd']

        if iuv_image_gt is not None:
            uvia_list = iuv_img2map(iuv_image_gt)
            if not cfg.DANET.FIXED_UV_EST:
                loss_U, loss_V, loss_IndexUV, loss_segAnn = self.body_uv_losses(u_pred, v_pred, index_pred, ann_pred,
                                                                                uvia_list)
                return_dict['losses']['loss_U'] = loss_U
                return_dict['losses']['loss_V'] = loss_V
                return_dict['losses']['loss_IndexUV'] = loss_IndexUV
                return_dict['losses']['loss_segAnn'] = loss_segAnn

        elif uvia_dp_gt is not None:
            loss_U, loss_V, loss_IndexUV, loss_segAnn = self.dp_uvia_losses(u_pred, v_pred, index_pred, ann_pred, **uvia_dp_gt)
            return_dict['losses']['loss_Udp'] = loss_U
            return_dict['losses']['loss_Vdp'] = loss_V
            return_dict['losses']['loss_IndexUVdp'] = loss_IndexUV
            return_dict['losses']['loss_segAnndp'] = loss_segAnn

        if cfg.DANET.FIXED_UV_EST:
            return_dict['uvia_pred'] = [u_pred.detach(), v_pred.detach(), index_pred.detach(), ann_pred.detach()]
        else:
            return_dict['uvia_pred'] = [u_pred, v_pred, index_pred, ann_pred]

        if cfg.DANET.DECOMPOSED:

            u_pred_cl, v_pred_cl, index_pred_cl, ann_pred_cl = iuvmap_clean(u_pred, v_pred, index_pred, ann_pred)

            partial_decon_feat = uv_est_dic['xd']

            skps_hm_pred = uv_est_dic['predict_hm']

            smpl_kps_hm_size = skps_hm_pred.size(-1)

            # return_dict['skps_hm_pred'] = torch.sum(skps_hm_pred.detach(), dim=1).unsqueeze(1)
            return_dict['skps_hm_pred'] = skps_hm_pred.detach()

            stn_centers = softmax_integral_tensor(10 * skps_hm_pred, skps_hm_pred.size(1), skps_hm_pred.size(-2),
                                                  skps_hm_pred.size(-1))
            stn_centers /= 0.5 * smpl_kps_hm_size
            stn_centers -= 1

            if smpl_kps_gt is not None:
                # train mode
                if cfg.DANET.STN_KPS_WEIGHTS > 0:
                    if smpl_kps_gt.shape[-1] == 4:
                        loss_roi = 0
                        for w in torch.unique(smpl_kps_gt[:, :, 3]):
                            if w == 0:
                                continue
                            kps_w_idx = smpl_kps_gt[:, :, 3] == w
                            # stn_centers_target = smpl_kps_gt[:, :, :2][kps_w1_idx]
                            loss_roi += F.smooth_l1_loss(stn_centers[kps_w_idx], smpl_kps_gt[:, :, :2][kps_w_idx], size_average=False) * w
                        loss_roi /= smpl_kps_gt.size(0)
                    else:
                        stn_centers_target = smpl_kps_gt[:, :, :2]
                        loss_roi = F.smooth_l1_loss(stn_centers, stn_centers_target, size_average=False) / smpl_kps_gt.size(0)

                    loss_roi *= cfg.DANET.STN_KPS_WEIGHTS
                    return_dict['losses']['loss_roi'] = loss_roi

                if self.training and cfg.DANET.STN_CENTER_JITTER > 0:
                    stn_centers = stn_centers + cfg.DANET.STN_CENTER_JITTER * (torch.rand(stn_centers.size()).cuda(stn_centers.device) - 0.5)

            if cfg.DANET.STN_PART_VIS_SCORE > 0:
                part_hidden_score = []
                for i in range(24):
                    score_map = torch.max(index_pred_cl[:, self.smpl2dp_part[i]], dim=1)[0].detach()
                    score_i = F.grid_sample(score_map.unsqueeze(1), stn_centers[:, i].unsqueeze(1).unsqueeze(1)).detach()
                    part_hidden_score.append(score_i.squeeze(-1).squeeze(-1).squeeze(-1))

                part_hidden_score = torch.stack(part_hidden_score)
                part_hidden_score = part_hidden_score < cfg.DANET.STN_PART_VIS_SCORE

            else:
                part_hidden_score = None

            maps_transformed = []

            thetas, scales = self.affine_para(stn_centers, part_hidden_score)

            for i in range(24):
                theta_i = thetas[i]
                scale_i = scales[i]

                grid = F.affine_grid(theta_i.detach(), partial_decon_feat.size())
                maps_transformed_i = F.grid_sample(partial_decon_feat, grid)

                maps_transformed.append(maps_transformed_i)

            return_dict['stn_kps_pred'] = stn_centers.detach()

            part_maps = torch.cat(maps_transformed, dim=1)

            part_iuv_pred = self.iuv_est.final_pred.predict_partial_iuv(part_maps)
            part_map_size = part_iuv_pred.size(-1)
            # (bs, 24, 3, 7, 56, 56)
            part_iuv_pred = part_iuv_pred.view(part_iuv_pred.size(0), len(self.dp2smpl_mapping), 3, -1,
                                               part_map_size,
                                               part_map_size)

            if cfg.DANET.INPUT_MODE in ['iuv_feat', 'feat', 'iuv_gt_feat']:
                return_dict['part_featmaps'] = part_maps.view(part_maps.size(0), 24, -1, part_maps.size(-2), part_maps.size(-1))

            ## partial uv gt
            if iuv_image_gt is not None:
                pred_gt_ratio = float(part_map_size) / uvia_list[0].size(-1)
                iuv_resized = [F.interpolate(uvia_list[i], scale_factor=pred_gt_ratio, mode='nearest') for i in
                               range(3)]
                iuv_simplified = self.part_iuv_simp(iuv_resized)
                part_iuv_gt = []
                for i in range(len(iuv_simplified)):
                    part_iuv_i = iuv_simplified[i]
                    grid = F.affine_grid(thetas[i].detach(), part_iuv_i.size())
                    part_iuv_i = F.grid_sample(part_iuv_i, grid)
                    part_iuv_i = part_iuv_i.view(-1, 3, len(self.dp2smpl_mapping[i]) + 1, part_map_size, part_map_size)
                    part_iuv_gt.append(part_iuv_i)

                return_dict['part_iuv_gt'] = torch.stack(part_iuv_gt, dim=1)

                loss_p_U, loss_p_V, loss_p_IndexUV = None, None, None
                for i in range(len(part_iuv_gt)):
                    part_uvia_list = [part_iuv_gt[i][:, iuv] for iuv in range(3)]
                    part_uvia_list.append(None)

                    p_iuv_pred_i = [part_iuv_pred[:, i, iuv] for iuv in range(3)]

                    loss_p_U_i, loss_p_V_i, loss_p_IndexUV_i, _ = self.body_uv_losses(p_iuv_pred_i[0], p_iuv_pred_i[1],
                                                                                      p_iuv_pred_i[2], None,
                                                                                      part_uvia_list)

                    if i == 0:
                        loss_p_U, loss_p_V, loss_p_IndexUV = loss_p_U_i, loss_p_V_i, loss_p_IndexUV_i
                    else:
                        loss_p_U += loss_p_U_i
                        loss_p_V += loss_p_V_i
                        loss_p_IndexUV += loss_p_IndexUV_i

                loss_p_U /= 24.
                loss_p_V /= 24.
                loss_p_IndexUV /= 24.

                return_dict['losses']['loss_pU'] = loss_p_U
                return_dict['losses']['loss_pV'] = loss_p_V
                return_dict['losses']['loss_pIndexUV'] = loss_p_IndexUV

            return_dict['part_iuv_pred'] = part_iuv_pred

        return return_dict


    def affine_para(self, stn_centers, part_hidden=None):
        thetas = []
        scales = []

        kps_box_diag = torch.max(stn_centers, dim=1)[0] - torch.min(stn_centers, dim=1)[0]
        scale_box = torch.max(kps_box_diag, dim=1)[0] / 2.

        for i in range(24):
            p_ind = self.smpl_parents[0][i]
            c_ind = self.smpl_children[1][i]
            center_i = stn_centers[:, i].detach()
            if i == 0:
                scale_i = scale_box
            else:
                scale_c = torch.norm(stn_centers[:, c_ind] - stn_centers[:, i], dim=1) / 2.
                scale_p = torch.norm(stn_centers[:, p_ind] - stn_centers[:, i], dim=1) / 2.
                scale_i = 2 * torch.max(torch.stack([scale_c, scale_p], dim=0), dim=0)[0]

            scale_i = scale_i.detach()
            scale_i = scale_i * F.relu(self.learned_ratio[i])
            scale_i = scale_i + F.relu(self.learned_offset[i])

            if self.training and cfg.DANET.STN_SCALE_JITTER > 0:
                scale_i = scale_i * (1 + cfg.DANET.STN_SCALE_JITTER * (torch.rand(scale_i.size()).cuda(scale_i.device) - 0.5))

            if i != 0 and part_hidden is not None:
                scale_i[part_hidden[i]] = 0.8 * scale_box[part_hidden[i]]

            if self.training and cfg.DANET.STN_SCALE_JITTER > 0:
                scale_i = scale_i * (1 + cfg.DANET.STN_SCALE_JITTER * (torch.rand(scale_i.size()).cuda(scale_i.device) - 0.5))

            theta_i = torch.zeros(stn_centers.size(0), 2, 3).cuda()
            theta_i[:, 0, 0] = scale_i
            theta_i[:, 1, 1] = scale_i
            theta_i[:, :, -1] = center_i

            thetas.append(theta_i)
            scales.append(scale_i)

        return thetas, scales


    def body_uv_losses(self, u_pred, v_pred, index_pred, ann_pred, uvia_list):
        batch_size = u_pred.size(0)

        Umap, Vmap, Imap, Annmap = uvia_list

        Itarget = torch.argmax(Imap, dim=1)
        Itarget = Itarget.view(-1).to(torch.int64)

        index_pred = index_pred.permute([0, 2, 3, 1]).contiguous()
        index_pred = index_pred.view(-1, Imap.size(1))

        loss_U = F.smooth_l1_loss(u_pred[Imap > 0], Umap[Imap > 0], size_average=False) / batch_size
        loss_V = F.smooth_l1_loss(v_pred[Imap > 0], Vmap[Imap > 0], size_average=False) / batch_size
        loss_IndexUV = F.cross_entropy(index_pred, Itarget)

        loss_U *= cfg.DANET.POINT_REGRESSION_WEIGHTS
        loss_V *= cfg.DANET.POINT_REGRESSION_WEIGHTS

        if ann_pred is None:
            loss_segAnn = None
        else:
            Anntarget = torch.argmax(Annmap, dim=1)
            Anntarget = Anntarget.view(-1).to(torch.int64)
            ann_pred = ann_pred.permute([0, 2, 3, 1]).contiguous()
            ann_pred = ann_pred.view(-1, Annmap.size(1))
            loss_segAnn = F.cross_entropy(ann_pred, Anntarget)

        return loss_U, loss_V, loss_IndexUV, loss_segAnn


    def dp_uvia_losses(self, U_estimated, V_estimated, Index_UV, Ann_Index,
                       body_uv_X_points,
                       body_uv_Y_points,
                       body_uv_I_points,
                       body_uv_Ind_points,
                       body_uv_U_points,
                       body_uv_V_points,
                       body_uv_point_weights,
                       body_uv_ann_labels,
                       body_uv_ann_weights):
        """Mask R-CNN body uv specific losses."""
        device_id = U_estimated.get_device()

        ## Reshape for GT blobs.
        ## Concat Ind,x,y to get Coordinates blob.
        Coordinates = torch.cat(
            (body_uv_Ind_points.unsqueeze(2), body_uv_X_points.unsqueeze(2), body_uv_Y_points.unsqueeze(2)), dim=2)
        ##
        ### Now reshape UV blobs, such that they are 1x1x(196*NumSamples)xNUM_PATCHES
        ## U blob to
        ##
        U_points_reshaped = body_uv_U_points.view(-1, cfg.DANET.NUM_PATCHES + 1, 196)
        U_points_reshaped_transpose = torch.transpose(U_points_reshaped, 1, 2).contiguous()
        U_points = U_points_reshaped_transpose.view(1, 1, -1, cfg.DANET.NUM_PATCHES + 1)
        ## V blob
        ##
        V_points_reshaped = body_uv_V_points.view(-1, cfg.DANET.NUM_PATCHES + 1, 196)
        V_points_reshaped_transpose = torch.transpose(V_points_reshaped, 1, 2).contiguous()
        V_points = V_points_reshaped_transpose.view(1, 1, -1, cfg.DANET.NUM_PATCHES + 1)
        ###
        ## UV weights blob
        ##
        Uv_point_weights_reshaped = body_uv_point_weights.view(-1, cfg.DANET.NUM_PATCHES + 1, 196)
        Uv_point_weights_reshaped_transpose = torch.transpose(Uv_point_weights_reshaped, 1, 2).contiguous()
        Uv_point_weights = Uv_point_weights_reshaped_transpose.view(1, 1, -1, cfg.DANET.NUM_PATCHES + 1)

        #####################
        ###  Pool IUV for points via bilinear interpolation.
        Coordinates[:, :, 1:3] -= cfg.DANET.HEATMAP_SIZE / 2.
        Coordinates[:, :, 1:3] *= 2. / cfg.DANET.HEATMAP_SIZE
        grid = Coordinates[:, :, 1:3].unsqueeze(1)
        interp_U = F.grid_sample(U_estimated, grid)
        interp_U = torch.transpose(interp_U.squeeze(2), 1, 2).contiguous()
        interp_V = F.grid_sample(V_estimated, grid)
        interp_V = torch.transpose(interp_V.squeeze(2), 1, 2).contiguous()
        interp_Index_UV = F.grid_sample(Index_UV, grid)
        interp_Index_UV = torch.transpose(interp_Index_UV.squeeze(2), 1, 2).contiguous()
        interp_Index_UV = interp_Index_UV.view(-1, cfg.DANET.NUM_PATCHES + 1)

        ## Reshape interpolated UV coordinates to apply the loss.

        interp_U_reshaped = interp_U.view(1, 1, -1, cfg.DANET.NUM_PATCHES + 1)
        interp_V_reshaped = interp_V.view(1, 1, -1, cfg.DANET.NUM_PATCHES + 1)
        ###

        ### Do the actual labels here !!!!
        ## The mask segmentation loss (dense)
        num_cls_Index = Ann_Index.size(1)
        Ann_Index_reshaped = Ann_Index.view(-1, num_cls_Index, cfg.DANET.HEATMAP_SIZE ** 2)
        Ann_Index_reshaped = torch.transpose(Ann_Index_reshaped, 1, 2).contiguous().view(-1, num_cls_Index)
        body_uv_ann_labels_reshaped_int = body_uv_ann_labels.to(torch.int64)
        loss_seg_AnnIndex = F.cross_entropy(Ann_Index_reshaped, body_uv_ann_labels_reshaped_int.view(-1))
        loss_seg_AnnIndex *= cfg.DANET.INDEX_WEIGHTS

        ## Point Patch Index Loss.
        I_points_reshaped = body_uv_I_points.view(-1)
        I_points_reshaped_int = I_points_reshaped.to(torch.int64)
        loss_IndexUVPoints = F.cross_entropy(interp_Index_UV, I_points_reshaped_int)
        loss_IndexUVPoints *= cfg.DANET.PART_WEIGHTS
        ## U and V point losses.
        loss_Upoints = net_utils.smooth_l1_loss(interp_U_reshaped, U_points, Uv_point_weights, Uv_point_weights)
        loss_Upoints *= cfg.DANET.POINT_REGRESSION_WEIGHTS

        loss_Vpoints = net_utils.smooth_l1_loss(interp_V_reshaped, V_points, Uv_point_weights, Uv_point_weights)
        loss_Vpoints *= cfg.DANET.POINT_REGRESSION_WEIGHTS

        return loss_Upoints, loss_Vpoints, loss_IndexUVPoints, loss_seg_AnnIndex


    def part_iuv_simp(self, mapList):
        """partial iuv simplification."""
        n_channels = len(mapList)
        maps = torch.stack(mapList, dim=1)
        device_id = maps.get_device()
        map_size = maps.size(-1)

        maps_grouped = []

        for i in range(len(self.dp2smpl_mapping)):
            maps_grouped_i = maps[:, :, self.dp2smpl_mapping[i], :, :]

            # add back ground
            maps_bg = torch.zeros(maps_grouped_i[:, :, 0].size()).cuda(device_id)
            maps_bg[:, -1] = torch.sum(maps_grouped_i[:, -1], dim=1) < 0.5
            maps_bg = maps_bg.unsqueeze(2)

            maps_grouped_i = torch.cat([maps_bg, maps_grouped_i], dim=2)

            maps_grouped_i = maps_grouped_i.view(-1, n_channels * (len(self.dp2smpl_mapping[i]) + 1), map_size,
                                                 map_size)
            maps_grouped.append(maps_grouped_i)

        return maps_grouped

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


class SMPL_Head(nn.Module):
    def __init__(self, as_renderer_only=False, orig_size=224, feat_in_dim=None):
        super(SMPL_Head, self).__init__()

        self.mapping_to_detectron = None
        self.orphans_in_detectron = None
        self.orig_size = orig_size

        K = np.array([[560., 0., 112.],
                      [0., 560., 112.],
                      [0., 0., 1.]])

        ##  x90 * z90 * y flip
        R = np.array([[0., 1., 0.],
                      [0., 0., -1.],
                      [1., 0., 0.]])

        ## h36m
        R = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])

        ## backward
        R_b = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., -1.]])

        t = np.array([0, 0, 5])

        if self.orig_size != 224:
            rander_scale = self.orig_size / float(224)
            K[0, 0] *= rander_scale
            K[1, 1] *= rander_scale
            K[0, 2] *= rander_scale
            K[1, 2] *= rander_scale

        self.K = torch.FloatTensor(K[None, :, :])
        self.R = torch.FloatTensor(R[None, :, :])
        self.R_b = torch.FloatTensor(R_b[None, :, :])
        self.t = torch.FloatTensor(t[None, None, :])

        self.smpl = SMPL(model_type=cfg.DANET.SMPL_MODEL_TYPE, obj_saveable=True, max_batch_size=max(cfg.TRAIN.BATCH_SIZE, cfg.TEST.BATCH_SIZE))

        self.coco_plus2coco = [14, 15, 16, 17, 18, 9, 8, 10, 7, 11, 6, 3, 2, 4, 1, 5, 0]

        DP = dp_utils.DensePoseMethods()

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
                self.smpl_para_Outs = DecomposedPredictor(feat_in_dim)
            else:
                self.smpl_para_Outs = GlobalPredictor(feat_in_dim)


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
        target = in_dict['target'] if 'target' in in_dict else None
        target_kps = in_dict['target_kps'] if 'target_kps' in in_dict else None
        target_kps3d = in_dict['target_kps3d'] if 'target_kps3d' in in_dict else None
        kp3d_type = in_dict['kp3d_type'] if 'kp3d_type' in in_dict else None
        target_verts = in_dict['target_verts'] if 'target_verts' in in_dict else None
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

        # return_dict['losses']['stn_loc'] = smpl_out_dict['loss_stn']
        for k, v in smpl_out_dict['losses'].items():
            return_dict['losses'][k] = v

        loss_orth = self.orthogonal_loss(para)
        loss_orth *= cfg.DANET.ORTHOGONAL_WEIGHTS
        return_dict['losses']['Rs_orth'] = loss_orth

        if target is not None:
            if cfg.DANET.PARA_WEIGHTS > 0:
                loss_para = self.smpl_para_losses(para, target)
                loss_para *= cfg.DANET.PARA_WEIGHTS
                return_dict['losses']['smpl_para'] = loss_para

                if smpl_out_dict.has_key('smpl_pose'):
                    for stack_i in range(len(smpl_out_dict['smpl_pose'])):
                        loss_rot = self.smpl_para_losses(smpl_out_dict['smpl_pose'][stack_i], target[:, 13:])
                        loss_rot *= cfg.DANET.PARA_WEIGHTS
                        return_dict['losses']['smpl_rotation'+str(stack_i)] = loss_rot

            if cfg.DANET.DECOMPOSED and smpl_out_dict.has_key('smpl_coord') and cfg.DANET.SMPL_KPS_WEIGHTS > 0:
                gt_beta = target[:, 3:13].contiguous().detach()
                gt_Rs = target[:, 13:].contiguous().view(-1, 24, 3, 3).detach()
                smpl_pts = self.smpl(gt_beta, Rs=gt_Rs, get_skin=False, add_smpl_joint=True)
                gt_smpl_coord = smpl_pts['smpl']
                for stack_i in range(len(smpl_out_dict['smpl_coord'])):
                    loss_smpl_coord = F.l1_loss(smpl_out_dict['smpl_coord'][stack_i], gt_smpl_coord,
                                                size_average=False) / gt_smpl_coord.size(0)
                    loss_smpl_coord *= cfg.DANET.SMPL_KPS_WEIGHTS
                    return_dict['losses']['smpl_position'+str(stack_i)] = loss_smpl_coord

        if target_kps is not None:
            if isinstance(target_kps, np.ndarray):
                target_kps = torch.from_numpy(target_kps).cuda(device_id)
            target_kps_vis = target_kps[:, :, -1].unsqueeze(-1).expand(-1, -1, 2)

            cam_gt = target[:, :3].detach() if target is not None else None
            shape_gt = target[:, 3:13].detach() if target is not None else None
            pose_gt = target[:, 13:].detach() if target is not None else None
            loss_proj_kps, loss_kps3d, loss_verts, proj_kps = self.projection_losses(para, target_kps, target_kps_vis, target_kps3d, kp3d_type, target_verts, cam_gt=cam_gt, shape_gt=shape_gt, pose_gt=pose_gt)

            if cfg.DANET.PROJ_KPS_WEIGHTS > 0:
                loss_proj_kps *= cfg.DANET.PROJ_KPS_WEIGHTS
                return_dict['losses']['proj_kps'] = loss_proj_kps

            if cfg.DANET.KPS3D_WEIGHTS > 0 and loss_kps3d is not None:
                loss_kps3d *= cfg.DANET.KPS3D_WEIGHTS
                return_dict['losses']['kps_3d'] = loss_kps3d

            if loss_verts is not None and cfg.DANET.VERTS_WEIGHTS > 0:
                loss_verts *= cfg.DANET.VERTS_WEIGHTS
                return_dict['losses']['smpl_verts'] = loss_verts

        return_dict['metrics']['none'] = loss_orth.detach()

        # pytorch0.4 bug on gathering scalar(0-dim) tensors
        for k, v in return_dict['losses'].items():
            if len(v.shape) == 0:
                return_dict['losses'][k] = v.unsqueeze(0)
        for k, v in return_dict['metrics'].items():
            if len(v.shape) == 0:
                return_dict['metrics'][k] = v.unsqueeze(0)

        return return_dict

    def smpl_para_losses(self, pred, target):

        # return F.smooth_l1_loss(pred, target)
        return F.l1_loss(pred, target)

    def orthogonal_loss(self, para):
        device_id = para.get_device()
        Rs_pred = para[:, 13:].contiguous().view(-1, 3, 3)
        Rs_pred_transposed = torch.transpose(Rs_pred, 1, 2)
        Rs_mm = torch.bmm(Rs_pred, Rs_pred_transposed)
        tensor_eyes = torch.eye(3).expand_as(Rs_mm).cuda(device_id)
        return F.mse_loss(Rs_mm, tensor_eyes)

    def projection_losses(self, para, target_kps, target_kps_vis=None, target_kps3d=None, kp3d_type=None, target_verts=None, cam_gt=None, shape_gt=None, pose_gt=None):
        device_id = para.get_device()

        batch_size = para.size(0)

        def weighted_l1_loss(input, target, weights=1, size_average=True):
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
            verts_pose, proj_kps = ps_dict['verts'], ps_dict['proj_kps']
            ps_dict_shape = self.smpl2kps(beta, Rs_gt, K, R, t, dist_coeffs)
            verts_shape = ps_dict_shape['verts']
        else:
            ps_dict = self.smpl2kps(beta, Rs, K, R, t, dist_coeffs)
            verts, proj_kps = ps_dict['verts'], ps_dict['proj_kps']


        if target_kps.size(1) == 14:
            proj_kps_pred = proj_kps[:, :14, :]
        elif target_kps.size(1) == 17:
            proj_kps_pred = proj_kps[:, self.coco_plus2coco, :]

        if target_kps3d is not None and cfg.DANET.KPS3D_WEIGHTS > 0 and torch.sum(kp3d_type==1) > 0:
            kps3d_from_smpl = ps_dict['cocoplus_kps'][:, :14]
            target_kps3d -= torch.mean(target_kps3d[:, [2, 3]], dim=1).unsqueeze(1)
            if kp3d_type is None:
                loss_kps3d = weighted_l1_loss(kps3d_from_smpl, target_kps3d)
            else:
                loss_kps3d = weighted_l1_loss(kps3d_from_smpl[kp3d_type==1], target_kps3d[kp3d_type==1])
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
        return_dict = {}
        if orig_size is None:
            orig_size = self.orig_size
        if theta is None:
            smpl_pts = self.smpl(beta, Rs=Rs, get_skin=True, add_smpl_joint=add_smpl_joint)
        else:
            smpl_pts = self.smpl(beta, theta=theta, get_skin=True, add_smpl_joint=add_smpl_joint)
        verts = smpl_pts['verts']
        kps = smpl_pts['cocoplus']
        joint_pelvis = torch.mean(kps[:, [2, 3]], dim=1).unsqueeze(1)
        verts = verts - joint_pelvis
        kps = kps - joint_pelvis
        return_dict['cocoplus_kps'] = kps
        if add_smpl_joint:
            joint3d_smpl = smpl_pts['smpl']
            joint3d_smpl = joint3d_smpl - joint_pelvis
        else:
            joint3d_smpl = None
        if selected_ind is not None:
            kps = kps[:, selected_ind]
        proj_kps = nr_projection(kps, K=K, R=R, t=t, dist_coeffs=dist_coeffs, orig_size=orig_size)
        proj_kps[:, :, 1] *= -1
        proj_kps[:, :, :2] *= cfg.DANET.HEATMAP_SIZE / 2.
        proj_kps[:, :, :2] += cfg.DANET.HEATMAP_SIZE / 2.

        return_dict['verts'] = verts
        return_dict['proj_kps'] = proj_kps
        return_dict['joint3d_smpl'] = joint3d_smpl

        return return_dict

    def make_uv_image(self, pose=None, beta=None, cam=None, Rs=None, joint_only=False, joint_type='coco',
                      add_smpl_joint=False, backward=False):
        batch_size = beta.size(0)
        if beta.is_cuda:
            dist_coeffs = torch.cuda.FloatTensor([[0.] * 5])
        else:
            dist_coeffs = torch.FloatTensor([[0.] * 5])

        K, R, t = self.camera_matrix(cam, backward)

        info_dict = {}

        if joint_type == 'smpl':
            add_smpl_joint = True

        if Rs is None:
            ps_dict = self.smpl2kps(beta, Rs, K, R, t, dist_coeffs, add_smpl_joint=add_smpl_joint,
                                                          theta=pose)
        else:
            ps_dict = self.smpl2kps(beta, Rs, K, R, t, dist_coeffs, add_smpl_joint=add_smpl_joint)
        verts, joint2d, joint3d_smpl = ps_dict['verts'], ps_dict['proj_kps'], ps_dict['joint3d_smpl']

        if add_smpl_joint or joint_type == 'smpl':
            joint_smpl = nr_projection(joint3d_smpl, K=K, R=R, t=t, dist_coeffs=dist_coeffs,
                                       orig_size=self.orig_size)
            joint_smpl[:, :, 1] *= -1
            joint_smpl = joint_smpl
        else:
            joint_smpl = None

        if joint_type == 'coco_plus':
            joint2d = joint2d
        elif joint_type == 'lsp':
            joint2d = joint2d[:, :14, :]
        elif joint_type == 'coco':
            joint2d = joint2d[:, self.coco_plus2coco, :]
        elif joint_type == 'smpl':
            joint2d = joint_smpl.clone()
            joint2d[:, :, :2] *= cfg.DANET.HEATMAP_SIZE / 2.
            joint2d[:, :, :2] += cfg.DANET.HEATMAP_SIZE / 2.
        joint2d[:, :, -1] = 1.
        joint2d = joint2d

        if self.vert_mapping is None:
            vertices = verts
        else:
            vertices = verts[:, self.vert_mapping, :]

        if joint_only:
            render_image = None
        else:
            device_id = beta.get_device()

            render_image = self.renderer(vertices, self.faces.cuda(device_id).expand(batch_size, -1, -1),
                                   self.textures.cuda(device_id).expand(batch_size, -1, -1, -1, -1, -1),
                                   K=K, R=R, t=t,
                                   mode='rgb',
                                   dist_coeffs=torch.FloatTensor([[0.] * 5]).cuda(device_id))

        info_dict['verts'] = verts
        info_dict['cocoplus_kps'] = ps_dict['cocoplus_kps']
        info_dict['joint3d_smpl'] = ps_dict['joint3d_smpl']

        info_dict['render_image'] = render_image
        info_dict['joint2d'] = joint2d
        info_dict['joint_smpl'] = joint_smpl

        return info_dict

    def kp_projection(self, cam, joint3d, backward=False):
        if joint3d.is_cuda:
            dist_coeffs = torch.cuda.FloatTensor([[0.] * 5])
        else:
            dist_coeffs = torch.FloatTensor([[0.] * 5])

        K, R, t = self.camera_matrix(cam, backward)

        pr_joint = nr_projection(joint3d, K=K, R=R, t=t, dist_coeffs=dist_coeffs,
                                 orig_size=self.orig_size)
        pr_joint[:, :, 1] *= -1
        pr_joint[:, :, :2] *= cfg.DANET.HEATMAP_SIZE / 2.
        pr_joint[:, :, :2] += cfg.DANET.HEATMAP_SIZE / 2.

        return pr_joint

    def camera_matrix(self, cam, backward=False):
        batch_size = cam.size(0)

        K = self.K.repeat(batch_size, 1, 1)
        if backward:
            R = self.R_b.repeat(batch_size, 1, 1)
        else:
            R = self.R.repeat(batch_size, 1, 1)
        t = self.t.repeat(batch_size, 1, 1)

        if cam.is_cuda:
            device_id = cam.get_device()
            K = K.cuda(device_id)
            R = R.cuda(device_id)
            t = t.cuda(device_id)

        K[:, 0, 0] *= cam[:, 0]
        K[:, 1, 1] *= cam[:, 0]
        K[:, [0, 1], [2, 2]] += K[:, [0, 1], [0, 1]] * cam[:, [1, 2]]

        return K, R, t

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
    def __init__(self, feat_in_dim=None):
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
    def __init__(self, feat_in_dim=None):
        super(DecomposedPredictor, self).__init__()

        # For cache
        self.mapping_to_detectron = None
        self.orphans_in_detectron = None

        self.add_feat_ch = feat_in_dim

        self.smpl_parents = smpl_structure('smpl_parents')
        self.smpl_children = smpl_structure('smpl_children')
        if cfg.DANET.PART_IUV_SIMP:
            self.dp2smpl_mapping = smpl_structure('dp2smpl_mapping')
        else:
            self.dp2smpl_mapping = [range(1, 25)] * 24

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

            self.pose_regressors = nn.ModuleList()
            for stack_i in range(1+cfg.DANET.REFINEMENT.STACK_NUM):
                self.pose_regressors.append(nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(self.rot_feat_len * 24, 9 * 24, kernel_size=1, groups=24)
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

            self.pose_regressors = nn.ModuleList()

            self.pose_regressors.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(self.rot_feat_len * 1 * 24, 9 * 24, kernel_size=1, groups=24)
            )
            )
            self.pose_regressors.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(self.rot_feat_len * 24, 9 * 24, kernel_size=1, groups=24)
            )
            )

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
                if cfg.DANET.REFINEMENT.POS_RES:
                    for i in range(24):
                        pos_feats[i] = pos_feats[i].repeat(1, 2, 1, 1) + pos_feats_refined[i]
                else:
                    pos_feats = pos_feats_refined

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

                if cfg.DANET.REFINEMENT.ROT_RES:
                    rot_feats = rot_feats + tran_rot_feats
                    part_feats = rot_feats.view(rot_feats.size(0), 24 * rot_feats.size(2), 1, 1)
                else:
                    part_feats = tran_rot_feats.contiguous().view(tran_rot_feats.size(0), 24 * tran_rot_feats.size(2), 1, 1)

                local_para = self.pose_regressors[s_i+1](part_feats).view(nbs, 24, -1)
                smpl_pose = local_para.view(local_para.size(0), -1)

        elif cfg.DANET.REFINE_STRATEGY == 'gcn':

            return_dict['smpl_coord'] = []
            return_dict['smpl_pose'] = []

            if self.training:
                local_para = self.pose_regressors[0](rot_feats.view(rot_feats.size(0), 24 * rot_feats.size(2), 1, 1)).view(
                    nbs, 24, -1)
                smpl_pose = local_para.view(local_para.size(0), -1)
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

                if cfg.DANET.REFINEMENT.POS_RES:
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

            if cfg.DANET.REFINEMENT.ROT_RES:
                rot_feats = rot_feats + tran_rot_feats
                part_feats = rot_feats.view(rot_feats.size(0), 24 * rot_feats.size(2), 1, 1)
            else:
                part_feats = tran_rot_feats.view(tran_rot_feats.size(0), 24 * tran_rot_feats.size(2), 1, 1)

            local_para = self.pose_regressors[-1](part_feats).view(nbs, 24, -1)
            smpl_pose = local_para.view(local_para.size(0), -1)

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

                if cfg.DANET.REFINEMENT.POS_RES:
                    l_pos_feat = pos_feats_init + l_pos_feat

                pos_feats_refined = l_pos_feat
                tran_rot_feats = pos_feats_refined.unsqueeze(-1).unsqueeze(-1)

                if cfg.DANET.REFINEMENT.ROT_RES:
                    rot_feats = rot_feats + tran_rot_feats
                    part_feats = rot_feats.view(rot_feats.size(0), 24 * rot_feats.size(2), 1, 1)
                else:
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
