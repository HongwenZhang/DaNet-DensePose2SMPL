from functools import wraps
import torch
from models.core.config import cfg
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils.iuvmap import iuv_img2map, iuv_map2img, iuvmap_clean
from utils.geometry import batch_rodrigues

from .iuv_estimator import IUV_Estimator
from .smpl_regressor import SMPL_Regressor


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


class DaNet(nn.Module):
    '''
    Ref.
    H. Zhang et al. DaNet: Decompose-and-aggregate Network for 3D Human Shape and Pose Estimation
    H. Zhang et al. Learning 3D Human Shape and Pose from Dense Body Parts
    '''
    def __init__(self, options, smpl_mean_params):
        """
        Initialize all the mean of - step.

        Args:
            self: (todo): write your description
            options: (dict): write your description
            smpl_mean_params: (dict): write your description
        """
        super(DaNet, self).__init__()

        self.options = options

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
        self.iuv2smpl = SMPL_Regressor(options=options, feat_in_dim=final_feat_dim, smpl_mean_params=smpl_mean_params)

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
        if 'part_iuv_pred' in uv_return_dict:
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

        if 'part_featmaps' in uv_return_dict:
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
        Forward the forward_dict

        Args:
            self: (todo): write your description
            in_dict: (dict): write your description
        """

        if type(in_dict) is not dict:
            in_dict = {'img': in_dict, 'pretrain_mode': False, 'vis_on': False, 'dataset_name': ''}

        image = in_dict['img']
        gt_pose = in_dict['opt_pose'] if 'opt_pose' in in_dict else None  # SMPL pose parameters
        gt_betas = in_dict['opt_betas'] if 'opt_betas' in in_dict else None  # SMPL beta parameters
        target_kps = in_dict['target_kps'] if 'target_kps' in in_dict else None
        target_kps3d = in_dict['target_kps3d'] if 'target_kps3d' in in_dict else None
        has_iuv = in_dict['has_iuv'].byte() if 'has_iuv' in in_dict else None
        has_dp = in_dict['has_dp'].byte() if 'has_dp' in in_dict else None
        has_kp3d = in_dict['has_pose_3d'].byte() if 'has_pose_3d' in in_dict else None  # flag that indicates whether 3D pose is valid
        target_smpl_kps = in_dict['target_smpl_kps'] if 'target_smpl_kps' in in_dict else None
        target_verts = in_dict['target_verts'] if 'target_verts' in in_dict else None
        valid_fit = in_dict['valid_fit'] if 'valid_fit' in in_dict else None

        batch_size = image.shape[0]

        if gt_pose is not None:
            gt_rotmat = batch_rodrigues(gt_pose.view(-1, 3)).view(-1, 24 * 3 * 3)
            target_cam = in_dict['target_cam']
            target = torch.cat([target_cam, gt_betas, gt_rotmat], dim=1)
            uv_image_gt = torch.zeros((batch_size, 3, cfg.DANET.HEATMAP_SIZE, cfg.DANET.HEATMAP_SIZE)).to(image.device)
            if torch.sum(has_iuv) > 0:
                uv_image_gt[has_iuv] = self.iuv2smpl.verts2uvimg(target_verts[has_iuv], cam=target_cam[has_iuv])  # [B, 3, 56, 56]
        else:
            target = None

        # target_iuv_dp = in_dict['target_iuv_dp'] if 'target_iuv_dp' in in_dict else None
        target_iuv_dp = in_dict['dp_dict'] if 'dp_dict' in in_dict else None

        if 'target_kps_coco' in in_dict:
            target_kps = in_dict['target_kps_coco']

        return_dict = {}
        return_dict['losses'] = {}
        return_dict['metrics'] = {}
        return_dict['visualization'] = {}
        return_dict['prediction'] = {}

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
            uv_return_dict = self.img2iuv(image, uv_image_gt, target_smpl_kps, pretrained=in_dict['pretrain_mode'], uvia_dp_gt=target_iuv_dp, has_iuv=has_iuv, has_dp=has_dp)

        u_pred, v_pred, index_pred, ann_pred = uv_return_dict['uvia_pred']
        if self.training and cfg.DANET.PART_IUV_ZERO > 0:
            zero_idxs = []
            for bs in range(u_pred.shape[0]):
                zero_idxs.append([int(i) + 1 for i in torch.nonzero(torch.rand(24) < cfg.DANET.PART_IUV_ZERO)])

        if self.training and cfg.DANET.PART_IUV_ZERO > 0:
            for bs in range(len(zero_idxs)):
                u_pred[bs, zero_idxs[bs]] *= 0
                v_pred[bs, zero_idxs[bs]] *= 0
                index_pred[bs, zero_idxs[bs]] *= 0

        u_pred_cl, v_pred_cl, index_pred_cl, ann_pred_cl = iuvmap_clean(u_pred, v_pred, index_pred, ann_pred)

        iuv_pred_clean = [u_pred_cl.detach(), v_pred_cl.detach(), index_pred_cl.detach(), ann_pred_cl.detach()]
        return_dict['visualization']['iuv_pred'] = iuv_pred_clean

        if in_dict['vis_on']:
            uvi_pred_clean = [u_pred_cl.detach(), v_pred_cl.detach(), index_pred_cl.detach(), ann_pred_cl.detach()]
            return_dict['visualization']['pred_uv'] = iuv_map2img(*uvi_pred_clean)
            return_dict['visualization']['gt_uv'] = uv_image_gt
            if 'stn_kps_pred' in uv_return_dict:
                return_dict['visualization']['stn_kps_pred'] = uv_return_dict['stn_kps_pred']

            # index_pred_cl shape:  2, 25, 56, 56
            return_dict['visualization']['index_sum'] = [torch.sum(index_pred_cl[:, 1:].detach()).unsqueeze(0), np.prod(index_pred_cl[:, 0].shape)]

            for key in ['skps_hm_pred', 'skps_hm_gt']:
                if key in uv_return_dict:
                    return_dict['visualization'][key] = torch.max(uv_return_dict[key], dim=1)[0].unsqueeze(1)
                    return_dict['visualization'][key][return_dict['visualization'][key] > 1] = 1.
                    skps_hm_vis = uv_return_dict[key]
                    skps_hm_vis = skps_hm_vis.reshape((skps_hm_vis.shape[0], skps_hm_vis.shape[1], -1))
                    skps_hm_vis = F.softmax(skps_hm_vis, 2)
                    skps_hm_vis = skps_hm_vis.reshape(skps_hm_vis.shape[0], skps_hm_vis.shape[1],
                                                      cfg.DANET.HEATMAP_SIZE, cfg.DANET.HEATMAP_SIZE)
                    return_dict['visualization'][key + '_soft'] = torch.sum(skps_hm_vis, dim=1).unsqueeze(1)
            # for key in ['part_uvi_pred', 'part_uvi_gt']:
            for key in ['part_uvi_gt']:
                if key in uv_return_dict:
                    part_uvi_pred_vis = uv_return_dict[key][0]
                    p_uvi_vis = []
                    for i in range(part_uvi_pred_vis.size(0)):
                        p_u_vis, p_v_vis, p_i_vis = [part_uvi_pred_vis[i, uvi].unsqueeze(0) for uvi in range(3)]
                        if p_u_vis.size(1) == 25:
                            p_uvi_vis_i = iuv_map2img(p_u_vis.detach(), p_v_vis.detach(), p_i_vis.detach())
                        else:
                            p_uvi_vis_i = iuv_map2img(p_u_vis.detach(), p_v_vis.detach(), p_i_vis.detach(),
                                                         ind_mapping=[0] + self.img2iuv.dp2smpl_mapping[i])
                        # p_uvi_vis_i = uvmap_vis(p_u_vis.detach(), p_v_vis.detach(), p_i_vis.detach(), self.img2iuv.dp2smpl_mapping[i])
                        p_uvi_vis.append(p_uvi_vis_i)
                    return_dict['visualization'][key] = torch.cat(p_uvi_vis, dim=0)

        if not in_dict['pretrain_mode']:

            iuv_map = torch.cat([u_pred_cl, v_pred_cl, index_pred_cl], dim=1)

            if cfg.DANET.INPUT_MODE in ['iuv_gt', 'iuv_gt_feat'] and 'part_iuv_gt' in uv_return_dict:
                part_iuv_map = uv_return_dict['part_iuv_gt']
                if self.training and cfg.DANET.PART_IUV_ZERO > 0:
                    for bs in range(len(zero_idxs)):
                        zero_channel = []
                        for zero_i in zero_idxs[bs]:
                            zero_channel.extend(
                                [(i, m_i + 1) for i, mapping in enumerate(self.img2iuv.dp2smpl_mapping) for m_i, map_idx in
                                 enumerate(mapping) if map_idx == zero_i])
                        zero_dp_i = [iterm[0] for iterm in zero_channel]
                        zero_p_i = [iterm[1] for iterm in zero_channel]
                        part_iuv_map[bs, zero_dp_i, :, zero_p_i] *= 0

                part_index_map = part_iuv_map[:, :, 2]
            elif 'part_iuv_pred' in uv_return_dict:
                part_iuv_pred = uv_return_dict['part_iuv_pred']
                if self.training and cfg.DANET.PART_IUV_ZERO > 0:
                    for bs in range(len(zero_idxs)):
                        zero_channel = []
                        for zero_i in zero_idxs[bs]:
                            zero_channel.extend(
                                [(i, m_i + 1) for i, mapping in enumerate(self.img2iuv.dp2smpl_mapping) for m_i, map_idx in
                                 enumerate(mapping) if map_idx == zero_i])
                        zero_dp_i = [iterm[0] for iterm in zero_channel]
                        zero_p_i = [iterm[1] for iterm in zero_channel]
                        part_iuv_pred[bs, zero_dp_i, :, zero_p_i] *= 0

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

            if 'part_featmaps' in uv_return_dict:
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
                                                 'has_kp3d': has_kp3d
                                                  })
            elif cfg.DANET.INPUT_MODE in ['iuv_feat', 'iuv_gt_feat']:
                smpl_return_dict = self.iuv2smpl({'iuv_map': {'iuv': iuv_map, 'feat': uv_return_dict['global_featmaps']},
                                                 'part_iuv_map': {'piuv': part_iuv_map, 'pfeat': part_feat_map},
                                                 'target': target,
                                                 'target_kps': target_kps,
                                                 'target_verts': target_verts,
                                                 'target_kps3d': target_kps3d,
                                                 'has_kp3d': has_kp3d
                                                  })
            elif cfg.DANET.INPUT_MODE in ['iuv', 'iuv_gt']:
                smpl_return_dict = self.iuv2smpl({'iuv_map': iuv_map,
                                                 'part_iuv_map': part_iuv_map,
                                                 'target': target,
                                                 'target_kps': target_kps,
                                                 'target_verts': target_verts,
                                                 'target_kps3d': target_kps3d,
                                                 'has_kp3d': has_kp3d,
                                                 'has_smpl': valid_fit
                                                  })
            elif cfg.DANET.INPUT_MODE == 'seg':
                # REMOVE _.detach
                smpl_return_dict = self.iuv2smpl({'iuv_map': {'index': index_pred_cl},
                                                 'part_iuv_map': {'pindex': part_index_map},
                                                 'target': target,
                                                 'target_kps': target_kps,
                                                 'target_verts': target_verts,
                                                 'target_kps3d': target_kps3d,
                                                 'has_kp3d': has_kp3d
                                                  })

            if in_dict['vis_on'] and part_index_map is not None:
                # part_index_map: 2, 24, 7, 56, 56
                return_dict['visualization']['p_index_sum'] = [torch.sum(part_index_map[:, :, 1:].detach()).unsqueeze(0),
                                                               np.prod(part_index_map[:, :, 0].shape)]

            if in_dict['vis_on'] and part_iuv_map is not None:
                part_uvi_pred_vis = part_iuv_map[0]
                p_uvi_vis = []
                for i in range(part_uvi_pred_vis.size(0)):
                    p_u_vis, p_v_vis, p_i_vis = [part_uvi_pred_vis[i, uvi].unsqueeze(0) for uvi in range(3)]
                    if p_u_vis.size(1) == 25:
                        p_uvi_vis_i = iuv_map2img(p_u_vis.detach(), p_v_vis.detach(), p_i_vis.detach())
                    else:
                        p_uvi_vis_i = iuv_map2img(p_u_vis.detach(), p_v_vis.detach(), p_i_vis.detach(),
                                                     ind_mapping=[0] + self.img2iuv.dp2smpl_mapping[i])
                    # p_uvi_vis_i = uvmap_vis(p_u_vis.detach(), p_v_vis.detach(), p_i_vis.detach(), self.img2iuv.dp2smpl_mapping[i])
                    p_uvi_vis.append(p_uvi_vis_i)
                return_dict['visualization']['part_uvi_pred'] = torch.cat(p_uvi_vis, dim=0)

        for key_name in ['losses', 'metrics', 'visualization', 'prediction']:
            if key_name in uv_return_dict:
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
        """
        Detects the weightronronronronron rule.

        Args:
            self: (todo): write your description
        """
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
