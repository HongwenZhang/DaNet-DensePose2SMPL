import os
import torch
from models.core.config import cfg
import pickle
from utils.smpl_utlis import smpl_structure
import torch.nn as nn
import torch.nn.functional as F

from utils.keypoints import softmax_integral_tensor, generate_heatmap
import utils.net as net_utils
from utils.iuvmap import iuv_img2map, iuv_map2img, iuvmap_clean

from models.module.hr_module import PoseHighResolutionNet
from models.module.res_module import PoseResNet


class IUV_Estimator(nn.Module):
    def __init__(self, pretrained=True):
        super(IUV_Estimator, self).__init__()

        if cfg.DANET.USE_LEARNED_RATIO:
            with open(os.path.join('./data/pretrained_model', 'learned_ratio.pkl'), 'rb') as f:
                # pretrain_ratio = pickle.load(f)
                pretrain_ratio = pickle.load(f, encoding='iso-8859-1')

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
        self.dp2smpl_mapping = smpl_structure('dp2smpl_mapping')

        if cfg.DANET.INPUT_MODE not in ['iuv_gt']:
            part_out_dim = 1 + len(self.dp2smpl_mapping[0])
            if cfg.DANET.IUV_REGRESSOR in ['resnet']:
                self.iuv_est = PoseResNet(part_out_dim=part_out_dim)
                if pretrained:
                    self.iuv_est.init_weights(cfg.MSRES_MODEL.PRETRAINED)
            elif cfg.DANET.IUV_REGRESSOR in ['hrnet']:
                self.iuv_est = PoseHighResolutionNet(part_out_dim=part_out_dim)
                if pretrained:
                    if cfg.HR_MODEL.PRETR_SET in ['imagenet']:
                        self.iuv_est.init_weights(cfg.HR_MODEL.PRETRAINED_IM)
                    elif cfg.HR_MODEL.PRETR_SET in ['coco']:
                        self.iuv_est.init_weights(cfg.HR_MODEL.PRETRAINED_COCO)

        self.bodyfeat_channels = 1024
        self.part_channels = 256

    def forward(self, data, iuv_image_gt=None, smpl_kps_gt=None, kps3d_gt=None, uvia_dp_gt=None, has_iuv=None, has_dp=None):
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

        if self.training and iuv_image_gt is not None:
            uvia_list = iuv_img2map(iuv_image_gt)
            loss_U, loss_V, loss_IndexUV, loss_segAnn = self.body_uv_losses(u_pred, v_pred, index_pred, ann_pred,
                                                                            uvia_list, has_iuv)
            return_dict['losses']['loss_U'] = loss_U
            return_dict['losses']['loss_V'] = loss_V
            return_dict['losses']['loss_IndexUV'] = loss_IndexUV
            return_dict['losses']['loss_segAnn'] = loss_segAnn

        if self.training and uvia_dp_gt is not None:
            if torch.sum(has_dp) > 0:
                dp_on = (has_dp == 1)
                uvia_dp_gt_ = {k: v[dp_on] if isinstance(v, torch.Tensor) else v for k, v in uvia_dp_gt.items()}
                loss_Udp, loss_Vdp, loss_IndexUVdp, loss_segAnndp = self.dp_uvia_losses(u_pred[dp_on], v_pred[dp_on],
                                                                                        index_pred[dp_on],
                                                                                        ann_pred[dp_on], **uvia_dp_gt_)
                return_dict['losses']['loss_Udp'] = loss_Udp
                return_dict['losses']['loss_Vdp'] = loss_Vdp
                return_dict['losses']['loss_IndexUVdp'] = loss_IndexUVdp
                return_dict['losses']['loss_segAnndp'] = loss_segAnndp
            else:
                return_dict['losses']['loss_Udp'] = torch.zeros(1).to(data.device)
                return_dict['losses']['loss_Vdp'] = torch.zeros(1).to(data.device)
                return_dict['losses']['loss_IndexUVdp'] = torch.zeros(1).to(data.device)
                return_dict['losses']['loss_segAnndp'] = torch.zeros(1).to(data.device)

        return_dict['uvia_pred'] = [u_pred, v_pred, index_pred, ann_pred]

        if cfg.DANET.DECOMPOSED:

            u_pred_cl, v_pred_cl, index_pred_cl, ann_pred_cl = iuvmap_clean(u_pred, v_pred, index_pred, ann_pred)

            partial_decon_feat = uv_est_dic['xd']

            skps_hm_pred = uv_est_dic['predict_hm']

            smpl_kps_hm_size = skps_hm_pred.size(-1)

            return_dict['skps_hm_pred'] = skps_hm_pred.detach()

            stn_centers = softmax_integral_tensor(10 * skps_hm_pred, skps_hm_pred.size(1), skps_hm_pred.size(-2),
                                                  skps_hm_pred.size(-1))
            stn_centers /= 0.5 * smpl_kps_hm_size
            stn_centers -= 1

            if self.training and smpl_kps_gt is not None:
                if cfg.DANET.STN_HM_WEIGHTS > 0:
                    smpl_kps_norm = smpl_kps_gt.detach().clone()
                    # [-1, 1]  ->  [0, 1]
                    smpl_kps_norm[:, :, :2] *= 0.5
                    smpl_kps_norm[:, :, :2] += 0.5
                    smpl_kps_norm = smpl_kps_norm.view(smpl_kps_norm.size(0) * smpl_kps_norm.size(1), -1)[:, :2]
                    skps_hm_gt, _ = generate_heatmap(smpl_kps_norm, heatmap_size=cfg.DANET.HEATMAP_SIZE)
                    skps_hm_gt = skps_hm_gt.view(smpl_kps_gt.size(0), smpl_kps_gt.size(1), cfg.BODY_UV_RCNN.HEATMAP_SIZE,
                                                cfg.DANET.HEATMAP_SIZE)
                    skps_hm_gt = skps_hm_gt.detach()
                    return_dict['skps_hm_gt'] = skps_hm_gt.detach()

                    loss_stnhm = F.smooth_l1_loss(skps_hm_pred, skps_hm_gt, size_average=True)  # / smpl_kps_gt.size(0)
                    loss_stnhm *= cfg.DANET.STN_HM_WEIGHTS
                    return_dict['losses']['loss_stnhm'] = loss_stnhm

                if cfg.DANET.STN_KPS_WEIGHTS > 0:
                    if smpl_kps_gt.shape[-1] == 3:
                        loss_roi = 0
                        for w in torch.unique(smpl_kps_gt[:, :, 2]):
                            if w == 0:
                                continue
                            kps_w_idx = smpl_kps_gt[:, :, 2] == w
                            # stn_centers_target = smpl_kps_gt[:, :, :2][kps_w1_idx]
                            loss_roi += F.smooth_l1_loss(stn_centers[kps_w_idx], smpl_kps_gt[:, :, :2][kps_w_idx], size_average=False) * w
                        loss_roi /= smpl_kps_gt.size(0)

                        loss_roi *= cfg.DANET.STN_KPS_WEIGHTS
                        return_dict['losses']['loss_roi'] = loss_roi

                if cfg.DANET.STN_CENTER_JITTER > 0:
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

            ## partial uv losses
            if self.training and iuv_image_gt is not None:
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
                                                                                      part_uvia_list, has_iuv)

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


    def body_uv_losses(self, u_pred, v_pred, index_pred, ann_pred, uvia_list, has_iuv=None):
        batch_size = u_pred.size(0)
        device = u_pred.device

        Umap, Vmap, Imap, Annmap = uvia_list

        if has_iuv is not None:
            if torch.sum(has_iuv) > 0:
                u_pred, v_pred, index_pred = u_pred[has_iuv], v_pred[has_iuv], index_pred[has_iuv]
                ann_pred = ann_pred[has_iuv] if ann_pred is not None else ann_pred
                Umap, Vmap, Imap = Umap[has_iuv], Vmap[has_iuv], Imap[has_iuv]
                Annmap = Annmap[has_iuv] if Annmap is not None else Annmap
            else:
                return (torch.zeros(1).to(device), torch.zeros(1).to(device), torch.zeros(1).to(device), torch.zeros(1).to(device))

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
