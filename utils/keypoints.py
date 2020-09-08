# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Keypoint utilities (somewhat specific to COCO keypoints)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def get_keypoints():
    """Get the COCO keypoints and their left/right flip coorespondence map."""
    # Keypoints are not available in the COCO json for the test split, so we
    # provide them here.
    keypoints = [
        'nose',
        'left_eye',
        'right_eye',
        'left_ear',
        'right_ear',
        'left_shoulder',
        'right_shoulder',
        'left_elbow',
        'right_elbow',
        'left_wrist',
        'right_wrist',
        'left_hip',
        'right_hip',
        'left_knee',
        'right_knee',
        'left_ankle',
        'right_ankle'
    ]
    keypoint_flip_map = {
        'left_eye': 'right_eye',
        'left_ear': 'right_ear',
        'left_shoulder': 'right_shoulder',
        'left_elbow': 'right_elbow',
        'left_wrist': 'right_wrist',
        'left_hip': 'right_hip',
        'left_knee': 'right_knee',
        'left_ankle': 'right_ankle'
    }
    return keypoints, keypoint_flip_map


def get_person_class_index():
    """Index of the person class in COCO."""
    return 1


def flip_keypoints(keypoints, keypoint_flip_map, keypoint_coords, width):
    """Left/right flip keypoint_coords. keypoints and keypoint_flip_map are
    accessible from get_keypoints().
    """
    flipped_kps = keypoint_coords.copy()
    for lkp, rkp in keypoint_flip_map.items():
        lid = keypoints.index(lkp)
        rid = keypoints.index(rkp)
        flipped_kps[:, :, lid] = keypoint_coords[:, :, rid]
        flipped_kps[:, :, rid] = keypoint_coords[:, :, lid]

    # Flip x coordinates
    flipped_kps[:, 0, :] = width - flipped_kps[:, 0, :] - 1
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = np.where(flipped_kps[:, 2, :] == 0)
    flipped_kps[inds[0], 0, inds[1]] = 0
    return flipped_kps


def flip_heatmaps(heatmaps):
    """Flip heatmaps horizontally."""
    keypoints, flip_map = get_keypoints()
    heatmaps_flipped = heatmaps.copy()
    for lkp, rkp in flip_map.items():
        lid = keypoints.index(lkp)
        rid = keypoints.index(rkp)
        heatmaps_flipped[:, rid, :, :] = heatmaps[:, lid, :, :]
        heatmaps_flipped[:, lid, :, :] = heatmaps[:, rid, :, :]
    heatmaps_flipped = heatmaps_flipped[:, :, :, ::-1]
    return heatmaps_flipped


def heatmaps_to_keypoints(maps, rois):
    """Extract predicted keypoint locations from heatmaps. Output has shape
    (#rois, 4, #keypoints) with the 4 rows corresponding to (x, y, logit, prob)
    for each keypoint.
    """
    # This function converts a discrete image coordinate in a HEATMAP_SIZE x
    # HEATMAP_SIZE image to a continuous keypoint coordinate. We maintain
    # consistency with keypoints_to_heatmap_labels by using the conversion from
    # Heckbert 1990: c = d + 0.5, where d is a discrete coordinate and c is a
    # continuous coordinate.
    offset_x = rois[:, 0]
    offset_y = rois[:, 1]

    widths = rois[:, 2] - rois[:, 0]
    heights = rois[:, 3] - rois[:, 1]
    widths = np.maximum(widths, 1)
    heights = np.maximum(heights, 1)
    widths_ceil = np.ceil(widths)
    heights_ceil = np.ceil(heights)

    # NCHW to NHWC for use with OpenCV
    maps = np.transpose(maps, [0, 2, 3, 1])
    min_size = cfg.KRCNN.INFERENCE_MIN_SIZE
    xy_preds = np.zeros(
        (len(rois), 4, cfg.KRCNN.NUM_KEYPOINTS), dtype=np.float32)
    for i in range(len(rois)):
        if min_size > 0:
            roi_map_width = int(np.maximum(widths_ceil[i], min_size))
            roi_map_height = int(np.maximum(heights_ceil[i], min_size))
        else:
            roi_map_width = widths_ceil[i]
            roi_map_height = heights_ceil[i]
        width_correction = widths[i] / roi_map_width
        height_correction = heights[i] / roi_map_height
        roi_map = cv2.resize(
            maps[i], (roi_map_width, roi_map_height),
            interpolation=cv2.INTER_CUBIC)
        # Bring back to CHW
        roi_map = np.transpose(roi_map, [2, 0, 1])
        roi_map_probs = scores_to_probs(roi_map.copy())
        w = roi_map.shape[2]
        for k in range(cfg.KRCNN.NUM_KEYPOINTS):
            pos = roi_map[k, :, :].argmax()
            x_int = pos % w
            y_int = (pos - x_int) // w
            assert (roi_map_probs[k, y_int, x_int] ==
                    roi_map_probs[k, :, :].max())
            x = (x_int + 0.5) * width_correction
            y = (y_int + 0.5) * height_correction
            xy_preds[i, 0, k] = x + offset_x[i]
            xy_preds[i, 1, k] = y + offset_y[i]
            xy_preds[i, 2, k] = roi_map[k, y_int, x_int]
            xy_preds[i, 3, k] = roi_map_probs[k, y_int, x_int]

    return xy_preds


def keypoints_to_heatmap_labels(keypoints, rois):
    """Encode keypoint location in the target heatmap for use in
    SoftmaxWithLoss.
    """
    # Maps keypoints from the half-open interval [x1, x2) on continuous image
    # coordinates to the closed interval [0, HEATMAP_SIZE - 1] on discrete image
    # coordinates. We use the continuous <-> discrete conversion from Heckbert
    # 1990 ("What is the coordinate of a pixel?"): d = floor(c) and c = d + 0.5,
    # where d is a discrete coordinate and c is a continuous coordinate.
    assert keypoints.shape[2] == cfg.KRCNN.NUM_KEYPOINTS

    shape = (len(rois), cfg.KRCNN.NUM_KEYPOINTS)
    heatmaps = blob_utils.zeros(shape)
    weights = blob_utils.zeros(shape)

    offset_x = rois[:, 0]
    offset_y = rois[:, 1]
    scale_x = cfg.KRCNN.HEATMAP_SIZE / (rois[:, 2] - rois[:, 0])
    scale_y = cfg.KRCNN.HEATMAP_SIZE / (rois[:, 3] - rois[:, 1])

    for kp in range(keypoints.shape[2]):
        vis = keypoints[:, 2, kp] > 0
        x = keypoints[:, 0, kp].astype(np.float32)
        y = keypoints[:, 1, kp].astype(np.float32)
        # Since we use floor below, if a keypoint is exactly on the roi's right
        # or bottom boundary, we shift it in by eps (conceptually) to keep it in
        # the ground truth heatmap.
        x_boundary_inds = np.where(x == rois[:, 2])[0]
        y_boundary_inds = np.where(y == rois[:, 3])[0]
        x = (x - offset_x) * scale_x
        x = np.floor(x)
        if len(x_boundary_inds) > 0:
            x[x_boundary_inds] = cfg.KRCNN.HEATMAP_SIZE - 1

        y = (y - offset_y) * scale_y
        y = np.floor(y)
        if len(y_boundary_inds) > 0:
            y[y_boundary_inds] = cfg.KRCNN.HEATMAP_SIZE - 1

        valid_loc = np.logical_and(
            np.logical_and(x >= 0, y >= 0),
            np.logical_and(
                x < cfg.KRCNN.HEATMAP_SIZE, y < cfg.KRCNN.HEATMAP_SIZE))

        valid = np.logical_and(valid_loc, vis)
        valid = valid.astype(np.int32)

        lin_ind = y * cfg.KRCNN.HEATMAP_SIZE + x
        heatmaps[:, kp] = lin_ind * valid
        weights[:, kp] = valid

    return heatmaps, weights


def scores_to_probs(scores):
    """Transforms CxHxW of scores to probabilities spatially."""
    channels = scores.shape[0]
    for c in range(channels):
        temp = scores[c, :, :]
        max_score = temp.max()
        temp = np.exp(temp - max_score) / np.sum(np.exp(temp - max_score))
        scores[c, :, :] = temp
    return scores


def nms_oks(kp_predictions, rois, thresh):
    """Nms based on kp predictions."""
    scores = np.mean(kp_predictions[:, 2, :], axis=1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ovr = compute_oks(
            kp_predictions[i], rois[i], kp_predictions[order[1:]],
            rois[order[1:]])
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def compute_oks(src_keypoints, src_roi, dst_keypoints, dst_roi):
    """Compute OKS for predicted keypoints wrt gt_keypoints.
    src_keypoints: 4xK
    src_roi: 4x1
    dst_keypoints: Nx4xK
    dst_roi: Nx4
    """

    sigmas = np.array([
        .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87,
        .87, .89, .89]) / 10.0
    vars = (sigmas * 2)**2

    # area
    src_area = (src_roi[2] - src_roi[0] + 1) * (src_roi[3] - src_roi[1] + 1)

    # measure the per-keypoint distance if keypoints visible
    dx = dst_keypoints[:, 0, :] - src_keypoints[0, :]
    dy = dst_keypoints[:, 1, :] - src_keypoints[1, :]

    e = (dx**2 + dy**2) / vars / (src_area + np.spacing(1)) / 2
    e = np.sum(np.exp(-e), axis=1) / e.shape[1]

    return e


def generate_heatmap(joints, heatmap_size, sigma=1, joints_vis=None):
    '''
    param joints:  [num_joints, 3]
    param joints_vis: [num_joints, 3]
    return: target, target_weight(1: visible, 0: invisible)
    '''
    num_joints = joints.shape[0]
    device = joints.device
    cur_device = torch.device(device.type, device.index)
    if not hasattr(heatmap_size, '__len__'):
        # width  height
        heatmap_size = [heatmap_size, heatmap_size]
    assert len(heatmap_size) == 2
    target_weight = np.ones((num_joints, 1), dtype=np.float32)
    if joints_vis is not None:
        target_weight[:, 0] = joints_vis[:, 0]
    target = torch.zeros((num_joints,
                          heatmap_size[1],
                          heatmap_size[0]),
                         dtype=torch.float32,
                         device=cur_device)

    tmp_size = sigma * 3

    for joint_id in range(num_joints):
        mu_x = int(joints[joint_id][0] * heatmap_size[0] + 0.5)
        mu_y = int(joints[joint_id][1] * heatmap_size[1] + 0.5)
        # Check that any part of the gaussian is in-bounds
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
        if ul[0] >= heatmap_size[0] or ul[1] >= heatmap_size[1] \
                or br[0] < 0 or br[1] < 0:
            # If not, just return the image as is
            target_weight[joint_id] = 0
            continue

        # # Generate gaussian
        size = 2 * tmp_size + 1
        # x = np.arange(0, size, 1, np.float32)
        # y = x[:, np.newaxis]
        # x0 = y0 = size // 2
        # # The gaussian is not normalized, we want the center value to equal 1
        # g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
        # g = torch.from_numpy(g.astype(np.float32))

        x = torch.arange(0, size, dtype=torch.float32, device=cur_device)
        y = x.unsqueeze(-1)
        x0 = y0 = size // 2
        # The gaussian is not normalized, we want the center value to equal 1
        g = torch.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

        # Usable gaussian range
        g_x = max(0, -ul[0]), min(br[0], heatmap_size[0]) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], heatmap_size[1]) - ul[1]
        # Image range
        img_x = max(0, ul[0]), min(br[0], heatmap_size[0])
        img_y = max(0, ul[1]), min(br[1], heatmap_size[1])

        v = target_weight[joint_id]
        if v > 0.5:
            target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

    return target, target_weight


def generate_3d_integral_preds_tensor(heatmaps, num_joints, x_dim, y_dim, z_dim):
    assert isinstance(heatmaps, torch.Tensor)

    if z_dim is not None:
        heatmaps = heatmaps.reshape((heatmaps.shape[0], num_joints, z_dim, y_dim, x_dim))

        accu_x = heatmaps.sum(dim=2)
        accu_x = accu_x.sum(dim=2)
        accu_y = heatmaps.sum(dim=2)
        accu_y = accu_y.sum(dim=3)
        accu_z = heatmaps.sum(dim=3)
        accu_z = accu_z.sum(dim=3)

        accu_x = accu_x * torch.cuda.comm.broadcast(torch.arange(x_dim, dtype=torch.float32), devices=[accu_x.device.index])[0]
        accu_y = accu_y * torch.cuda.comm.broadcast(torch.arange(y_dim, dtype=torch.float32), devices=[accu_y.device.index])[0]
        accu_z = accu_z * torch.cuda.comm.broadcast(torch.arange(z_dim, dtype=torch.float32), devices=[accu_z.device.index])[0]

        accu_x = accu_x.sum(dim=2, keepdim=True)
        accu_y = accu_y.sum(dim=2, keepdim=True)
        accu_z = accu_z.sum(dim=2, keepdim=True)
    else:
        heatmaps = heatmaps.reshape((heatmaps.shape[0], num_joints, y_dim, x_dim))

        accu_x = heatmaps.sum(dim=2)
        accu_y = heatmaps.sum(dim=3)

        accu_x = accu_x * torch.cuda.comm.broadcast(torch.arange(x_dim, dtype=torch.float32), devices=[accu_x.device.index])[0]
        accu_y = accu_y * torch.cuda.comm.broadcast(torch.arange(y_dim, dtype=torch.float32), devices=[accu_y.device.index])[0]

        accu_x = accu_x.sum(dim=2, keepdim=True)
        accu_y = accu_y.sum(dim=2, keepdim=True)
        accu_z = None

    return accu_x, accu_y, accu_z


# integral pose estimation
# https://github.com/JimmySuen/integral-human-pose/blob/99647e40ec93dfa4e3b6a1382c935cebb35440da/pytorch_projects/common_pytorch/common_loss/integral.py#L28
def softmax_integral_tensor(preds, num_joints, hm_width, hm_height, hm_depth=None):
    # global soft max
    preds = preds.reshape((preds.shape[0], num_joints, -1))
    preds = F.softmax(preds, 2)

    output_3d = False if hm_depth is None else True

    # integrate heatmap into joint location
    if output_3d:
        x, y, z = generate_3d_integral_preds_tensor(preds, num_joints, hm_width, hm_height, hm_depth)
        # x = x / float(hm_width) - 0.5
        # y = y / float(hm_height) - 0.5
        # z = z / float(hm_depth) - 0.5
        preds = torch.cat((x, y, z), dim=2)
        # preds = preds.reshape((preds.shape[0], num_joints * 3))
    else:
        x, y, _ = generate_3d_integral_preds_tensor(preds, num_joints, hm_width, hm_height, z_dim=None)
        # x = x / float(hm_width) - 0.5
        # y = y / float(hm_height) - 0.5
        preds = torch.cat((x, y), dim=2)
        # preds = preds.reshape((preds.shape[0], num_joints * 2))

    return preds
