from __future__ import absolute_import

import os
import numpy as np
import cv2
import scipy.misc
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# from .misc import *
# from .imutils import *

def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray


def im_to_numpy(img):
    img = to_numpy(img)
    img = np.transpose(img, (1, 2, 0)) # H*W*C
    return img


def im_to_torch(img):
    img = np.transpose(img, (2, 0, 1)) # C*H*W
    img = to_torch(img).float()
    if img.max() > 1:
        img /= 255
    return img


def color_normalize(x, mean, std):
    if x.size(0) == 1:
        x = x.repeat(3, 1, 1)

    for t, m, s in zip(x, mean, std):
        t.sub_(m)
    return x


def flip_back(flip_output, dataset='mpii'):
    """
    flip output map
    """
    if dataset ==  'mpii':
        matchedParts = (
            [0,5],   [1,4],   [2,3],
            [10,15], [11,14], [12,13]
        )
    else:
        print('Not supported dataset: ' + dataset)

    # flip output horizontally
    flip_output = fliplr(flip_output.numpy())

    # Change left-right parts
    for pair in matchedParts:
        tmp = np.copy(flip_output[:, pair[0], :, :])
        flip_output[:, pair[0], :, :] = flip_output[:, pair[1], :, :]
        flip_output[:, pair[1], :, :] = tmp

    return torch.from_numpy(flip_output).float()


def shufflelr(x, width, dataset='mpii'):
    """
    flip coords
    """
    if dataset ==  'mpii':
        matchedParts = (
            [0,5],   [1,4],   [2,3],
            [10,15], [11,14], [12,13]
        )
    else:
        print('Not supported dataset: ' + dataset)

    # Flip horizontal
    x[:, 0] = width - x[:, 0]

    # Change left-right parts
    for pair in matchedParts:
        tmp = x[pair[0], :].clone()
        x[pair[0], :] = x[pair[1], :]
        x[pair[1], :] = tmp

    return x


def fliplr(x):
    if x.ndim == 3:
        x = np.transpose(np.fliplr(np.transpose(x, (0, 2, 1))), (0, 2, 1))
    elif x.ndim == 4:
        for i in range(x.shape[0]):
            x[i] = np.transpose(np.fliplr(np.transpose(x[i], (0, 2, 1))), (0, 2, 1))
    return x.astype(float)


def get_transform(center, scale, res, rot=0):
    """
    General image processing functions
    """
    # Generate transformation matrix
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot # To match direction of rotation from cropping
        rot_mat = np.zeros((3,3))
        rot_rad = rot * np.pi / 180
        sn,cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0,:2] = [cs, -sn]
        rot_mat[1,:2] = [sn, cs]
        rot_mat[2,2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0,2] = -res[1]/2
        t_mat[1,2] = -res[0]/2
        t_inv = t_mat.copy()
        t_inv[:2,2] *= -1
        t = np.dot(t_inv,np.dot(rot_mat,np.dot(t_mat,t)))
    return t


def transform(pt, center, scale, res, invert=0, rot=0):
    # Transform pixel location to different reference
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int) + 1


def transform_pts(coords, center, scale, res, invert=0, rot=0):
    # size = coords.size()
    # coords = coords.view(-1, coords.size(-1))
    # print(coords.size())
    new_coords = coords.copy()
    for p in range(coords.shape[0]):
        new_coords[p, 0:2] = transform(coords[p, 0:2], center, scale, res, invert, rot)
    return new_coords

def get_box_xywh(pts):

    bbox = [[np.min(pts[i, :, 0]), np.min(pts[i, :, 1]), np.max(pts[i, :, 0]), np.max(pts[i, :, 1])]
            for i in range(pts.shape[0])]
    bbox = np.array(bbox)

    bbox[:, 2:] -= bbox[:, :2]

    return bbox

def crop_img(img, center, scale, res, rot=0):
    # img = im_to_numpy(img)
    center = center.copy()

    # Preprocessing for efficient cropping
    ht, wd = img.shape[0], img.shape[1]
    sf = scale * 200.0 / res[0]
    if sf < 2:
        sf = 1
    else:
        new_size = int(np.math.floor(max(ht, wd) / sf))
        new_ht = int(np.math.floor(ht / sf))
        new_wd = int(np.math.floor(wd / sf))
        if new_size < 2:
            return torch.zeros(res[0], res[1], img.shape[2]) \
                        if len(img.shape) > 2 else torch.zeros(res[0], res[1])
        else:
            img = scipy.misc.imresize(img, [new_ht, new_wd])
            center = center * 1.0 / sf
            scale = scale / sf

    # Upper left point
    ul = np.array(transform([0, 0], center, scale, res, invert=1))
    # Bottom right point
    br = np.array(transform(res, center, scale, res, invert=1))

    # Padding so that when rotated proper amount of context is included
    pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)
    if not rot == 0:
        ul -= pad
        br += pad

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape)

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])
    # try:
    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]
    # except:
    #     print(new_img.shape, img.shape)
    #     print(new_x, new_y, old_x, old_y, scale, center)

    if not rot == 0:
        # Remove padding
        new_img = scipy.misc.imrotate(new_img, rot)
        new_img = new_img[pad:-pad, pad:-pad]

    # new_img = im_to_torch(scipy.misc.imresize(new_img, res))
    new_img = scipy.misc.imresize(new_img, res)

    return new_img


def resize_img(img, scale_factor):
    new_size = (np.floor(np.array(img.shape[0:2]) * scale_factor)).astype(int)
    new_img = cv2.resize(img, (new_size[1], new_size[0]))
    # This is scale factor of [height, width] i.e. [y, x]
    actual_factor = [
        new_size[0] / float(img.shape[0]), new_size[1] / float(img.shape[1])
    ]
    return new_img, actual_factor


def scale_and_crop(image, scale, center, img_size):
    image_scaled, scale_factors = resize_img(image, scale)
    # Swap so it's [x, y]
    scale_factors = [scale_factors[1], scale_factors[0]]
    center_scaled = np.round(center * np.array(scale_factors)).astype(np.int)

    margin = int(img_size / 2)
    image_pad = np.pad(
        image_scaled, ((margin, ), (margin, ), (0, )), mode='edge')
    center_pad = center_scaled + margin
    # figure out starting point
    start_pt = center_pad - margin
    end_pt = center_pad + margin
    # crop:
    crop = image_pad[start_pt[1]:end_pt[1], start_pt[0]:end_pt[0], :]
    proc_param = {
        'scale': scale,
        'start_pt': start_pt,
        'end_pt': end_pt,
        'img_size': img_size
    }

    return crop, proc_param


def crop_box(img_shape, scale, center, crop_size):
    new_size = (np.floor(np.array(img_shape[0:2]) * scale)).astype(int)
    # new_img = cv2.resize(img, (new_size[1], new_size[0]))
    # This is scale factor of [height, width] i.e. [y, x]
    scale_factors = [
        new_size[0] / float(img_shape[0]), new_size[1] / float(img_shape[1])
    ]

    # Swap so it's [x, y]
    scale_factors = [scale_factors[1], scale_factors[0]]
    center_scaled = np.round(center * np.array(scale_factors)).astype(np.int)

    margin = int(crop_size / 2)
    # image_pad = np.pad(
    #     image_scaled, ((margin, ), (margin, ), (0, )), mode='edge')
    center_pad = center_scaled + margin
    # figure out starting point
    start_pt = center_pad - margin
    end_pt = center_pad + margin
    # crop:
    crop = image_pad[start_pt[1]:end_pt[1], start_pt[0]:end_pt[0], :]
    proc_param = {
        'scale': scale,
        'start_pt': start_pt,
        'end_pt': end_pt,
        'img_size': img_size
    }

    return crop, proc_param


def fill_img(ori_img, center, scale, new_img, rot=0):
    # img = im_to_numpy(img)

    img = ori_img[:]

    res = new_img.shape[:2]

    # Preprocessing for efficient cropping
    ht, wd = img.shape[0], img.shape[1]
    sf = scale * 200.0 / res[0]
    # if sf < 2:
    #     sf = 1
    # else:
    #     new_size = int(np.math.floor(max(ht, wd) / sf))
    #     new_ht = int(np.math.floor(ht / sf))
    #     new_wd = int(np.math.floor(wd / sf))
    #     if new_size < 2:
    #         return torch.zeros(res[0], res[1], img.shape[2]) \
    #                     if len(img.shape) > 2 else torch.zeros(res[0], res[1])
    #     else:
    #         img = scipy.misc.imresize(img, [new_ht, new_wd])
    #         center = center * 1.0 / sf
    #         scale = scale / sf

    # Upper left point
    ul = np.array(transform([0, 0], center, scale, res, invert=1))
    # Bottom right point
    br = np.array(transform(res, center, scale, res, invert=1))

    # Padding so that when rotated proper amount of context is included
    pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)
    if not rot == 0:
        ul -= pad
        br += pad

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    # new_img = np.zeros(new_shape)

    new_img = new_img[:]
    new_img[:, :, :3] *= 255

    new_img = scipy.misc.imresize(new_img, new_shape)

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])
    # new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]
    mask = new_img[new_y[0]:new_y[1], new_x[0]:new_x[1], 3]
    img[old_y[0]:old_y[1], old_x[0]:old_x[1]][mask>0] = new_img[new_y[0]:new_y[1], new_x[0]:new_x[1], :3][mask>0]

    return img


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
