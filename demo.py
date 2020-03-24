from __future__ import absolute_import
from __future__ import division

import argparse
import os
import logging
from easydict import EasyDict
from PIL import Image

import matplotlib
matplotlib.use('Agg')

from matplotlib.image import imsave

import numpy as np

import torch
import torchvision
from torchvision.utils import make_grid

import lib.nn as mynn
import lib.utils.net as net_utils
from lib.core.config import cfg, cfg_from_file
from lib.modeling.danet import DaNet
from lib.utils.logging import setup_logging
from skimage.transform import resize
from lib.utils.iuvmap import iuv_map2img

# Set up logging and load config options
logger = setup_logging(__name__)


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='DaNet for 3D Human Shape and Pose')

    parser.add_argument(
        '--cfg', dest='cfg_file', default='configs/smpl/smpl_encoder.yaml',
        help='config file for training / testing')

    parser.add_argument(
        '--load_ckpt', help='checkpoint path to load')

    parser.add_argument(
        '--img_dir', default='./imgs', type=str, help='path to test images')

    parser.add_argument(
        '--out_dir', default='./output', type=str, help='path to output results')

    parser.add_argument(
        '--use_opendr', help='use opendr renderer to visualize results', action='store_true')

    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()

    cfg_from_file(args.cfg_file)

    cfg.DANET.REFINEMENT = EasyDict(cfg.DANET.REFINEMENT)
    cfg.MSRES_MODEL.EXTRA = EasyDict(cfg.MSRES_MODEL.EXTRA)

    if args.use_opendr:
        from lib.utils.opendr_render import opendr_render
        if cfg.DANET.SMPL_MODEL_TYPE == 'male':
            smpl_model_path = './data/SMPL_data/basicmodel_m_lbs_10_207_0_v1.0.0.pkl'
        elif cfg.DANET.SMPL_MODEL_TYPE == 'neutral':
            smpl_model_path = './data/SMPL_data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'
        elif cfg.DANET.SMPL_MODEL_TYPE == 'female':
            smpl_model_path = './data/SMPL_data/basicModel_f_lbs_10_207_0_v1.0.0.pkl'
        dr_render = opendr_render(model_path=smpl_model_path)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    ### Model ###
    model = DaNet().cuda()

    ### Load checkpoint
    if args.load_ckpt:
        load_name = args.load_ckpt
        logging.info("loading checkpoint %s", load_name)
        checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage)
        net_utils.load_ckpt(model, checkpoint)
        del checkpoint

    model = mynn.DataParallel(model, minibatch=False)
    model.eval()

    img_path_list = [os.path.join(args.img_dir, name) for name in os.listdir(args.img_dir) if name.endswith('.jpg')]
    for i, path in enumerate(img_path_list):

        image = Image.open(path).convert('RGB')
        img_id = path.split('/')[-1][:-4]

        image_tensor = torchvision.transforms.ToTensor()(image).unsqueeze(0).cuda()

        # run inference
        pred_results = model.module.infer_net(image_tensor)

        para_pred = pred_results['para']

        cam_pred = para_pred[:, 0:3].contiguous()
        beta_pred = para_pred[:, 3:13].contiguous()
        Rs_pred = para_pred[:, 13:].contiguous().view(-1, 24, 3, 3)

        smpl_pts = model.module.iuv2smpl.smpl(beta_pred, Rs=Rs_pred, get_skin=True)
        kps3ds_pred = smpl_pts['cocoplus']
        vert_pred = smpl_pts['verts']

        # input image
        image_np = image_tensor[0].cpu().numpy()
        image_np = np.transpose(image_np, (1, 2, 0))

        ones_np = np.ones(image_np.shape[:2]) * 255
        ones_np = ones_np[:, :, None]

        image_in_rgba = np.concatenate((image_np, ones_np), axis=2)

        # estimated global IUV
        global_iuv = iuv_map2img(*pred_results['visualization']['iuv_pred'])[0].cpu().numpy()
        global_iuv = np.transpose(global_iuv, (1, 2, 0))
        global_iuv = resize(global_iuv, image_np.shape[:2])
        global_iuv_rgba = np.concatenate((global_iuv, ones_np), axis=2)

        # estimated patial IUV
        part_iuv_pred = pred_results['visualization']['part_iuv_pred'][0]
        p_iuv_vis = []
        for i in range(part_iuv_pred.size(0)):
            p_u_vis, p_v_vis, p_i_vis = [part_iuv_pred[i, iuv].unsqueeze(0) for iuv in range(3)]
            if p_u_vis.size(1) == 25:
                p_iuv_vis_i = iuv_map2img(p_u_vis.detach(), p_v_vis.detach(), p_i_vis.detach())
            else:
                p_iuv_vis_i = iuv_map2img(p_u_vis.detach(), p_v_vis.detach(), p_i_vis.detach(),
                                          ind_mapping=[0] + model.module.img2iuv.dp2smpl_mapping[i])
            p_iuv_vis.append(p_iuv_vis_i)
        part_iuv = torch.cat(p_iuv_vis, dim=0)
        part_iuv = make_grid(part_iuv, nrow=6, padding=0).cpu().numpy()
        part_iuv = np.transpose(part_iuv, (1, 2, 0))
        part_iuv_rgba = np.concatenate((part_iuv, np.ones(part_iuv.shape[:2])[:, :, None] * 255),
                                           axis=2)

        # rendered IUV of the predicted SMPL model
        smpl_projection = model.module.iuv2smpl.make_uv_image(Rs=Rs_pred[0].unsqueeze(0), beta=beta_pred[0].unsqueeze(0),
                                                             cam=cam_pred[0].unsqueeze(0), add_smpl_joint=True)
        render_iuv = smpl_projection['render_image'].squeeze(0).cpu().numpy()
        render_iuv = np.transpose(render_iuv, (1, 2, 0))
        render_iuv = resize(render_iuv, image_np.shape[:2])

        img_render_iuv = image_np.copy()
        img_render_iuv[render_iuv > 0] = render_iuv[render_iuv > 0]

        img_render_iuv_rgba = np.concatenate((img_render_iuv, ones_np), axis=2)

        img_vis_list = [image_in_rgba, global_iuv_rgba, part_iuv_rgba, img_render_iuv_rgba]

        if args.use_opendr:
            # visualize the predicted SMPL model using the opendr renderer
            joint_pelvis = torch.mean(kps3ds_pred[:, [2, 3]], dim=1).unsqueeze(1)
            vert_centered = vert_pred - joint_pelvis

            K, _, _ = model.module.iuv2smpl.camera_matrix(cam_pred)
            _, _, img_smpl, smpl_rgba = dr_render.render(
                image_tensor[0].cpu().numpy(), K.cpu().numpy(),
                vert_centered[0].cpu().numpy())

            img_smpl_rgba = np.concatenate((img_smpl, ones_np), axis=2)

            img_vis_list.extend([img_smpl_rgba, smpl_rgba])

        img_vis = np.concatenate(img_vis_list, axis=1)
        img_vis[img_vis<0.0] = 0.0
        img_vis[img_vis>1.0] = 1.0
        # omit the forth channel (alpha channel) [transparency]
        imsave(os.path.join(args.out_dir, img_id + '_result.png'), img_vis[:,:,:3])


if __name__ == '__main__':
    main()
