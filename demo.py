"""
This script can be used to produce demo results.
Example usage:
```
python3 demo.py --checkpoint=data/pretrained_model/danet_model_h36m_itw.pt --img_dir ./examples --use_opendr
```
"""
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

from models import SMPL
from models.core.config import cfg, cfg_from_file
from models.danet import DaNet
from skimage.transform import resize
from utils.iuvmap import iuv_map2img

import path_config

import logging
logger = logging.getLogger(__name__)


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='DaNet for 3D Human Shape and Pose')

    parser.add_argument(
        '--cfg', dest='cfg_file', default='configs/danet_h36m_itw.yaml',
        help='config file for training / testing')
    parser.add_argument(
        '--checkpoint', help='checkpoint path to load')
    parser.add_argument(
        '--img_dir', default='./examples', type=str, help='path to test images')
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
    cfg.batch_size = 1

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if cfg.DANET.SMPL_MODEL_TYPE == 'male':
        smpl_male = SMPL(path_config.SMPL_MODEL_DIR,
                         gender='male',
                         create_transl=False).to(device)
        smpl = smpl_male
    elif cfg.DANET.SMPL_MODEL_TYPE == 'neutral':
        smpl_neutral = SMPL(path_config.SMPL_MODEL_DIR,
                            create_transl=False).to(device)
        smpl = smpl_neutral
    elif cfg.DANET.SMPL_MODEL_TYPE == 'female':
        smpl_female = SMPL(path_config.SMPL_MODEL_DIR,
                           gender='female',
                           create_transl=False).to(device)
        smpl = smpl_female

    if args.use_opendr:
        from utils.opendr_render import opendr_render
        dr_render = opendr_render(ratio=1)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    ### Model ###
    model = DaNet(cfg, path_config.SMPL_MEAN_PARAMS).to(device)

    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()

    img_path_list = [os.path.join(args.img_dir, name) for name in os.listdir(args.img_dir) if name.endswith('.jpg')]
    for i, path in enumerate(img_path_list):

        image = Image.open(path).convert('RGB')
        img_id = path.split('/')[-1][:-4]

        image_tensor = torchvision.transforms.ToTensor()(image).unsqueeze(0).cuda()

        # run inference
        pred_dict = model.infer_net(image_tensor)
        para_pred = pred_dict['para']
        camera_pred = para_pred[:, 0:3].contiguous()
        betas_pred = para_pred[:, 3:13].contiguous()
        rotmat_pred = para_pred[:, 13:].contiguous().view(-1, 24, 3, 3)

        # input image
        image_np = image_tensor[0].cpu().numpy()
        image_np = np.transpose(image_np, (1, 2, 0))

        ones_np = np.ones(image_np.shape[:2]) * 255
        ones_np = ones_np[:, :, None]

        image_in_rgba = np.concatenate((image_np, ones_np), axis=2)

        # estimated global IUV
        global_iuv = iuv_map2img(*pred_dict['visualization']['iuv_pred'])[0].cpu().numpy()
        global_iuv = np.transpose(global_iuv, (1, 2, 0))
        global_iuv = resize(global_iuv, image_np.shape[:2])
        global_iuv_rgba = np.concatenate((global_iuv, ones_np), axis=2)

        # estimated patial IUV
        part_iuv_pred = pred_dict['visualization']['part_iuv_pred'][0]
        p_iuv_vis = []
        for i in range(part_iuv_pred.size(0)):
            p_u_vis, p_v_vis, p_i_vis = [part_iuv_pred[i, iuv].unsqueeze(0) for iuv in range(3)]
            if p_u_vis.size(1) == 25:
                p_iuv_vis_i = iuv_map2img(p_u_vis.detach(), p_v_vis.detach(), p_i_vis.detach())
            else:
                p_iuv_vis_i = iuv_map2img(p_u_vis.detach(), p_v_vis.detach(), p_i_vis.detach(),
                                          ind_mapping=[0] + model.img2iuv.dp2smpl_mapping[i])
            p_iuv_vis.append(p_iuv_vis_i)
        part_iuv = torch.cat(p_iuv_vis, dim=0)
        part_iuv = make_grid(part_iuv, nrow=6, padding=0).cpu().numpy()
        part_iuv = np.transpose(part_iuv, (1, 2, 0))
        part_iuv_rgba = np.concatenate((part_iuv, np.ones(part_iuv.shape[:2])[:, :, None] * 255),
                                           axis=2)

        # rendered IUV of the predicted SMPL model
        smpl_output = smpl(betas=betas_pred, body_pose=rotmat_pred[:, 1:],
                                   global_orient=rotmat_pred[:, 0].unsqueeze(1), pose2rot=False)
        verts_pred = smpl_output.vertices
        render_iuv = model.iuv2smpl.verts2uvimg(verts_pred[0].unsqueeze(0), cam=camera_pred[0].unsqueeze(0))
        render_iuv = render_iuv[0].cpu().numpy()

        render_iuv = np.transpose(render_iuv, (1, 2, 0))
        render_iuv = resize(render_iuv, image_np.shape[:2])

        img_render_iuv = image_np.copy()
        img_render_iuv[render_iuv > 0] = render_iuv[render_iuv > 0]

        img_render_iuv_rgba = np.concatenate((img_render_iuv, ones_np), axis=2)

        img_vis_list = [image_in_rgba, global_iuv_rgba, part_iuv_rgba, img_render_iuv_rgba]

        if args.use_opendr:
            # visualize the predicted SMPL model using the opendr renderer
            K = model.iuv2smpl.K[0].cpu().numpy()
            _, _, img_smpl, smpl_rgba = dr_render.render(image_tensor[0].cpu().numpy(),
                                                         camera_pred[0].cpu().numpy(), K,
                                                         verts_pred.cpu().numpy(),
                                                         smpl_neutral.faces)

            img_smpl_rgba = np.concatenate((img_smpl, ones_np), axis=2)
            img_vis_list.extend([img_smpl_rgba, smpl_rgba])

        img_vis = np.concatenate(img_vis_list, axis=1)
        img_vis[img_vis < 0.0] = 0.0
        img_vis[img_vis > 1.0] = 1.0
        imsave(os.path.join(args.out_dir, img_id + '_result.png'), img_vis)


if __name__ == '__main__':
    main()
