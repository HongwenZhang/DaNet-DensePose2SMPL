"""
This script can be used to evaluate a trained model on 3D pose/shape and masks/part segmentation. You first need to download the datasets and preprocess them.
Example usage:
```
python3 eval_coco.py --checkpoint=data/pretrained_model/danet_model_h36m_itw.pt
```
Running the above command will compute the 2D keypoint detection AP on COCO.
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
import argparse
from tqdm import tqdm
import time

import path_config
import constants
from models import hmr, SMPL
from models.danet import DaNet

from datasets import COCODataset


from utils.geometry import perspective_projection
from utils.transforms import transform_preds

from models.core.config import cfg, cfg_from_file
from easydict import EasyDict

import logging
logger = logging.getLogger(__name__)

# Define command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default=None, help='Path to network checkpoint')
parser.add_argument('--batch_size', default=16, type=int, help='Batch size for testing')
parser.add_argument('--shuffle', default=False, action='store_true', help='Shuffle data')
parser.add_argument('--num_workers', default=8, type=int, help='Number of processes for data loading')
parser.add_argument('--result_file', default=None, help='If set, save detections to a .npz file')
parser.add_argument('--regressor', type=str, default='danet', choices=['hmr', 'danet'], help='Name of the SMPL regressor.')
parser.add_argument('--danet_cfg_file', type=str, default='./configs/danet_h36m_itw.yaml', help='path to config file.')
parser.add_argument('--output_dir', type=str, default='./output', help='output directory.')


def run_evaluation(model, dataset, result_file,
                   batch_size=32, img_res=224, 
                   num_workers=32, shuffle=False, options=None):
    """Run evaluation on the datasets and metrics we report in the paper. """

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Transfer model to the GPU
    model.to(device)

    # Load SMPL model
    smpl_neutral = SMPL(path_config.SMPL_MODEL_DIR,
                        create_transl=False).to(device)

    save_results = result_file is not None
    # Disable shuffling if you want to save the results
    if save_results:
        shuffle = False
    # Create dataloader for the dataset
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    # Store SMPL parameters
    smpl_pose = np.zeros((len(dataset), 72))
    smpl_betas = np.zeros((len(dataset), 10))
    smpl_camera = np.zeros((len(dataset), 3))
    pred_joints = np.zeros((len(dataset), 17, 3))

    num_joints = 17

    num_samples = len(dataset)
    print('dataset length: {}'.format(num_samples))
    all_preds = np.zeros(
        (num_samples, num_joints, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()

        for step, batch in enumerate(tqdm(data_loader, desc='Eval', total=len(data_loader))):
            images = batch['img'].to(device)
            scale = batch['scale'].numpy()
            center = batch['center'].numpy()

            num_images = images.size(0)

            gt_keypoints_2d = batch['keypoints']  # 2D keypoints
            # De-normalize 2D keypoints from [-1,1] to pixel space
            gt_keypoints_2d_orig = gt_keypoints_2d.clone()
            gt_keypoints_2d_orig[:, :, :-1] = 0.5 * img_res * (gt_keypoints_2d_orig[:, :, :-1] + 1)

            if options.regressor == 'hmr':
                pred_rotmat, pred_betas, pred_camera = model(images)
            elif options.regressor == 'danet':
                danet_pred_dict = model.infer_net(images)
                para_pred = danet_pred_dict['para']
                pred_camera = para_pred[:, 0:3].contiguous()
                pred_betas = para_pred[:, 3:13].contiguous()
                pred_rotmat = para_pred[:, 13:].contiguous().view(-1, 24, 3, 3)

            pred_output = smpl_neutral(betas=pred_betas, body_pose=pred_rotmat[:, 1:],
                                       global_orient=pred_rotmat[:, 0].unsqueeze(1), pose2rot=False)

            # pred_vertices = pred_output.vertices
            pred_J24 = pred_output.joints[:, -24:]
            pred_JCOCO = pred_J24[:, constants.J24_TO_JCOCO]

            # Convert Weak Perspective Camera [s, tx, ty] to camera translation [tx, ty, tz] in 3D given the bounding box size
            # This camera translation can be used in a full perspective projection
            pred_cam_t = torch.stack([pred_camera[:,1],
                                      pred_camera[:,2],
                                      2*constants.FOCAL_LENGTH/(img_res * pred_camera[:, 0] +1e-9)],dim=-1)

            camera_center = torch.zeros(len(pred_JCOCO), 2, device=pred_camera.device)
            pred_keypoints_2d = perspective_projection(pred_JCOCO,
                                                       rotation=torch.eye(3, device=pred_camera.device).unsqueeze(0).expand(len(pred_JCOCO), -1, -1),
                                                       translation=pred_cam_t,
                                                       focal_length=constants.FOCAL_LENGTH,
                                                       camera_center=camera_center)

            coords = pred_keypoints_2d + (img_res / 2.)
            coords = coords.cpu().numpy()
            # Normalize keypoints to [-1,1]
            # pred_keypoints_2d = pred_keypoints_2d / (img_res / 2.)

            gt_keypoints_coco = gt_keypoints_2d_orig[:, -24:][:, constants.J24_TO_JCOCO]

            preds = coords.copy()

            scale_ = np.array([scale, scale]).transpose()

            # Transform back
            for i in range(coords.shape[0]):
                preds[i] = transform_preds(
                    coords[i], center[i], scale_[i], [img_res, img_res]
                )

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = 1.
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = center[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = scale_[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(scale_*200, 1)
            all_boxes[idx:idx + num_images, 5] = 1.
            image_path.extend(batch['imgname'])

            idx += num_images

        ckp_name = options.regressor
        name_values, perf_indicator = dataset.evaluate(
            all_preds, options.output_dir, all_boxes, image_path, ckp_name,
            filenames, imgnums
        )

        model_name = options.regressor
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

    # Save reconstructions to a file for further processing
    if save_results:
        np.savez(result_file, pred_joints=pred_joints, pose=smpl_pose, betas=smpl_betas, camera=smpl_camera)


# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    print(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    print('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    print(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )


if __name__ == '__main__':
    args = parser.parse_args()

    # load danet configures
    cfg_from_file(args.danet_cfg_file)
    cfg.DANET.REFINEMENT = EasyDict(cfg.DANET.REFINEMENT)
    cfg.MSRES_MODEL.EXTRA = EasyDict(cfg.MSRES_MODEL.EXTRA)

    if args.regressor == 'hmr':
        model = hmr(path_config.SMPL_MEAN_PARAMS)
    elif args.regressor == 'danet':
        model = DaNet(args, path_config.SMPL_MEAN_PARAMS)

    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model'], strict=False)

    model.eval()

    dataset = COCODataset(None, 'coco', 'val2014', is_train=False)
    # Run evaluation
    run_evaluation(model, dataset, args.result_file,
                   batch_size=args.batch_size,
                   shuffle=args.shuffle, options=args)
