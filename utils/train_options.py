import os
import json
import argparse
import numpy as np
from collections import namedtuple
import socket
from datetime import datetime

import logging
logger = logging.getLogger(__name__)

class TrainOptions():

    def __init__(self):
        """
        Initialize the command line arguments.

        Args:
            self: (todo): write your description
        """
        self.parser = argparse.ArgumentParser()

        req = self.parser.add_argument_group('Required')
        req.add_argument('--name', required=True, help='Name of the experiment')

        gen = self.parser.add_argument_group('General')
        gen.add_argument('--time_to_run', type=int, default=np.inf, help='Total time to run in seconds. Used for training in environments with timing constraints')
        gen.add_argument('--resume', dest='resume', default=False, action='store_true', help='Resume from checkpoint (Use latest checkpoint by default')
        gen.add_argument('--num_workers', type=int, default=8, help='Number of processes used for data loading')
        pin = gen.add_mutually_exclusive_group()
        pin.add_argument('--pin_memory', dest='pin_memory', action='store_true')
        pin.add_argument('--no_pin_memory', dest='pin_memory', action='store_false')
        gen.set_defaults(pin_memory=True)

        io = self.parser.add_argument_group('io')
        io.add_argument('--log_dir', default='logs', help='Directory to store logs')
        io.add_argument('--checkpoint', default=None, help='Path to checkpoint')
        io.add_argument('--from_json', default=None, help='Load options from json file instead of the command line')
        io.add_argument('--pretrained_checkpoint', default=None, help='Load a pretrained checkpoint at the beginning training') 

        train = self.parser.add_argument_group('Training Options')
        train.add_argument('--num_epochs', type=int, default=65, help='Total number of training epochs')
        train.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
        train.add_argument('--regressor', type=str, default='hmr', help='Name of the SMPL regressor.')
        train.add_argument('--danet_cfg_file', type=str, default='./configs/danet_demo.yaml', help='config file path for DaNet.')
        train.add_argument('--hrnet_cfg_file', type=str, default='./configs/hr_config.yaml', help='config file path for HRNet.')
        train.add_argument('--batch_size', type=int, default=64, help='Batch size')
        train.add_argument('--summary_steps', type=int, default=100, help='Summary saving frequency')
        train.add_argument('--test_steps', type=int, default=1000, help='Testing frequency during training')
        train.add_argument('--checkpoint_steps', type=int, default=10000, help='Checkpoint saving frequency')
        train.add_argument('--img_res', type=int, default=224, help='Rescale bounding boxes to size [img_res, img_res] before feeding them in the network') 
        train.add_argument('--rot_factor', type=float, default=30, help='Random rotation in the range [-rot_factor, rot_factor]') 
        train.add_argument('--noise_factor', type=float, default=0.4, help='Randomly multiply pixel values with factor in the range [1-noise_factor, 1+noise_factor]') 
        train.add_argument('--scale_factor', type=float, default=0.25, help='Rescale bounding boxes by a factor of [1-scale_factor,1+scale_factor]') 
        train.add_argument('--ignore_3d', default=False, action='store_true', help='Ignore GT 3D data (for unpaired experiments') 
        train.add_argument('--shape_loss_weight', default=0, type=float, help='Weight of per-vertex loss') 
        train.add_argument('--keypoint_loss_weight', default=5., type=float, help='Weight of 2D and 3D keypoint loss') 
        train.add_argument('--pose_loss_weight', default=1., type=float, help='Weight of SMPL pose loss') 
        train.add_argument('--beta_loss_weight', default=0.001, type=float, help='Weight of SMPL betas loss') 
        train.add_argument('--openpose_train_weight', default=0., help='Weight for OpenPose keypoints during training') 
        train.add_argument('--gt_train_weight', default=1., help='Weight for GT keypoints during training') 
        train.add_argument('--run_smplify', default=False, action='store_true', help='Run SMPLify during training') 
        train.add_argument('--smplify_threshold', type=float, default=100., help='Threshold for ignoring SMPLify fits during training') 
        train.add_argument('--num_smplify_iters', default=50, type=int, help='Number of SMPLify iterations')
        train.add_argument('--vis_interval', default=2000, type=int, help='Visualize training info every N iterations')
        train.add_argument('--pretr_step', default=10000, type=int, help='Step number for pretraining uv estimator.')
        train.add_argument('--single_dataset', default=False, action='store_true', help='Use a single dataset')
        train.add_argument('--single_dataname', type=str, default='h36m', help='Name of the single dataset.')
        train.add_argument('--use_coco', default=False, action='store_true', help='Use the coco dataset')
        train.add_argument('--no_3dhp', default=False, action='store_true', help='Do not use the mpi-inf-3dhp dataset')
        train.add_argument('--timestamp', type=str, default='', help='timestamp for checkpoint path')
        train.add_argument('--daloss_only', default=False, action='store_true', help='use the losses defined in DaNet only')
        train.add_argument('--use_hmr_hr', default=False, action='store_true', help='use the HRNet for HMR')
        train.add_argument('--use_final_fit', default=False, action='store_true', help='Use the final fits')
        train.add_argument('--eval_pve', default=False, action='store_true', help='evaluate PVE')
        train.add_argument('--dp_itw', default=False, action='store_true', help='Use DensePose-COCO as the only in-the-wild data')

        shuffle_train = train.add_mutually_exclusive_group()
        shuffle_train.add_argument('--shuffle_train', dest='shuffle_train', action='store_true', help='Shuffle training data')
        shuffle_train.add_argument('--no_shuffle_train', dest='shuffle_train', action='store_false', help='Don\'t shuffle training data')
        shuffle_train.set_defaults(shuffle_train=True)

        misc = self.parser.add_argument_group('Misc Options')
        misc.add_argument('--misc', default=None, type=str, nargs="*",
                            help='other parameters')
        return

    def parse_args(self):
        """Parse input arguments."""
        self.args = self.parser.parse_args()
        # If config file is passed, override all arguments with the values from the config file
        if self.args.from_json is not None:
            path_to_json = os.path.abspath(self.args.from_json)
            with open(path_to_json, "r") as f:
                json_args = json.load(f)
                json_args = namedtuple("json_args", json_args.keys())(**json_args)
                return json_args
        else:
            if len(self.args.timestamp) == 0:
                timestamp = datetime.now().strftime('%b%d-%H-%M-%S')  # + '_' + socket.gethostname()
                self.args.timestamp = timestamp
            logger.info('log stamp: {}'.format(self.args.timestamp))
            self.args.log_dir = os.path.join(os.path.abspath(self.args.log_dir), self.args.name, 'log_' + self.args.timestamp)
            self.args.summary_dir = os.path.join(self.args.log_dir, 'tb_summary')
            if not os.path.exists(self.args.summary_dir):
                os.makedirs(self.args.summary_dir)
            self.args.checkpoint_dir = os.path.join(self.args.log_dir, 'checkpoints')
            if not os.path.exists(self.args.checkpoint_dir):
                os.makedirs(self.args.checkpoint_dir)
            self.save_dump()
            return self.args

    def save_dump(self):
        """Store all argument values to a json file.
        The default location is logs/expname/config.json.
        """
        if not os.path.exists(self.args.log_dir):
            os.makedirs(self.args.log_dir)
        with open(os.path.join(self.args.log_dir, "config.json"), "w") as f:
            json.dump(vars(self.args), f, indent=4)
        return
