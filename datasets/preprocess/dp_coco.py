import os
from os.path import join
import sys
import json
import numpy as np
# from .read_openpose import read_openpose
import utils.segms as segm_utils

def db_coco_extract(dataset_path, subset, out_path):

    # convert joints to global order
    joints_idx = [19, 20, 21, 22, 23, 9, 8, 10, 7, 11, 6, 3, 2, 4, 1, 5, 0]

    # bbox expansion factor
    scaleFactor = 1.2

    # structs we need
    imgnames_, scales_, centers_, parts_, smpl_2dkps_, dp_annot_ = [], [], [], [], [], []
    im_id_, id_ = [], []

    # subfolders for different subsets
    subfolders = {'train': 'train2014', 'minival': 'val2014', 'valminusminival': 'val2014', 'test': 'test2014'}

    # json annotation file
    json_path = os.path.join(dataset_path, 
                             'annotations', 
                             'densepose_coco_2014_{}.json'.format(subset))
    json_data = json.load(open(json_path, 'r'))

    imgs = {}
    for img in json_data['images']:
        imgs[img['id']] = img

    has_dp_count = 0
    no_dp_count = 0
    for annot in json_data['annotations']:
        im_id, id = annot['image_id'], annot['id']
        if 'dp_masks' not in annot.keys():
            # print('dp_masks not in annot')
            no_dp_count += 1
            continue
        # keypoints processing
        keypoints = annot['keypoints']
        keypoints = np.reshape(keypoints, (17, 3))
        keypoints[keypoints[:, 2] > 0, 2] = 1
        # if sum(keypoints[5:, 2] > 0) < 12:
        #     no_dp_count += 1
        #     continue
        has_dp_count += 1

        # check if all major body joints are annotated
        # if sum(keypoints[5:,2]>0) < 12:
        #     continue
        # create smpl joints from coco keypoints
        smpl_2dkp = kp_coco2smpl(keypoints.copy())
        # image name
        image_id = annot['image_id']
        img_name = str(imgs[image_id]['file_name'])
        img_name_full = join(subfolders[subset], img_name)
        # keypoints
        part = np.zeros([24, 3])
        part[joints_idx] = keypoints
        # scale and center
        bbox = annot['bbox']
        center = [bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2]
        scale = scaleFactor*max(bbox[2], bbox[3])/200

        dp_annot = {'bbox': annot['bbox'],
                    'dp_x': annot['dp_x'],
                    'dp_y': annot['dp_y'],
                    'dp_I': annot['dp_I'],
                    'dp_U': annot['dp_U'],
                    'dp_V': annot['dp_V'],
                    'dp_masks': annot['dp_masks']
                    }

        # store data
        imgnames_.append(img_name_full)
        centers_.append(center)
        scales_.append(scale)
        parts_.append(part)
        smpl_2dkps_.append(smpl_2dkp)
        dp_annot_.append(dp_annot)
        im_id_.append(im_id)
        id_.append(id)

    print('# samples with dp: {}; # samples without dp: {}'.format(has_dp_count, no_dp_count))

    # store the data struct
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    out_file = os.path.join(out_path, 'dp_coco_2014_{}.npz'.format(subset))
    np.savez(out_file, imgname=imgnames_,
                       center=centers_,
                       scale=scales_,
                       part=parts_,
                       smpl_2dkps=smpl_2dkps_,
                       dp_annot=dp_annot_,
                       im_id=im_id_,
                       id=id_)


def kp_coco2smpl(kps_coco):
    smpl2coco = [[1,  2,  4,  5,  7,  8,  16, 17, 18, 19, 20, 21],
                 [11, 12, 13, 14, 15, 16, 5,  6,  7,  8,  9,  10]]

    kps_smpl = np.zeros((24, 4))

    kps_smpl[smpl2coco[0], :2] = kps_coco[smpl2coco[1], :2]
    kps_smpl[smpl2coco[0], 3] = kps_coco[smpl2coco[1], 2] / 2.

    if all(kps_coco[[11, 12], 2] > 0):
        kps_smpl[0, :2] = np.mean(kps_coco[[11, 12], :2], axis=0)
        kps_smpl[0, 3] = 0.5

    if all(kps_coco[[5, 6], 2] > 0):
        kps_smpl[12, :2] = np.mean(kps_coco[[5, 6], :2], axis=0)
        kps_smpl[12, 3] = 0.5

    if kps_smpl[12, 3] > 0 and kps_coco[0, 2] > 0:
        kps_smpl[15, :2] = (kps_smpl[12, :2] + kps_coco[0, :2]) / 2.
        kps_smpl[15, 3] = 0.5

    if kps_smpl[0, 3] > 0 and kps_smpl[12, 3] > 0:
        kps_smpl[6, :2] = np.mean(kps_smpl[[0, 12], :2], axis=0)
        kps_smpl[9, :2] = kps_smpl[6, :2]
        kps_smpl[6, 3] = 0.5
        kps_smpl[9, 3] = 0.5

    if kps_smpl[0, 3] > 0 and kps_smpl[6, 3] > 0:
        kps_smpl[3, :2] = np.mean(kps_smpl[[0, 6], :2], axis=0)
        kps_smpl[3, 3] = 0.5

    if kps_smpl[9, 3] > 0 and kps_smpl[16, 3] > 0:
        kps_smpl[13, :2] = np.mean(kps_smpl[[9, 16], :2], axis=0)
        kps_smpl[13, 3] = 0.5

    if kps_smpl[9, 3] > 0 and kps_smpl[17, 3] > 0:
        kps_smpl[14, :2] = np.mean(kps_smpl[[9, 17], :2], axis=0)
        kps_smpl[14, 3] = 0.5

    hand_foot = [[7, 8, 20, 21], [10, 11, 22, 23]]
    for i in range(4):
        if kps_smpl[hand_foot[0][i], 3] > 0:
            kps_smpl[hand_foot[1][i], :2] = kps_smpl[hand_foot[0][i], :2]
            kps_smpl[hand_foot[1][i], 3] = 0.5

    kps_smpl[:, 2] = kps_smpl[:, 3]

    return kps_smpl[:, :3].copy()

if __name__ == '__main__':
    import path_config as cfg
    db_coco_extract(cfg.COCO_ROOT, 'train', 'notebooks/output/extras')
    # db_coco_extract(cfg.COCO_ROOT, 'minival', 'notebooks/output/extras')
