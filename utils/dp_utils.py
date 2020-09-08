import numpy as np
import cv2

from utils.imutils import transform

import utils.blob as blob_utils
import utils.segms as segm_utils
import utils.densepose_methods as dp_methods

DP = dp_methods.DensePoseMethods()

def dp_annot_process(ann, heatmap_size, crop_res, center, scale, IsFlipped):
    bb_xywh = np.array(ann['bbox'])

    bbox_gt = [bb_xywh[0], bb_xywh[1], bb_xywh[0] + bb_xywh[2], bb_xywh[1] + bb_xywh[3]]
    # Cropped Upper left point
    crop_ul = np.array(transform([1, 1], center, scale, [crop_res] * 2, invert=1)) - 1
    # Cropped Bottom right point
    crop_br = np.array(
        transform([crop_res + 1] * 2, center, scale, [crop_res] * 2, invert=1)) - 1
    bbox_crop = np.concatenate([crop_ul, crop_br])

    dp_dict = {}
    M = heatmap_size

    # Create blobs for densepose supervision.
    ################################################## The mask
    All_labels = blob_utils.zeros(M ** 2, int32=True)
    All_Weights = blob_utils.zeros(M ** 2, int32=True)
    ################################################# The points
    X_points = blob_utils.zeros(196, int32=False)
    Y_points = blob_utils.zeros(196, int32=False)
    Ind_points = blob_utils.zeros(196, int32=True)
    I_points = blob_utils.zeros(196, int32=True)
    U_points = blob_utils.zeros(196, int32=False)
    V_points = blob_utils.zeros(196, int32=False)
    Uv_point_weights = blob_utils.zeros(196, int32=False)
    #################################################

    Ilabel = segm_utils.GetDensePoseMask(ann['dp_masks'])
    #
    GT_I = np.array(ann['dp_I'])
    GT_U = np.array(ann['dp_U'])
    GT_V = np.array(ann['dp_V'])
    GT_x = np.array(ann['dp_x'])
    GT_y = np.array(ann['dp_y'])
    GT_weights = np.ones(GT_I.shape).astype(np.float32)
    #
    ## Do the flipping of the densepose annotation !
    if IsFlipped:
        GT_I, GT_U, GT_V, GT_x, GT_y, Ilabel = DP.get_symmetric_densepose(GT_I, GT_U, GT_V, GT_x, GT_y,
                                                                          Ilabel)
    #
    roi_fg = bbox_crop
    roi_gt = bbox_gt
    #
    x1 = roi_fg[0];
    x2 = roi_fg[2]
    y1 = roi_fg[1];
    y2 = roi_fg[3]
    #
    x1_source = roi_gt[0];
    x2_source = roi_gt[2]
    y1_source = roi_gt[1];
    y2_source = roi_gt[3]
    #
    x_targets = (np.arange(x1, x2, (x2 - x1) / float(M)) - x1_source) * (255. / (x2_source - x1_source))
    y_targets = (np.arange(y1, y2, (y2 - y1) / float(M)) - y1_source) * (255. / (y2_source - y1_source))
    #
    x_targets = x_targets[0:M]  ## Strangely sometimes it can be M+1, so make sure size is OK!
    y_targets = y_targets[0:M]
    #
    [X_targets, Y_targets] = np.meshgrid(x_targets, y_targets)
    New_Index = cv2.remap(Ilabel, X_targets.astype(np.float32), Y_targets.astype(np.float32),
                          interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=(0))
    # #
    All_L = np.zeros(New_Index.shape)
    All_W = np.ones(New_Index.shape)
    #
    All_L = New_Index
    #
    gt_length_x = x2_source - x1_source
    gt_length_y = y2_source - y1_source
    #
    GT_y = ((GT_y / 255. * gt_length_y) + y1_source - y1) * (float(M) / (y2 - y1))
    GT_x = ((GT_x / 255. * gt_length_x) + x1_source - x1) * (float(M) / (x2 - x1))
    #
    GT_I[GT_y < 0] = 0
    GT_I[GT_y > (M - 1)] = 0
    GT_I[GT_x < 0] = 0
    GT_I[GT_x > (M - 1)] = 0
    #
    points_inside = GT_I > 0
    GT_U = GT_U[points_inside]
    GT_V = GT_V[points_inside]
    GT_x = GT_x[points_inside]
    GT_y = GT_y[points_inside]
    GT_weights = GT_weights[points_inside]
    GT_I = GT_I[points_inside]
    #
    X_points[0:len(GT_x)] = GT_x
    Y_points[0:len(GT_y)] = GT_y
    # Ind_points[i, 0:len(GT_I)] = i
    I_points[0:len(GT_I)] = GT_I
    U_points[0:len(GT_U)] = GT_U
    V_points[0:len(GT_V)] = GT_V
    Uv_point_weights[0:len(GT_weights)] = GT_weights

    All_labels[:] = np.reshape(All_L.astype(np.int32), M ** 2)
    All_Weights[:] = np.reshape(All_W.astype(np.int32), M ** 2)

    # K = cfg.BODY_UV_RCNN.NUM_PATCHES
    K = 24
    # print(K)
    #
    U_points = np.tile(U_points, [K + 1])
    V_points = np.tile(V_points, [K + 1])
    Uv_Weight_Points = np.zeros(U_points.shape)
    #
    for jjj in range(1, K + 1):
        Uv_Weight_Points[jjj * I_points.shape[0]: (jjj + 1) * I_points.shape[0]] = (I_points == jjj).astype(
            np.float32)
        # Uv_Weight_Points[:, jjj * I_points.shape[1]: (jjj + 1) * I_points.shape[1]] = (I_points == jjj).astype(
        #     np.float32)

    ##
    dp_dict['body_uv_ann_labels'] = np.array(All_labels).astype(np.int32)
    dp_dict['body_uv_ann_weights'] = np.array(All_Weights).astype(np.float32)
    #
    ##########################
    dp_dict['body_uv_X_points'] = X_points.astype(np.float32)
    dp_dict['body_uv_Y_points'] = Y_points.astype(np.float32)
    dp_dict['body_uv_Ind_points'] = Ind_points.astype(np.float32)
    dp_dict['body_uv_I_points'] = I_points.astype(np.float32)
    dp_dict['body_uv_U_points'] = U_points.astype(
        np.float32)  #### VERY IMPORTANT :   These are switched here :
    dp_dict['body_uv_V_points'] = V_points.astype(np.float32)
    dp_dict['body_uv_point_weights'] = Uv_Weight_Points.astype(np.float32)

    return dp_dict