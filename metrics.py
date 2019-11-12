import glob

from data_util import *


def metrics_numpy(model, directory=None, anchors=anchors_list):

    if directory == None:
        directory = directory_val

    anchors = np.array(anchors, dtype=np.float32)
    anchors = np.reshape(anchors, newshape=(1, 1, 1, anchors.shape[0], 3))

    avg_iou_whole = 0
    recall50_whole = 0
    recall75_whole = 0
    avg_obj_whole = 0
    avg_noobj_whole = 0
    avg_cat_whole = 0
    true_counter = 0

    print('searching in directory{}'.format(directory))

    for file in glob.glob(os.path.join(directory, '*.mat')):

        print('loading {0}'.format(file))
        matr = scipy.io.loadmat(file)

        for num_frame in range(len(matr['Matr'])):

            # frame = matr['Matr'][num_frame]
            # frame = frame[::2, ::2, ::2]
            # frame = np.reshape(frame, (1, 512, 128, 40, 1))
            # targets = abs(matr['Labels'][num_frame])
            # targets = targets / 2
            # targets = np.reshape(targets, newshape=(1, 20, 5))
            # y_true = preprocess_true_boxes(targets)

            frame, targets, y_true = process_matfile(matr, num_frame)

            y_pred = model.predict(frame)

            cell_x = np.reshape(np.tile(np.arange(max_grid_d), max_grid_h * max_grid_w),
                                (max_grid_h, max_grid_d, max_grid_w, 1, 1))

            cell_y = np.transpose(cell_x, (1, 2, 0, 3, 4))
            cell_z = np.transpose(cell_x, (2, 0, 1, 3, 4))
            cell_grid = np.tile(np.concatenate((cell_z, cell_y, cell_x), -1), (1, 1, 1, 6, 1))

            input_image = np.zeros((512, 128, 40))
            y_true = y_true[0]
            y_pred = y_pred[0]

            object_mask = np.expand_dims(y_true[..., 4], 4)

            grid_h = y_true.shape[0]
            grid_w = y_true.shape[1]
            grid_d = y_true.shape[2]
            grid_factor = np.reshape([grid_h, grid_w, grid_d], (1, 1, 1, 1, 3))

            net_h = input_image.shape[0]
            net_w = input_image.shape[1]
            net_d = input_image.shape[2]
            net_factor = np.reshape([net_h, net_w, net_d], (1, 1, 1, 1, 3))


            pred_box_xy = sigmoid(y_pred[..., :2])
            pred_box_wh = y_pred[..., 2:4]
            pred_box_conf = np.expand_dims(sigmoid(y_pred[..., 4]), 4)
            pred_box_class = np.exp(y_pred[..., 5:]) / np.expand_dims(np.sum(np.exp(y_pred[..., 5:]), axis=-1), -1)

            true_box_xy = y_true[..., 0:2]
            true_box_wh = y_true[..., 2:4]
            true_box_conf = y_true[..., 4]
            true_box_class = np.argmax(y_true[..., 5:], axis=-1)

            conf_delta = pred_box_conf - 0

            true_xy = (true_box_xy + cell_grid[:grid_h, :grid_w, :grid_d, :, :2]) * (
                        net_factor[..., :2] / grid_factor[..., :2])
            true_wh = np.exp(true_box_wh) * anchors[..., :2]

            true_wh_half = true_wh / 2.
            true_mins = true_xy - true_wh_half
            true_maxes = true_xy + true_wh_half

            pred_xy = (pred_box_xy + cell_grid[:grid_h, :grid_w, :grid_d, :, :2]) * (
                        net_factor[..., :2] / grid_factor[..., :2])
            pred_wh = np.exp(pred_box_wh) * anchors[..., :2]

            pred_wh_half = pred_wh / 2.
            pred_mins = pred_xy - pred_wh_half
            pred_maxes = pred_xy + pred_wh_half

            intersect_mins = np.maximum(pred_mins, true_mins)
            intersect_maxes = np.minimum(pred_maxes, true_maxes)
            intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
            intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

            true_areas = true_wh[..., 0] * true_wh[..., 1]
            pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

            union_areas = pred_areas + true_areas - intersect_areas
            iou_scores = intersect_areas / union_areas
            iou_scores = object_mask * np.expand_dims(iou_scores, 4)

            conf_delta *= np.array(iou_scores < ignore_thresh, dtype=np.float)

            count = np.sum(object_mask)
            count_noobj = np.sum(1 - object_mask)
            detect_mask = ((pred_box_conf * object_mask) >= 0.5).astype(np.float)
            class_mask = np.expand_dims((np.argmax(pred_box_class, -1) == true_box_class).astype(np.float), 4)
            recall50 = np.sum((iou_scores >= 0.5) * detect_mask * class_mask) / (count + 1e-3)
            recall75 = np.sum((iou_scores >= 0.75).astype(np.float) * detect_mask * class_mask) / (count + 1e-3)
            avg_iou = np.sum(iou_scores) / (count + 1e-3)
            avg_obj = np.sum(pred_box_conf * object_mask) / (count + 1e-3)
            avg_noobj = np.sum(pred_box_conf * (1 - object_mask)) / (count_noobj + 1e-3)
            avg_cat = np.sum(object_mask * class_mask) / (count + 1e-3)
            #         print(count, count_noobj, recall50, recall75)

            avg_iou_whole += avg_iou
            recall50_whole += recall50
            recall75_whole += recall75
            avg_obj_whole += avg_obj
            avg_noobj_whole += avg_noobj
            avg_cat_whole += avg_cat
            true_counter += 1

    print('avg_iou {0}'.format(avg_iou_whole / true_counter))
    print('recall50 {0}'.format(recall50_whole / true_counter))
    print('recall75 {0}'.format(recall75_whole / true_counter))
    print('avg_obj {0}'.format(avg_obj_whole / true_counter))
    print('avg_noobj {0}'.format(avg_noobj_whole / true_counter))
    print('avg_cat {0}'.format(avg_cat_whole / true_counter))


def metrics_numpy_targets(model, directory=directory_val, anchors=anchors_list, verbose = 0):

    iou_thresh = 0.6
    obj_thresh = 0.5

    anchors = np.array(anchors, dtype=np.float32)
    anchors = np.reshape(anchors, newshape=(1, 1, 1, anchors.shape[0], 3))

    avg_iou_whole = 0
    recall50_whole = 0
    precision_whole = 0
    true_counter = 0

    print('searching in directory{}'.format(directory))

    for file in glob.glob(os.path.join(directory, '*.mat')):

        print('loading {0}'.format(file))
        matr = scipy.io.loadmat(file)

        for num_frame in range(0, len(matr['Matr'])):

            # frame = matr['Matr'][num_frame]
            # frame = frame[::2, ::2, ::2]
            # frame = np.reshape(frame, (1, 512, 128, 40, 1))
            # targets = abs(matr['Labels'][num_frame])
            # targets = targets / 2
            # targets = np.reshape(targets, newshape=(1, 20, 5))
            # y_true = preprocess_true_boxes(targets)

            frame, targets, _ = process_matfile(matr, num_frame)

            y_pred = model.predict(frame)

            predicted_boxes = decode_netout(y_pred, obj_thresh= obj_thresh)

            targets = targets[targets[..., 0] > 0]
            #             print('targets', targets)
            #             print('predicts', predicted_boxes[0])

            if (len(predicted_boxes) != 0) and (targets.shape[0] != 0):

                if verbose == 1:
                    print('before NMS:', len(predicted_boxes))
                predicted_boxes = do_nms_exp(predicted_boxes, 0.5)

                if verbose == 1:
                    print('after NMS:', len(predicted_boxes))

                matric_iou = np.zeros((len(predicted_boxes), len(targets)))

                for i in range(len(predicted_boxes)):
                    box_1 = predicted_boxes[i][0:4]
                    for j in range(len(targets)):
                        box_2 = targets[j][1:]
                        matric_iou[i, j] = bbox_iou_exp(box_1, box_2)

                iou_vector = np.max(matric_iou, axis=1)
                iou_rows = np.max(matric_iou, axis=0)
                avg_iou = np.sum(iou_vector) / len(iou_vector)

                if verbose == 1:
                    print('avg_iou:', avg_iou)

                indeces_TP = iou_vector > iou_thresh
                indeces_FN = iou_rows < iou_thresh
                TP_count = np.sum(indeces_TP)
                FP_count = np.sum(np.invert(indeces_TP))
                FN_count = np.sum(indeces_FN)

                precision = TP_count / (TP_count + FP_count)
                recall_50 = TP_count / (TP_count + FN_count)

                if verbose == 1:
                    print('precision: {0} , recall50: {1}'.format(precision, recall_50))
                    print('counts', TP_count, FP_count, iou_vector.shape)
                    print('iou_vector+ indeces:', iou_vector[iou_vector > iou_thresh], iou_vector > iou_thresh)
                    print()

            else:
                avg_iou = 0
                recall_50 = 0
                precision = 0
            #                 true_counter = 0

            avg_iou_whole += avg_iou
            recall50_whole += recall_50
            precision_whole += precision
            true_counter += len(predicted_boxes)

    print('avg_iou {0}'.format(avg_iou_whole / true_counter))
    print('avg_precision {0}'.format(precision_whole / true_counter))
    print('recall50 {0}'.format(recall50_whole / true_counter))
    print('true_counter: {0}'.format(true_counter))