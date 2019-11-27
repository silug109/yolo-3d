from config import *
from data_util import *

def metrics_numpy_new(model, directory, anchors=anchors_list):
    '''
    функция вычисляет показатели различных метрик при работе модели и выводит их.

    :param model: скомпилированная модель
    :param directory: папка, из которой будет создаваться набор директорий
                                    для считывания файлов и сбора статистики
    :param anchors: якорные боксы
    :return: None, хотя можно выводить возвращать метрики для создания графиков.
    '''

    full_dir_list = create_full_dir_list(directory) # генерируем из папки все директории содержащие файл(кадр с камеры, спектр, размеченные кадры)

    anchors = np.array(anchors, dtype=np.float32)
    anchors = np.reshape(anchors, newshape=(1, 1, 1, anchors.shape[0], 3))

    #набор метрик для датасета
    avg_iou_whole = 0
    recall50_whole = 0
    recall75_whole = 0
    avg_obj_whole = 0
    avg_noobj_whole = 0
    avg_cat_whole = 0
    recall50_obj_whole = 0
    recall75_obj_whole = 0
    true_counter = 0
    conf_change_whole = 0

    print('searching in directory{}'.format(directory))

    for file in full_dir_list['pathname_base']:
        frame, targets, y_true = process_oneframe(file) #выгружаем разметку и данные по названию директории
        y_pred = model.predict(frame) # предсказываем в yolo-формате разметку

        cell_x = np.reshape(np.tile(np.arange(max_grid_d), max_grid_h * max_grid_w),
                            (max_grid_h, max_grid_d, max_grid_w, 1, 1))
        cell_y = np.transpose(cell_x, (1, 2, 0, 3, 4))
        cell_z = np.transpose(cell_x, (2, 0, 1, 3, 4))
        cell_grid = np.tile(np.concatenate((cell_z, cell_y, cell_x), -1), (1, 1, 1, 6, 1))

        input_image = np.zeros((IMAGE_H//striding, IMAGE_W//striding, IMAGE_D//striding))
        y_true = y_true[0] # избавляемся от нулевого измерения по батчам.
        y_pred = y_pred[0] # избавляемся от нулевого измерения по батчам.
        object_mask = np.expand_dims(y_true[..., 2 * N_DIM], 4)

        grid_h, grid_w, grid_d = GRID_H, GRID_W, GRID_D
        grid_factor = np.reshape([grid_h, grid_w, grid_d], (1, 1, 1, 1, 3))

        net_h = input_image.shape[0]
        net_w = input_image.shape[1]
        net_d = input_image.shape[2]
        net_factor = np.reshape([net_h, net_w, net_d], (1, 1, 1, 1, 3))
        # преобразуем из скрытого пространства результаты к нормальному виду(Декодируем)
        pred_box_xy = sigmoid(y_pred[..., :N_DIM]) # local_x,local_y = sigma(t_x,t_y) координаты центров боксов
        pred_box_wh = y_pred[..., N_DIM:2 * N_DIM] # координаты отвечающие за размеры предсказанных боксов,
        pred_box_conf = np.expand_dims(sigmoid(y_pred[..., 2 * N_DIM]), 4) #  уверенность, что здесь есть объект
        pred_box_class = np.exp(y_pred[..., 2 * N_DIM + 1:]) / np.expand_dims(
                    np.sum(np.exp(y_pred[..., 2 * N_DIM + 1:]), axis=-1), -1)

        true_box_xy = y_true[..., 0:N_DIM] #тренировочные данные уже закодированы в надлежащем виде
        true_box_wh = y_true[..., N_DIM:2 * N_DIM]
        true_box_conf = np.expand_dims(y_true[..., 2 * N_DIM], axis=-1)
        true_box_class = np.argmax(y_true[..., 1 + 2 * N_DIM:], axis=-1)

        true_xy = (true_box_xy + cell_grid[:grid_h, :grid_w, :grid_d, :, :N_DIM]) * (
                net_factor[..., :N_DIM] / grid_factor[..., :N_DIM])  #для высчитывания IoU метрики необходимо перевести данные в глобальнфый формат
        true_wh = np.exp(true_box_wh) * anchors[..., :N_DIM] # exp(h)* anchor

        true_wh_half = true_wh / 2.
        true_mins = true_xy - true_wh_half # высчитываем координаты bottom и top точек бокса, прибавляя к пространнственным координатам соответствующее измерение по размеру
        true_maxes = true_xy + true_wh_half
        #аналогично с предсказзаной структурой
        pred_xy = (pred_box_xy + cell_grid[:grid_h, :grid_w, :grid_d, :, :N_DIM]) * (
                net_factor[..., :N_DIM] / grid_factor[..., :N_DIM])
        pred_wh = np.exp(pred_box_wh) * anchors[..., :N_DIM]

        pred_wh_half = pred_wh / 2.
        pred_mins = pred_xy - pred_wh_half
        pred_maxes = pred_xy + pred_wh_half

        intersect_mins = np.maximum(pred_mins, true_mins) #высчитываем максимальные среди минимальных
        intersect_maxes = np.minimum(pred_maxes, true_maxes) # высчитываем минимальные среди максимальных координат
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        if N_DIM == 2:
            intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1] # высчитываем размеры области пересечения
            true_areas = true_wh[..., 0] * true_wh[..., 1]  # высчитываем область разметки
            pred_areas = pred_wh[..., 0] * pred_wh[..., 1]  # высчитываем область предсказанную
        else:
            intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1] * intersect_wh[..., 2] # высчитываем размеры области пересечения
            true_areas = true_wh[..., 0] * true_wh[..., 1] * true_wh[..., 2] #высчитываем область разметки
            pred_areas = pred_wh[..., 0] * pred_wh[..., 1] * pred_wh[..., 2] #высчитываем область предсказанную

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores = intersect_areas / union_areas
        iou_scores = object_mask * np.expand_dims(iou_scores, 4)

        count = np.sum(object_mask)
        count_noobj = np.sum(1 - object_mask)
        detect_mask = ((pred_box_conf * object_mask) >= 0.5).astype(np.float)
        detect_mask_raw = object_mask * pred_box_conf
        class_mask = np.expand_dims((np.argmax(pred_box_class, -1) == true_box_class).astype(np.float), 4)
        recall50 = np.sum((iou_scores >= 0.5) * detect_mask * class_mask) / (count + 1e-3)
        recall75 = np.sum((iou_scores >= 0.75).astype(np.float) * detect_mask * class_mask) / (count + 1e-3)
        recall50_obj = np.sum((iou_scores >= 0.5) * detect_mask) / (count + 1e-3)
        recall75_obj = np.sum((iou_scores >= 0.75).astype(np.float) * detect_mask) / (count + 1e-3)

        avg_iou = np.sum(iou_scores) / (count + 1e-3)
        avg_obj = np.sum(pred_box_conf * object_mask) / (count + 1e-3)
        avg_noobj = np.sum(pred_box_conf * (1 - object_mask)) / (count_noobj + 1e-3)
        avg_cat = np.sum(object_mask * class_mask) / (count + 1e-3)

        avg_iou_whole += avg_iou
        recall50_whole += recall50
        recall75_whole += recall75
        recall50_obj_whole += recall50_obj
        recall75_obj_whole += recall75_obj
        avg_obj_whole += avg_obj
        avg_noobj_whole += avg_noobj
        avg_cat_whole += avg_cat
        conf_change_whole += np.sum(detect_mask_raw)
        true_counter += 1

    print('avg_iou {0}'.format(avg_iou_whole / true_counter))
    print('recall50 {0}'.format(recall50_whole / true_counter))
    print('recall75 {0}'.format(recall75_whole / true_counter))
    print('recall50_obj {0}'.format(recall50_obj_whole / true_counter))
    print('recall75_obj {0}'.format(recall75_obj_whole / true_counter))
    print('avg_obj {0}'.format(avg_obj_whole / true_counter))
    print('avg_noobj {0}'.format(avg_noobj_whole / true_counter))
    print('avg_cat {0}'.format(avg_cat_whole / true_counter))
    print('conf_change {0}'.format(conf_change_whole))



def metrics_numpy_targets_new(model, directory_list, verbose = 0):
    '''
    Функция, оценивающая работу модели по метрикам, учитывая не yolo-репрезентацию, а готовые bounding box

    :param model: скомпилированная модель
    :param directory_list: список директорий, по который производится просчет. как пример create_Full_dir_list()['pathname_base'] из data_utils
    :param verbose: наскольок подробно выводить информацию int
    :return:
    '''

    avg_iou_whole = 0
    recall50_whole = 0
    precision_whole = 0
    true_counter = 0
    prediction_counter = 0

    print('searching in directory{}'.format(directory_list[0]))

    # full_dir_list  = create_full_dir_list(directory)
    for file in directory_list:

        frame, labels, _ = process_oneframe(file) #загружаем всю информацию из файла
        y_pred = model.predict(frame) # делаем предикт
        predicted_boxes = decode_netout(y_pred, obj_thresh= obj_thresh) #декодируем

        if (len(predicted_boxes) != 0) and (len(labels) != 0):
            if verbose == 1:
                print('before NMS:', len(predicted_boxes))
            predicted_boxes = do_nms_exp(predicted_boxes, 0.5) # производим nms фильтрацию
            if verbose == 1:
                print('after NMS:', len(predicted_boxes))

            matric_iou = np.zeros((len(predicted_boxes), len(labels)))
            for i in range(len(predicted_boxes)):
                box_1 = predicted_boxes[i][0:2*N_DIM]
                for j in range(len(labels)):
                    box_2 = labels[j][1:]
                    matric_iou[i, j] = bbox_iou_exp(box_1, box_2)

            iou_vector = np.max(matric_iou, axis=1) # predicts
            iou_rows = np.max(matric_iou, axis=0) # targets
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
        true_counter += len(labels)
        prediction_counter += len(predicted_boxes)

    print('avg_iou {0}'.format(avg_iou_whole / true_counter))
    print('avg_precision {0}'.format(precision_whole / true_counter))
    print('recall50 {0}'.format(recall50_whole / true_counter))
    print('true_counter: {0}'.format(true_counter))
    print('prediction counter:  {0}'. format(prediction_counter))
