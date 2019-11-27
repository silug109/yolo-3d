import random

import numpy as np
import scipy.io
import glob
import json
import os
from PIL import Image

from sklearn.model_selection import train_test_split

from config import *


# old version
# def process_matfile(matr, num_frame):
#     Labels = np.array(matr["Labels"][num_frame])
#     Labels[:, 1:5] = Labels[:, 1:5] / 2 # 1-ое значение - класс объекта, дальше координаты 1:5 двумерные. Их необходимо поделить пополам на config.scale_factor
#     object_mask = Labels[:, 0]
#     if object_mask.shape[0] != 0:
#         targets = np.expand_dims(Labels, 0)
#         targets = abs(targets)
#         frame = matr["Matr"][num_frame]
#         frame = frame[::striding, ::striding, ::striding]
#         frame = np.reshape(frame, (1,frame.shape[0], frame.shape[1], frame.shape[2], 1))
#         y_true = preprocess_true_boxes(targets)
#     return (frame, targets[0], y_true)
#
# def load_from_matfile(matr, num_frame):
#     frame = matr["Matr"][num_frame]
#     frame = frame[::2, ::2, ::2]
#     return frame
#
# def return_val(pathname_val):
#     for file in pathname_val:
#         matr = scipy.io.loadmat(os.path.join(directory_val ,file))
#         frame_data = np.zeros((40 ,IMAGE_H//striding ,IMAGE_W//striding ,IMAGE_D//striding ,1))
#         targets_data = np.zeros((40 ,20 ,5))
#         y_true_data = np.zeros((40 ,GRID_H ,GRID_W ,GRID_D ,num_boxes ,1+ N_DIM*2+num_classes))
#
#         for num_frame in range(len(matr['Labels'])):
#             frame, targets, y_true = process_matfile(matr,num_frame)
#             frame_data[num_frame ,...] = frame
#             targets_data[num_frame ,...] = targets[0]
#             y_true_data[num_frame ,...] = y_true
#     return (frame_data, y_true_data)
#
# def get_full_frame_dir(pathname = pathname):
#     dir_list = os.listdir(pathname)
#     full_frame_dir = []
#     for dir_item in dir_list:
#         for num in range(40):
#             full_frame_dir.append([os.path.join(pathname, dir_item), num])
#
#     train_list, test_list = train_test_split(full_frame_dir, train_size=0.96)
#
#     return full_frame_dir, train_list, test_list
#
# def generator_train(train_list = None):
#     '''
#
#     :param train_list:
#     :return:
#     '''
#     if train_list == None:
#         _, train_list, _ = get_full_frame_dir()
#
#     while True:
#         random.shuffle(train_list)
#         for item in train_list:
#             file = item[0]
#             num_frame = item[1]
#             matr = scipy.io.loadmat(file)
#             frame, _, y_true = process_matfile(matr, num_frame)
#             yield (frame, y_true)
#
# def return_test(test_list):
#     '''
#     Возвращает кортеж из 2 ndarray массивов исходя из списка директорий test_list и номеров кадров из директорий
#     :param test_list: список кортежей из директории и номера кадра
#     :return: frame_data.shape = (len(test_list), input_shape, 1)
#     '''
#     counter = 0
#     for item in test_list:
#
#         file = item[0]
#         num_frame = item[1]
#         matr = scipy.io.loadmat(file)
#
#         frame_data = np.zeros((len(test_list), IMAGE_H/striding, IMAGE_W/striding, IMAGE_D/striding, 1))
#         targets_data = np.zeros((len(test_list), 20, 5))
#         y_true_data = np.zeros((len(test_list), GRID_H, GRID_W, GRID_D, num_boxes, 1+ num_classes+2*N_DIM))
#
#         frame, targets, y_true = process_matfile(matr, num_frame)
#         frame_data[counter, ...] = frame
#         targets_data[counter, ...] = targets[0]
#         y_true_data[counter, ...] = y_true
#         counter += 1
#     return (frame_data, y_true_data)
#
# def preprocess_true_boxes(true_boxes, anchors=anchors_list):
#     '''
#     Return processed tensor in yolo-format from target
#     :param true_boxes: матрица, содержащая bounding box'ы в формате numpy.ndarray(1,20,5/7),
#                                                     где первое измерение - число фреймов,
#                                                     второе измерение - число максимально возможныъ боксов при разметке на кадре,
#                                                     третье измерение - число параметров бокса 5 - для двумерного случая, 7 - для трехмерного
#                                                     conf,coord1,coord2,[coord3], height,width, [depth]
#     :param anchors: якорные bounding boxes. Разный формат для двумерных и трехмерных разметок.
#     :param num_classes: количество классов int
#     :return: тензор размера (1,input_shape, len(anchors), num_classes+1+4/6
#     '''
#     true_boxes = np.array(true_boxes)
#     grid_shape = (GRID_H, GRID_W, GRID_D)
#     original_input_shape = (IMAGE_H//striding, IMAGE_W//striding, IMAGE_D//striding)
#
#     anchors = np.array(anchors)
#
#     grid_shape = np.array(grid_shape, dtype='int32')
#     boxes_xy = np.array(true_boxes[..., 1:3])
#     boxes_wh = np.array(true_boxes[..., 3:5])
#
#     div_true_boxes_xy = (boxes_xy % (original_input_shape[0:N_DIM] / grid_shape[0:N_DIM])) / (
#                 original_input_shape[0:N_DIM]/ grid_shape[0:N_DIM])
#     div_true_boxes_wh = boxes_wh
#     div_true_boxes = np.concatenate((div_true_boxes_xy, div_true_boxes_wh), axis=2)
#
#     true_boxes[..., 1:3] = boxes_xy // (original_input_shape[0:2] / grid_shape[0:2][::])
#     true_boxes[..., 3:5] = boxes_wh // (
#                 original_input_shape[0:2] / grid_shape[0:2][::])  # третья координата не добавлена еще
#
#     m = true_boxes.shape[0]
#     y_true = np.zeros((m, grid_shape[0], grid_shape[1], grid_shape[2], num_boxes, N_DIM*2 + 1 + num_classes),
#                       dtype='float32')
#     anchors = np.expand_dims(anchors, 0)
#     anchors = anchors[..., 0:N_DIM]  # здесь изменить потом
#     anchor_maxes = anchors / 2.
#     anchor_mins = -anchor_maxes
#     valid_mask = boxes_wh[..., 0] > 0
#
#     for b in range(m):
#
#         wh = boxes_wh[b, valid_mask[b]]
#         if len(wh) == 0: continue
#         wh = np.expand_dims(wh, -2)
#
#         box_maxes = wh / 2.
#         box_mins = -box_maxes
#
#         intersect_mins = np.maximum(box_mins, anchor_mins)
#         intersect_maxes = np.minimum(box_maxes, anchor_maxes)
#
#         # N_DIM == 3
#         intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
#         intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
#         box_area = wh[..., 0] * wh[..., 1]
#         anchor_area = anchors[..., 0] * anchors[..., 1]
#
#         iou = intersect_area / (box_area + anchor_area - intersect_area + 1e-4)
#         best_anchor = np.argmax(iou, axis=-1)
#
#         for t, n in enumerate(best_anchor):
#             i = true_boxes[b, t, 2].astype('int32')
#             j = true_boxes[b, t, 1].astype('int32')
#             l = int(20 // (original_input_shape[2] / grid_shape[2]))
#             #             k = n
#             #             c = 1 #stable version
#
#             if true_boxes[b, t, 0] in class_dict.keys():
#                 c = class_dict[true_boxes[b, t, 0]]
#             else:
#                 c = 1
#             #             print(true_boxes[b,t,0], c)
#             for k in range(len(anchors[0])):
#                 y_true[b, j, i, l, k, 0:N_DIM] = div_true_boxes_xy[b, t, :]
#                 y_true[b, j, i, l, k, N_DIM:N_DIM*2] = np.log(div_true_boxes_wh[b, t, :] / anchors[0, k] + 1e-3)
#                 #                 y_true[b, j, i, l, k, 4] = 1*iou[t,k]
#                 y_true[b, j, i, l, k, N_DIM*2] = 1
#                 y_true[b, j, i, l, k, N_DIM*2+1+ c] = 1
#     return y_true


#new

def create_full_dir_list(data_directory):
    full_dir_list = {}
    full_dir_list['pathname_base'] = []
    for dir in os.listdir(data_directory):
        files = glob.glob(os.path.join(data_directory,dir,'*.png'))
        # print(files[0])
        num_file = [os.path.basename(i).split('.')[0] for i in files]
        num_file.sort(key = lambda item: int(item))
        pathnames_base = [os.path.join(data_directory,dir,num) for num in num_file]

        pathname_images = [pathname +'.png' for pathname in pathnames_base]
        pathname_frames = [pathname +'.npy' for pathname in pathnames_base]
        pathname_labels = [pathname + '.json' for pathname in pathnames_base]
        # print(pathnames_base)
        full_dir_list['pathname_base'] += pathnames_base
    return full_dir_list

def process_pathname_base(pathname_base):
    pathname_image = pathname_base + '.png'
    pathname_frame = pathname_base + '.npy'
    pathname_label = pathname_base + '.json'
    # print(pathname_image, pathname_frame, pathname_label)

    image = Image.open(pathname_image)
    frame = np.load(pathname_frame)

    target = json.load(open(pathname_label, 'r'))

    # image.show()
    # print(frame.shape)
    # pprint.pprint(target)

    return image, frame, target

def process_json(target):
    labels = []
    if 'bounding_boxes' in target.keys():
        for bbox in target['bounding_boxes']:
            label = []
            label.append(bbox['class_ind'])
            label.append(abs(bbox['x']))
            label.append(abs(bbox['y']))
            label.append(abs(bbox['z']))
            label.append(abs(bbox['length']))
            label.append(abs(bbox['width']))
            label.append(abs(bbox['height']))
            # label[1:] = label[1:] # так как мы разреживаем пространство
            labels.append(label)
    return labels


def process_oneframe(pathname):
    image,frame, target = process_pathname_base(pathname)
    labels = process_json(target)
    if len(labels) > 0:
        labels = np.array(labels)
        labels[:,1:] = labels[:,1:]/2
        y_true = preprocess_true_boxes_new(labels)
    else:
        y_true = np.zeros((1,GRID_H,GRID_W,GRID_D, num_boxes, N_DIM*2 + 1 + num_classes), dtype='float32')
    # print(labels)

    frame = frame[::striding, ::striding, ::striding]
    frame = np.reshape(frame, (1, frame.shape[0], frame.shape[1], frame.shape[2], 1))

    return  frame, labels, y_true

def preprocess_true_boxes_new(true_boxes, anchors=anchors_list):
    '''
    Return processed tensor in yolo-format from target
    :param true_boxes: матрица, содержащая bounding box'ы в формате numpy.ndarray(1,20,5/7),
                                                    где первое измерение - число фреймов,
                                                    второе измерение - число максимально возможныъ боксов при разметке на кадре,
                                                    третье измерение - число параметров бокса 5 - для двумерного случая, 7 - для трехмерного
                                                    conf,coord1,coord2,[coord3], height,width, [depth]
    :param anchors: якорные bounding boxes. Разный формат для двумерных и трехмерных разметок.
    :param num_classes: количество классов int
    :return: тензор размера (1,input_shape, len(anchors), num_classes+1+4/6
    '''
    true_boxes = np.array(true_boxes)
    grid_shape = (GRID_H, GRID_W, GRID_D)
    original_input_shape = (IMAGE_H//striding, IMAGE_W//striding, IMAGE_D//striding)

    anchors = np.array(anchors)

    grid_shape = np.array(grid_shape, dtype='int32')
    original_input_shape = np.array(original_input_shape, dtype = 'int32')
    boxes_xyz = np.array(true_boxes[..., 1:1+N_DIM])
    boxes_lwh = np.array(true_boxes[..., 1+N_DIM:1+2*N_DIM])

    div_true_boxes_xyz = (boxes_xyz % (original_input_shape[0:N_DIM] / grid_shape[0:N_DIM])) / (original_input_shape[0:N_DIM]/ grid_shape[0:N_DIM])
    div_true_boxes_lwh = boxes_lwh
    div_true_boxes = np.concatenate((div_true_boxes_xyz, div_true_boxes_lwh), axis=1)

    true_boxes[..., 1:1+N_DIM] = boxes_xyz // (original_input_shape[0:N_DIM] / grid_shape[0:N_DIM][::])
    true_boxes[..., 1+N_DIM:1+2*N_DIM] = boxes_lwh // (original_input_shape[0:N_DIM] / grid_shape[0:N_DIM][::])  # третья координата не добавлена еще

    # m = true_boxes.shape[0]
    y_true = np.zeros((grid_shape[0], grid_shape[1], grid_shape[2], num_boxes, N_DIM*2 + 1 + num_classes), dtype='float32')
    anchors = np.expand_dims(anchors, 0)
    anchors = anchors[..., 0:N_DIM]  # здесь изменить потом
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes

    boxes_lwh = np.expand_dims(boxes_lwh, -2)
    box_maxes = boxes_lwh / 2.
    box_mins = -box_maxes

    intersect_mins = np.maximum(box_mins, anchor_mins)
    intersect_maxes = np.minimum(box_maxes, anchor_maxes)

    # N_DIM == 3
    intersect_lwh = np.maximum(intersect_maxes - intersect_mins, 0.)
    if N_DIM == 2:
        intersect_area = intersect_lwh[..., 0] * intersect_lwh[..., 1]
        box_area = boxes_lwh[..., 0] * boxes_lwh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
    if N_DIM == 3:
        intersect_area = intersect_lwh[..., 0] * intersect_lwh[..., 1]*intersect_lwh[...,2]
        box_area = boxes_lwh[..., 0] * boxes_lwh[..., 1]* boxes_lwh[...,2]
        anchor_area = anchors[..., 0] * anchors[..., 1]* anchors[...,2]

    iou = intersect_area / (box_area + anchor_area - intersect_area + 1e-4)
    best_anchor = np.argmax(iou, axis=-1)

    for t, n in enumerate(best_anchor):
        i = true_boxes[t, 1].astype('int32')
        j = true_boxes[t, 2].astype('int32')
        l = true_boxes[t, 3].astype('int32')

        if true_boxes[t, 0] in class_dict.keys():
            c = class_dict[true_boxes[t, 0]]
        else:
            c = 1
        for k in range(len(anchors[0])):
            y_true[i, j, l, k, 0:N_DIM] = div_true_boxes_xyz[t, :]
            y_true[i, j, l, k, N_DIM:N_DIM*2] = np.log(div_true_boxes_lwh[t, :] / anchors[0, k] + 1e-3)
            #                 y_true[b, j, i, l, k, 4] = 1*iou[t,k]
            y_true[i, j, l, k, N_DIM*2] = 1
            y_true[i, j, l, k, N_DIM*2+1+ c] = 1

    y_true = np.expand_dims(y_true, 0)
    return y_true

def generator_train_new(train_list):
    while True:
        random.shuffle(train_list)
        for pathname in train_list:
            frame, labels, y_true = process_oneframe(pathname)
            yield (frame, y_true)

def return_test_new(test_list):
    frame_data = np.zeros((len(test_list), IMAGE_H // striding, IMAGE_W // striding, IMAGE_D // striding, 1))
    y_true_data = np.zeros((len(test_list), GRID_H, GRID_W, GRID_D, num_boxes, 1 + num_classes + 2 * N_DIM))
    counter = 0
    for pathname in test_list:
        frame,_, y_true = process_oneframe(pathname)
        frame_data[counter, ...] = frame
        y_true_data[counter, ...] = y_true
        counter += 1
    return (frame_data, y_true_data)

# doesn't depend on file system

def decode_netout(netout, obj_thresh=obj_thresh, anchors=anchors_list):
    grid_shape = (GRID_H, GRID_W, GRID_D)
    original_input_shape = (IMAGE_H // striding, IMAGE_W // striding, IMAGE_D // striding)
    grid_h, grid_w, grid_d = netout.shape[1:4]

    netout = np.array(netout.reshape((GRID_H, GRID_W, GRID_D, num_boxes, -1)))
    nb_class = netout.shape[-1] - 5
    boxes = []

    netout[..., :N_DIM] = sigmoid(netout[..., :N_DIM])
    netout[..., N_DIM * 2] = sigmoid(netout[..., N_DIM * 2])

    netout[..., N_DIM * 2 + 1:] = netout[..., N_DIM * 2][..., np.newaxis] * softmax(netout[..., N_DIM * 2 + 1:])
    netout[..., N_DIM * 2 + 1:] *= netout[..., N_DIM * 2 + 1:] > obj_thresh

    for row in range(GRID_H):
        for col in range(GRID_W):
            for dep in range(GRID_D):
                for b in range(num_boxes):
                    objectness = netout[row, col, dep, b, N_DIM * 2]
                    if (objectness <= obj_thresh): continue
                    if N_DIM == 2:
                        x, y, h, w = netout[row, col, dep, b, :N_DIM * 2]
                    elif N_DIM == 3:
                        x, y, z, h, w, d = netout[row, col, dep, b, :N_DIM * 2]
                    # x,y,z,w,h,d = netout[row,col,dep,b, :N_DIM*2] N_DIM == 3
                    x = (row + x) * (original_input_shape[0] / grid_shape[0])
                    y = (col + y) * (original_input_shape[1] / grid_shape[1])
                    z = (dep + z) * (original_input_shape[2] / grid_shape[2])
                    h = anchors[b][0] * np.exp(h)
                    w = anchors[b][1] * np.exp(w)
                    if N_DIM == 3:
                        d = anchors[b][2] * np.exp(d)
                    classes = netout[row, col, dep, b, N_DIM * 2 + 1:]
                    box_class = np.argmax(softmax(classes))
                    if N_DIM == 2:
                        box = [x, y, h, w, objectness, box_class]
                    elif N_DIM == 3:
                        box = [x, y, z, h, w, d, objectness, box_class]
                    boxes.append(box)
    return boxes


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x, axis=-1):
    x = x - np.amax(x, axis, keepdims=True)
    e_x = np.exp(x)
    return e_x / e_x.sum(axis, keepdims=True)


def do_nms(boxes, nms_thresh):
    if len(boxes) > 0:
        nb_class = len(boxes[0].classes)
    else:
        return
    for c in range(nb_class):
        sorted_indices = np.argsort([-box.classes[c] for box in boxes])
        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]
            if boxes[index_i].classes[c] == 0: continue
            for j in range(i + 1, len(sorted_indices)):
                index_j = sorted_indices[j]
                if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                    boxes[index_j].classes[c] = 0


def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b
    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2, x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2, x4) - x3


def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
    intersect = intersect_w * intersect_h
    w1, h1 = box1.xmax - box1.xmin, box1.ymax - box1.ymin
    w2, h2 = box2.xmax - box2.xmin, box2.ymax - box2.ymin
    union = w1 * h1 + w2 * h2 - intersect
    return float(intersect) / union





def do_nms_exp(boxes, nms_thresh=0.5):
    boxes = list(boxes)
    if len(boxes) == 0:
        return
    sorted_indices = np.argsort([-box[2*N_DIM] for box in boxes])
    for i in range(len(sorted_indices)):
        index_i = sorted_indices[i]
        for j in range(i + 1, len(sorted_indices)):
            index_j = sorted_indices[j]
            if boxes[index_j][0] == 0: continue

            if bbox_iou_exp(boxes[index_i], boxes[index_j]) >= nms_thresh:
                boxes[index_j] = [0, 0, 0, 0, 0, 0, 0]
    return [box for box in boxes if box[0] != 0]


def bbox_iou_exp(box1, box2):
    intersect_h = _interval_overlap([box1[0] - box1[N_DIM] / 2, box1[0] + box1[N_DIM] / 2],
                                    [box2[0] - box2[N_DIM] / 2, box2[0] + box2[N_DIM] / 2])
    intersect_w = _interval_overlap([box1[1] - box1[1+N_DIM] / 2, box1[1] + box1[1+N_DIM] / 2],
                                    [box2[1] - box2[1+N_DIM] / 2, box2[1] + box2[1+N_DIM] / 2])
    intersect_d = _interval_overlap([box1[2] - box1[2 + N_DIM] / 2, box1[2] + box1[2 + N_DIM] / 2],
                                    [box2[2] - box2[2 + N_DIM] / 2, box2[2] + box2[2 + N_DIM] / 2])
    intersect = intersect_h * intersect_w * intersect_d
    h1, w1, d1  = box1[N_DIM], box1[1+N_DIM] , box1[2+N_DIM]
    h2, w2, d2 = box2[N_DIM], box2[1+ N_DIM], box2[2+N_DIM]
    union = w1 * h1 * d1 + w2 * h2 * d2 - intersect
    return float(intersect) / union



# считает сколько различных классов есть и количество объектов по каждому классу
# pathname = os.path.abspath('data')
# class_dict = {}
# for file in glob.glob(pathname+'/*.mat'):
#     matr = scipy.io.loadmat(file)
#     for num_frame in range(len(matr['Labels'])):
#         Labels = np.array(matr["Labels"][num_frame])
# #         Labels = Labels/2
#         Labels[:,1:5] = Labels[:,1:5]/2

#         object_mask = Labels[Labels[:,0]>0,:]
#         if object_mask.shape[0] != 0:
# #             print(object_mask[])
#             for object_label in object_mask[:,0]:
# #                 print(object_label)
#                 if object_label in class_dict.keys():
#                     class_dict[object_label] += 1
#                 else:
#                     class_dict.setdefault(object_label, 1)

# print(class_dict)


# pathname = os.path.abspath('data')
# counter_frames = 0
# counter_boxes = 0
# for file in glob.glob(pathname+'/*.mat'):
#     matr = scipy.io.loadmat(file)
#     for num_frame in range(len(matr['Labels'])):
#         Labels = np.array(matr["Labels"][num_frame])
# #         Labels = Labels/2
#         Labels[:,1:5] = Labels[:,1:5]/2

#         object_mask = Labels[Labels[:,0]>0,:]
#         if object_mask.shape[0] != 0:
#             counter_frames += 1
#             counter_boxes += object_mask.shape[0]

# print(counter_frames,counter_boxes)







# # код для чекания битых кадров

# pathname = os.path.abspath('data')
# for file in glob.glob(pathname+'/*.mat'):
#     matr = scipy.io.loadmat(file)
#     for num_frame in range(len(matr['Labels'])):
#         Labels = np.array(matr["Labels"][num_frame])
#         Labels = Labels/2
#         object_mask = Labels[:,0]
#         if object_mask.shape[0] != 0:
#             true_boxes = Labels[:,:]
#             targets = np.expand_dims(Labels,0)
#             frame = matr["Matr"][num_frame]
#             frame = frame[::2,::2,::2]
#             frame = np.reshape(frame,(1,frame.shape[0],frame.shape[1],frame.shape[2],1))
#             y_true = preprocess_true_boxes(targets)
#             print(file, num_frame,model.evaluate([frame,targets], y_true ))
