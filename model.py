import keras.backend as K
import numpy as np
import tensorflow as tf
from keras import layers
from keras import models
from keras.optimizers import Adam

from config import *


def get_model():
    '''
    :return: возвращает арзитектуру сети для Yolo
    '''
    input_image = layers.Input(shape=(IMAGE_H/striding ,IMAGE_W/striding ,IMAGE_D/striding ,1))
    x = layers.Conv3D(filters = 16 ,kernel_size=(10 ,10 ,10) ,strides = (1 ,1 ,1) ,padding = 'same', name = 'conv_0_1') \
        (input_image)
    x = layers.BatchNormalization(name='norm_0_1')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)
    x = layers.MaxPool3D(pool_size=(2 ,2 ,1))(x)

    x = layers.Conv3D(filters = 16 ,kernel_size=(10 ,10 ,10) ,strides = (1 ,1 ,1) ,padding = 'same' ,name = 'conv_1_1')(x)
    x = layers.BatchNormalization(name='norm_1_1')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)
    x = layers.MaxPool3D(pool_size=(4 ,4 ,4))(x)

    x = layers.Conv3D(filters = 16 ,kernel_size=(10 ,10 ,10) ,strides = (1 ,1 ,1) ,padding = 'same')(x)
    x = layers.BatchNormalization(name='norm_5_1')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)
    x = layers.MaxPool3D(pool_size=(4 ,2 ,2))(x)

    x = layers.Conv3D(filters= (4+1+num_classes )*num_boxes, kernel_size=(1 ,1 ,1), strides =(1 ,1 ,1), padding = 'same', name= 'yolo')(x)
    output = layers.Reshape((GRID_H ,GRID_W ,GRID_D ,num_boxes, 4 + 1 + num_classes))(x)
    model = models.Model(input_image, output)
    return model


def get_model_2():
    '''
    :return: возвращает арзитектуру сети для Yolo
    '''
    input_image = layers.Input(shape=(IMAGE_H/striding ,IMAGE_W/striding ,IMAGE_D/striding ,1))
    x = layers.Conv3D(filters = 16 ,kernel_size=(5 ,5 ,5) ,strides = (1 ,1 ,1) ,padding = 'same', name = 'conv_0_1') \
        (input_image)
    x = layers.Conv3D(filters = 16 ,kernel_size=(5 ,5 ,5) ,strides = (1 ,1 ,1) ,padding = 'same', name = 'conv_0_2') \
        (x)
    x = layers.BatchNormalization(name='norm_0_1')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)
    x = layers.MaxPool3D(pool_size=(2 ,2 ,1))(x)

    x = layers.Conv3D(filters = 32 ,kernel_size=(5 ,5 ,5) ,strides = (1 ,1 ,1) ,padding = 'same' ,name = 'conv_1_1')(x)
    x = layers.Conv3D(filters = 32 ,kernel_size=(5 ,5 ,5) ,strides = (1 ,1 ,1) ,padding = 'same' ,name = 'conv_1_2')(x)
    x = layers.BatchNormalization(name='norm_1_1')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)
    x = layers.MaxPool3D(pool_size=(4 ,4 ,4))(x)

    x = layers.Conv3D(filters = 32 ,kernel_size=(5 ,5 ,5) ,strides = (1 ,1 ,1) ,padding = 'same', name = 'conv_2_1')(x)
    x = layers.Conv3D(filters = 32 ,kernel_size=(5 ,5 ,5) ,strides = (1 ,1 ,1) ,padding = 'same', name = 'conv_2_2')(x)
    x = layers.BatchNormalization(name='norm_5_1')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)
    x = layers.MaxPool3D(pool_size=(4 ,2 ,2))(x)

    x = layers.Conv3D(filters= 32, kernel_size=(1 ,1 ,1), strides =(1 ,1 ,1), padding = 'same', name= 'pred_yolo')(x)
    x = layers.Conv3D(filters= (2*N_DIM+1+num_classes )*num_boxes, kernel_size=(1 ,1 ,1), strides =(1 ,1 ,1), padding = 'same', name= 'yolo')(x)
    output = layers.Reshape((GRID_H ,GRID_W ,GRID_D ,num_boxes, 2*N_DIM + 1 + num_classes))(x)
    model = models.Model(input_image, output)
    return model

def load_model_2(load_weights = False):
    model = get_model_2()
    mypotim = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model_loss_bend = model_loss()

    model.compile(loss=model_loss_bend, optimizer=mypotim)

    return model


def load_model(load_weights = False):
    model = get_model()
    mypotim = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model_loss_bend = model_loss()

    model.compile(loss=model_loss_bend, optimizer=mypotim)

    if load_weights:
        model.load_weights('warehouse/03.10.19_good_model/weights_good_loader.h5')
    return model





def loss(y_true, y_pred, anchors= anchors_list):

    # max_grid_h, max_grid_w , max_grid_d = (16,16,16)

    anchors = tf.cast(anchors, dtype = tf.float32)
    anchors = tf.reshape(anchors, shape= [1,1,1,anchors.shape[0],3])  # преобразуем разметочные боксы к виду (1,1,1,len(anchors),3)

    # генерируем сетку для добавления к локальным координатам предсказания(они нормированы по остатку и на размер ячейки),
    # поэтому надо добавлять к ним сетку целых частей сетки. cell_grid(i,j,k) = i+j+k
    cell_x = tf.cast(tf.reshape(tf.tile(tf.range(max_grid_d), [max_grid_h*max_grid_w]), (max_grid_h, max_grid_w, max_grid_d, 1, 1)),dtype= tf.float32)
    cell_y = tf.transpose(cell_x, (1,2,0,3,4))
    cell_z = tf.transpose(cell_x, (2,0,1,3,4))
    cell_grid = tf.tile(tf.concat([cell_z,cell_y,cell_x], -1), [1,1,1,anchors.shape[3],1])

    input_image = tf.cast(np.zeros((IMAGE_H//striding,IMAGE_W//striding,IMAGE_D//striding)), tf.int32)

    y_true = K.reshape(y_true, tf.shape(y_true[0,...])) # squeeze this matrix

    y_pred = tf.reshape(y_pred, tf.shape(y_pred)[1:]) #squeeze another way this matrix

    object_mask     = tf.expand_dims(y_true[..., 4], 4) #unsqueeze this matrix

    grid_h      = tf.shape(y_true)[0]
    grid_w      = tf.shape(y_true)[1]
    grid_d      = tf.shape(y_true)[2]
    # grid_h = GRID_H
    # grid_w = GRID_W
    # grid_d = GRID_D

    grid_factor = tf.reshape(tf.cast([grid_h, grid_w, grid_d], tf.float32), [1,1,1,1,3])

    net_h       = tf.shape(input_image)[0]
    net_w       = tf.shape(input_image)[1]
    net_d       = tf.shape(input_image)[2]
    # net_h = IMAGE_H
    # net_w = IMAGE_W
    # net_d = IMAGE_D
    net_factor  = tf.reshape(tf.cast([net_h, net_w, net_d], tf.float32), [1,1,1,1,3])


    pred_box_xy    = tf.sigmoid(y_pred[..., :N_DIM])
    # pred_box_xy    = y_pred[..., :2]
    pred_box_wh    = y_pred[..., N_DIM:N_DIM*2]                                                       # t_wh
    pred_box_conf  = tf.expand_dims(tf.sigmoid(y_pred[..., N_DIM*2]), 4)
    # pred_box_conf  = tf.expand_dims(y_pred[..., 4], 4)    # adjust confidence
    pred_box_class = K.softmax(y_pred[..., N_DIM+1:])

    true_box_xy    = y_true[..., 0:N_DIM] # [0:1]
    true_box_wh    = y_true[..., N_DIM:N_DIM*2] # t_wh - log(wh/anchors)
    true_box_conf  = tf.expand_dims(y_true[..., N_DIM*2], 4)
    true_box_class = y_true[...,N_DIM*2+1:]
    true_box_class = tf.argmax(y_true[...,N_DIM*2+1:],axis= -1)

    conf_delta  = pred_box_conf - 0

    true_xy = (true_box_xy + cell_grid[:grid_h,:grid_w,:grid_d,:,:N_DIM]) * (net_factor[...,:N_DIM]/grid_factor[...,:N_DIM])
    true_wh = tf.exp(true_box_wh) * anchors[...,:N_DIM]


    true_wh_half = true_wh / 2.
    true_mins    = true_xy - true_wh_half
    true_maxes   = true_xy + true_wh_half

    pred_xy = (pred_box_xy + cell_grid[:grid_h,:grid_w,:grid_d,:,:N_DIM]) * (net_factor[...,:N_DIM]/grid_factor[...,:N_DIM])
    pred_wh = tf.exp(pred_box_wh) * anchors[...,:N_DIM]


    pred_wh_half = pred_wh / 2.
    pred_mins    = pred_xy - pred_wh_half
    pred_maxes   = pred_xy + pred_wh_half

    intersect_mins  = tf.maximum(pred_mins,  true_mins)
    intersect_maxes = tf.minimum(pred_maxes, true_maxes)
    intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)

    # если N_DIM == 3: intersect_wh[..., 0] * intersect_wh[..., 1]* intersect[...,3]
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    #аналогично
    true_areas = true_wh[..., 0] * true_wh[..., 1]
    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores  = tf.truediv(intersect_areas, union_areas)
    iou_scores  = object_mask * tf.expand_dims(iou_scores, 4)

    conf_delta *= tf.to_float(iou_scores < ignore_thresh)


    count       = tf.reduce_sum(object_mask)
    count_noobj = tf.reduce_sum(1 - object_mask)
    detect_mask = tf.to_float((pred_box_conf*object_mask) >= obj_thresh)
    class_mask  = tf.expand_dims(tf.to_float(tf.equal(tf.argmax(pred_box_class, -1), true_box_class)), 4)
    recall50    = tf.reduce_sum(tf.to_float(iou_scores >= 0.5 ) * detect_mask  * class_mask) / (count + 1e-3)
    recall75    = tf.reduce_sum(tf.to_float(iou_scores >= 0.75) * detect_mask  * class_mask) / (count + 1e-3)
    avg_iou     = tf.reduce_sum(iou_scores) / (count + 1e-3)
    avg_obj     = tf.reduce_sum(pred_box_conf  * object_mask)  / (count + 1e-3)
    avg_noobj   = tf.reduce_sum(pred_box_conf  * (1-object_mask))  / (count_noobj + 1e-3)
    avg_cat     = tf.reduce_sum(object_mask * class_mask) / (count + 1e-3)

    true_box_xy, true_box_wh, xywh_mask = [true_box_xy, true_box_wh, object_mask]

    wh_scale = tf.exp(true_box_wh) * anchors[...,:N_DIM]/net_factor[...,:N_DIM] # ????????
    # wh_scale = tf.abs(true_box_wh)
    wh_scale = tf.expand_dims(2 - wh_scale[..., 0] * wh_scale[..., 1], axis=4) # N_DIM == 3 crash

    xy_delta    = xywh_mask   * (pred_box_xy-true_box_xy) *wh_scale * xywh_scale
    wh_delta    = xywh_mask   * (pred_box_wh-true_box_wh) * wh_scale * xywh_scale
    # wh_delta    = xywh_mask   * (pred_wh - true_wh) * xywh_scale
    conf_delta  = xywh_mask * (pred_box_conf-true_box_conf) * obj_scale + (1-xywh_mask) * conf_delta * noobj_scale
    class_delta = object_mask * tf.expand_dims(tf.keras.backend.sparse_categorical_crossentropy(true_box_class, pred_box_class),4)* class_scale


    # loss_xy    = tf.reduce_sum(tf.square(xy_delta),       list(range(0,5)))
    # loss_wh    = tf.reduce_sum(tf.square(wh_delta),       list(range(0,5)))
    # loss_conf  = tf.reduce_sum(tf.square(conf_delta),     list(range(0,5)))
    # loss_class = tf.reduce_sum(class_delta,               list(range(0,5)))
    loss_xy = tf.reduce_sum(tf.square(xy_delta))
    loss_wh = tf.reduce_sum(tf.square(wh_delta))
    loss_conf = tf.reduce_sum(tf.square(conf_delta))
    loss_class = tf.reduce_sum(class_delta)
    loss = loss_xy+ loss_wh + loss_conf + loss_class

    loss = tf.Print(loss, [ avg_obj], message='avg_obj \t\t', summarize=1000)
    loss = tf.Print(loss, [ avg_noobj], message='avg_noobj \t\t', summarize=1000)
    loss = tf.Print(loss, [ avg_iou], message='avg_iou \t\t', summarize=1000)
    loss = tf.Print(loss, [ avg_cat], message='avg_cat \t\t', summarize=1000)
    loss = tf.Print(loss, [ recall50], message='recall50 \t', summarize=1000)
    loss = tf.Print(loss, [ recall75], message='recall75 \t', summarize=1000)
    loss = tf.Print(loss, [ count], message='count \t', summarize=1000)
    loss = tf.Print(loss, [ tf.reduce_sum(loss_xy),
                                   tf.reduce_sum(loss_wh),
                                   tf.reduce_sum(loss_conf),
                                   tf.reduce_sum(loss_class)],  message='loss xy, wh, conf, class: \t',   summarize=1000)
    loss = tf.Print(loss, [ tf.reduce_mean(wh_scale)], message='wh_scale')

    return loss*grid_scale

def new_loss(y_true, y_pred, anchors= anchors_list):

    anchors = tf.cast(anchors, dtype = tf.float32)
    anchors = tf.reshape(anchors, shape= [1,1,1,anchors.shape[0],3])  # преобразуем разметочные боксы к виду (1,1,1,len(anchors),3)

    # генерируем сетку для добавления к локальным координатам предсказания(они нормированы по остатку и на размер ячейки),
    # поэтому надо добавлять к ним сетку целых частей сетки. cell_grid(i,j,k) = i+j+k
    cell_x = tf.cast(tf.reshape(tf.tile(tf.range(max_grid_d), [max_grid_h*max_grid_w]), (max_grid_h, max_grid_w, max_grid_d, 1, 1)),dtype= tf.float32)
    cell_y = tf.transpose(cell_x, (1,2,0,3,4))
    cell_z = tf.transpose(cell_x, (2,0,1,3,4))
    cell_grid = tf.tile(tf.concat([cell_z,cell_y,cell_x], -1), [1,1,1,anchors.shape[3],1])

    input_image = tf.cast(np.zeros((IMAGE_H//striding,IMAGE_W//striding,IMAGE_D//striding)), tf.int32)

    y_true = K.reshape(y_true, tf.shape(y_true[0,...])) # squeeze this matrix

    y_pred = tf.reshape(y_pred, tf.shape(y_pred)[1:]) #squeeze another way this matrix

    object_mask     = tf.expand_dims(y_true[..., 4], 4) #unsqueeze this matrix

    # grid_h      = tf.shape(y_true)[0]
    # grid_w      = tf.shape(y_true)[1]
    # grid_d      = tf.shape(y_true)[2]
    grid_h = GRID_H
    grid_w = GRID_W
    grid_d = GRID_D

    grid_factor = tf.reshape(tf.cast([grid_h, grid_w, grid_d], tf.float32), [1,1,1,1,3])

    # net_h       = tf.shape(input_image)[0]
    # net_w       = tf.shape(input_image)[1]
    # net_d       = tf.shape(input_image)[2]
    net_h = IMAGE_H//striding
    net_w = IMAGE_W//striding
    net_d = IMAGE_D//striding
    net_factor  = tf.reshape(tf.cast([net_h, net_w, net_d], tf.float32), [1,1,1,1,3])

    pred_box_xy    = tf.sigmoid(y_pred[..., :N_DIM])
    pred_box_wh    = y_pred[..., N_DIM:N_DIM*2]                                                       # t_wh
    pred_box_conf  = tf.expand_dims(tf.sigmoid(y_pred[..., N_DIM*2]), 4)   # adjust confidence
    pred_box_class = K.softmax(y_pred[..., N_DIM+1:])

    true_box_xy    = y_true[..., 0:N_DIM] # [0:1]
    true_box_wh    = y_true[..., N_DIM:N_DIM*2] # t_wh - log(wh/anchors)
    true_box_conf  = tf.expand_dims(y_true[..., N_DIM*2], 4)
    true_box_class = y_true[...,N_DIM*2+1:]
    true_box_class = tf.argmax(y_true[...,N_DIM*2+1:],axis= -1)

    true_xy = (true_box_xy + cell_grid[:grid_h,:grid_w,:grid_d,:,:N_DIM]) * (net_factor[...,:N_DIM]/grid_factor[...,:N_DIM])
    true_wh = tf.exp(true_box_wh) * anchors[...,:N_DIM]

    true_wh_half = true_wh / 2.
    true_mins    = true_xy - true_wh_half
    true_maxes   = true_xy + true_wh_half

    pred_xy = (pred_box_xy + cell_grid[:grid_h,:grid_w,:grid_d,:,:N_DIM]) * (net_factor[...,:N_DIM]/grid_factor[...,:N_DIM])
    pred_wh = tf.exp(pred_box_wh) * anchors[...,:N_DIM]

    pred_wh_half = pred_wh / 2.
    pred_mins    = pred_xy - pred_wh_half
    pred_maxes   = pred_xy + pred_wh_half

    intersect_mins  = tf.maximum(pred_mins,  true_mins)
    intersect_maxes = tf.minimum(pred_maxes, true_maxes)
    intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)

    if N_DIM == 3:
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]* intersect_wh[...,2]
        true_areas = true_wh[..., 0] * true_wh[..., 1]* true_wh[...,2]
        pred_areas = pred_wh[..., 0] * pred_wh[..., 1] * pred_wh[...,2]
    else:
        # если N_DIM == 3: intersect_wh[..., 0] * intersect_wh[..., 1]* intersect[...,3]
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
        true_areas = true_wh[..., 0] * true_wh[..., 1]
        pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores  = tf.truediv(intersect_areas, union_areas)
    iou_scores  = object_mask * tf.expand_dims(iou_scores, 4)
    iou_delta = tf.to_float(iou_scores < ignore_thresh)*obj_scale

    count       = tf.reduce_sum(object_mask)
    count_noobj = tf.reduce_sum(1 - object_mask)
    detect_mask = tf.to_float((pred_box_conf*object_mask) >= obj_thresh)
    class_mask  = tf.expand_dims(tf.to_float(tf.equal(tf.argmax(pred_box_class, -1), true_box_class)), 4)
    # recall50    = tf.reduce_sum(tf.to_float(iou_scores >= 0.5 ) * detect_mask  * class_mask) / (count + 1e-3)
    # recall75    = tf.reduce_sum(tf.to_float(iou_scores >= 0.75) * detect_mask  * class_mask) / (count + 1e-3)
    recall50 = tf.reduce_sum(tf.to_float(iou_scores >= 0.5) * detect_mask ) / (count + 1e-3)
    recall75 = tf.reduce_sum(tf.to_float(iou_scores >= 0.75) * detect_mask ) / (count + 1e-3)
    avg_iou     = tf.reduce_sum(iou_scores) / (count + 1e-3)
    avg_obj     = tf.reduce_sum(pred_box_conf  * object_mask)  / (count + 1e-3)
    avg_noobj   = tf.reduce_sum(pred_box_conf  * (1-object_mask))  / (count_noobj + 1e-3)
    avg_cat     = tf.reduce_sum(object_mask * class_mask) / (count + 1e-3)

    true_box_xy, true_box_wh, xywh_mask = [true_box_xy, true_box_wh, object_mask]

    xy_delta = xywh_mask * (pred_box_xy - true_box_xy) * xywh_scale
    wh_delta    = xywh_mask   * (tf.sqrt(pred_wh) - tf.sqrt(true_wh)) * xywh_scale
    conf_delta  = xywh_mask * (pred_box_conf-true_box_conf) * obj_scale
    noobj_conf_delta = (1-xywh_mask) *(pred_box_conf-true_box_conf) * noobj_scale
    class_delta = object_mask * tf.expand_dims(tf.keras.backend.sparse_categorical_crossentropy(true_box_class, pred_box_class),4)* class_scale

    loss_xy = tf.reduce_sum(tf.square(xy_delta))
    loss_wh = tf.reduce_sum(tf.square(wh_delta))
    loss_conf = tf.reduce_sum(tf.square(conf_delta))
    loss_noobj_conf = tf.reduce_sum(tf.square(noobj_conf_delta))
    loss_class = tf.reduce_sum(class_delta)
    loss = loss_xy+ loss_wh + loss_conf + loss_class + loss_noobj_conf

    loss = tf.Print(loss, [ avg_obj], message='avg_obj \t\t', summarize=1000)
    loss = tf.Print(loss, [ avg_noobj], message='avg_noobj \t\t', summarize=1000)
    loss = tf.Print(loss, [ avg_iou], message='avg_iou \t\t', summarize=1000)
    loss = tf.Print(loss, [ avg_cat], message='avg_cat \t\t', summarize=1000)
    loss = tf.Print(loss, [ recall50], message='recall50 \t', summarize=1000)
    loss = tf.Print(loss, [ recall75], message='recall75 \t', summarize=1000)
    loss = tf.Print(loss, [ count], message='count \t', summarize=1000)
    loss = tf.Print(loss, [ tf.reduce_sum(loss_xy),
                                   tf.reduce_sum(loss_wh),
                                   tf.reduce_sum(loss_conf),
                                   tf.reduce_sum(loss_class), tf.reduce_sum(loss_noobj_conf)],  message='loss xy, wh, conf, class , noobj: \t',   summarize=1000)
    # loss = tf.Print(loss, [ tf.reduce_mean(wh_scale)], message='wh_scale')

    return loss*grid_scale


def model_loss():
    def yolo_loss(y_true,y_pred):
        return new_loss(y_true,y_pred)
    return yolo_loss

def save_logs(model,val_loss_arr, loss_arr):
    np.save('val_loss_progress.npy', np.ravel(np.array(val_loss_arr)))
    np.save('loss_progress.npy', np.ravel(np.array(loss_arr)))
    model.save('train_full_train.h5')
    model.save_weights('weights_good_loader.h5')



# json_model  = model.to_json()
# with open('model_arch.json', 'w') as json_file:
# json.dump(json_model, json_file)

# def predict_frame(model, frame):
#
#     pred = model.predict(frame)
#     targets = decode_netout(pred)
#
#     return