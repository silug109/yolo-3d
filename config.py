import os

# class_dict_counter = {49.0: 1608, 53.0: 548, 52.0: 73, 51.0: 266, 50.0: 1} #distributu
class_dict = {49.0: 0,50.0: 1,51.0:2, 52.0: 3, 53.0: 4}
color_dict = {0: 'r', 1: 'g', 2:'b', 3:'y' , 4:'c' }
labels_model= ['auto','people','tree','beton','sign']
num_classes = len(labels_model)

anchors_list = [[40,120,40],[35,60,40],[205,110,40], [40,100,40], [50,150,40],[75,110,40]] #new new ones
num_boxes = len(anchors_list)



ALPHA = 0.1
N_DIM = 2


IMAGE_H, IMAGE_W, IMAGE_D = (1024,256,80)
GRID_H, GRID_W, GRID_D = (16,8,5)
striding = int(2)
max_grid_h, max_grid_w, max_grid_d = (16, 16, 16)


ignore_thresh = 0.4
grid_scale = 1
obj_scale = 5
noobj_scale = 30
xywh_scale = 1
class_scale = 1
obj_thresh = 0.6
iou_thresh = 0.6








class config_model():
    def __init__(self):

        self.input_shape = (1024,512,80)
        self.netout_shape = (16,8,5)
        self.labels = ['auto','people','tree','beton','sign']
        self.num_classes = len(self.labels)
        self.anchors_list = [[40, 120, 40], [35, 60, 40], [205, 110, 40], [40, 100, 40], [50, 150, 40], [75, 110, 40]]

        self.num_boxes = len(self.anchors_list)

        self.class_dict = {49.0: 0, 50.0: 1, 51.0: 2, 52.0: 3, 53.0: 4}
        self.color_dict = {0: 'r', 1: 'g', 2: 'b', 3: 'y', 4: 'c'}

        self.BATCH_SIZE = 1
        self.WARM_UP_BATCHES = 0
        self.TRUE_BOX_BUFFER = 50
        self.IMAGE_H, self.IMAGE_W, self.IMAGE_D = (1024, 256, 80)
        self.GRID_H, self.GRID_W, self.GRID_D = (16, 8, 5)
        self.ALPHA = 0.1


    def display(self):
        print([item for item in  dir(self) if not(item.startswith('__'))])


from pathlib import Path
pathname = os.path.abspath('data')
val_dir = Path('val_data')
if val_dir.is_dir():
    directory_val = os.path.abspath('val_data')
    pathname_val = os.listdir(directory_val)



