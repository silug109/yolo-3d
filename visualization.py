import glob

import cv2
import imageio
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import open3d as o3d

from data_util import *


def visualise_bird_view(frame ,targets, boxes, do_nms = False, do_photo = False, nms = False):
    '''

    :param frame:
    :param targets:
    :param boxes:
    :param do_nms:
    :param do_photo:
    :param nms:
    :return:
    '''


    fig, axs = plt.subplots(1 ,2, figsize = (10 ,10))

    im = np.sum(frame ,axis = (0 ,3 ,4))

    if nms == True:
        boxes = do_nms_exp(boxes, 0.05)
        print(len(boxes))

    axs[0].imshow(im)
    axs[1].imshow(im)
    x_ticks = np.arange(0, im.shape[1], 20)
    y_ticks = np.arange(0, im.shape[0], 20)
    axs[0].set_xticks(x_ticks)
    axs[0].set_yticks(y_ticks)
    axs[1].set_xticks(x_ticks)
    axs[1].set_yticks(y_ticks)
    axs[0].set_title('target boxes')
    axs[1].set_title('predicted boxes')

    for num_target in range(targets.shape[1]):
        x = targets[0 ,num_target ,2]
        y = targets[0 ,num_target ,1]
        w = targets[0 ,num_target ,4]
        h = targets[0 ,num_target ,3]
        x_t = x- w / 2
        x_b = x + w / 2
        y_t = y - h / 2
        y_b = y + h / 2
        rect = patches.Rectangle((x_t, y_t), w, h, linewidth=1, edgecolor='r', facecolor='none')
        axs[0].add_patch(rect)

    for box in boxes:
        x = box[1]
        y = box[0]
        w = box[3]
        h = box[2]
        x_t = x - w / 2
        x_b = x + w / 2
        y_t = y - h / 2
        y_b = y + h / 2
        class_box = box[5]
        rect = patches.Rectangle((x_t, y_t), w, h, linewidth=1, edgecolor= color_dict[class_box], facecolor='none')
        axs[1].add_patch(rect)

    if do_photo:
        plt.savefig('comparison.png')


def plot_logs(val_loss_arr, loss_arr, save = False):
    plt.figure( figsize= (10,10))
    plt.plot(np.ravel(np.array(val_loss_arr)))
    plt.plot(np.ravel(np.array(loss_arr)))
    if save:
        plt.savefig('results/train_good_loader.png')
    plt.show()

def show_comparison_object_level(y_true, y_pred, save = False):
    fig, axs = plt.subplots(1, 3)
    fig.suptitle('confidence map')
    axs[0].imshow(np.sum(y_true, axis=(0, 3, 4, 5)))
    axs[0].set_title('true map')
    axs[1].imshow(y_true[0, :, :, 2, 0, N_DIM*2])
    axs[2].set_title('pred map')
    axs[2].imshow(sigmoid(y_pred[0, :, :, 2, 0, N_DIM*2]))

    if save:
        plt.savefig('zhopa.jpg')


def erosion_frame(frame):
    kernel = np.ones((7, 7), np.uint8)
    erosion = cv2.dilate(frame[0, ..., 0], kernel, iterations=1)
    return erosion

def histogram_equalization(frame):
    '''
    function returns 3d frame with equalised histogram
    :param frame: numpy tensor of shape (1,GRID_H, GRID_W, GRID_D, num_anchors, num_classes+4/6/+1)
    :return: same shape frame
    '''

    histogram_array = np.histogram(frame[0, ..., 0], bins=256)
    cdf = histogram_array[0].cumsum()
    cdf_normalized = cdf * histogram_array[0].max() / cdf.max()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    frame_equalised = cdf[frame[0, ..., 0]]
    return frame_equalised



# 3D visualization

def pointcloud_coords_generation(frame, range_max = 67, azimuth_range_max = 57, elevation_max = 16):
    '''
    :param frame: (config.size[1], size[2], config.size[3])
    :return: ndarray(num_points, 4)
    '''
    R = np.arange(0,range_max, range_max/512)
    theta = np.arange(-azimuth_range_max, azimuth_range_max, 2*azimuth_range_max/128)
    epsilon = np.arange(0,elevation_max,elevation_max/40)

    points_cord = []
    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
            for k in range(0,frame.shape[2]-6):
                if frame[i,j,k] > 0.1:

                    x = R[i]* np.cos(theta[j]*np.pi/180)* np.cos(epsilon[k]*np.pi/180)
                    y = R[i]* np.sin(theta[j]*np.pi/180)* np.cos(epsilon[k]*np.pi/180)
                    z = R[i]* np.sin(epsilon[k]*np.pi/180)

                    points_cord.append([x,y,z,frame[i,j,k]])

    points_cord = np.array(points_cord)
    colors_arr  = np.swapaxes(np.vstack((points_cord[:,3],points_cord[:,3],points_cord[:,3]))/255,0,1)
    return points_cord , colors_arr

def create_pointcloud(points_coord, colors_arr = None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_coord[:, :3])
    # if colors_arr != None:
    #     pcd.colors = o3d.utility.Vector3dVector()
    return pcd

def sample_visualization(points_coord):
    pcd = create_pointcloud(points_coord)
    o3d.visualization.draw_geometries([pcd])
    return 'success'

def line_creation(R, visualize = False):
    R = 60
    points = [[0,0,0]] + [[R*np.cos(azi*np.pi/180), R*np.sin(azi*np.pi/180), 0] for azi in range(-90,90,10)]
    lines = [[0,i+1] for i in range(len(points)- 1)]
    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    if visualize:
        o3d.visualization.draw_geometries([line_set])
    return line_set


def custom_draw_geometry_with_key_callback(matr):
    global counter
    counter = 0

    frame = load_from_matfile(matr, num_frame=0)

    points_cord, colors_arr = pointcloud_coords_generation(frame)
    pcd = create_pointcloud(points_cord, colors_arr)

    line_set = line_creation(60)

    fig = plt.figure(figsize=(20, 20))

    def change_background_to_black(vis):
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        return False

    def load_render_option(vis):
        vis.get_render_option().load_from_json(
            "../../TestData/renderoption.json")
        return False

    def capture_depth(vis):
        depth = vis.capture_depth_float_buffer()
        plt.imshow(np.asarray(depth))
        plt.show()
        return False

    def capture_image(vis):
        image = vis.capture_screen_float_buffer()
        plt.imshow(np.asarray(image))
        plt.show()
        return False

    def change_frame(vis):
        ctr = vis.get_view_control()
        fov = ctr.get_field_of_view()
        #         print(fov)
        num_frame = np.random.randint(0, 40)
        new_pcd_attr, new_colors_arr = pointcloud_coords_generation(load_from_matfile(matr, num_frame))
        new_pcd_cords = new_pcd_attr[0][:, :3]
        pcd.points = o3d.utility.Vector3dVector(new_pcd_cords)
        #         pcd.colors = o3d.utility.Vector3dVector(new_pcd_colors)
        vis.update_geometry()

        return False

    def next_frame(vis):
        global counter

        ctr = vis.get_view_control()
        fov = ctr.get_field_of_view()
        num_frame = counter

        try:
            # new_pcd_attr = frames_np[num_frame]
            new_frame = load_from_matfile(matr, num_frame)
        except IndexError:
            counter = 0
            num_frame = counter
            # new_pcd_attr = frames_np[num_frame]
            new_frame = load_from_matfile(matr, num_frame)

        frame = new_frame
        new_pcd_attr = pointcloud_coords_generation(frame)

        new_pcd_cords = new_pcd_attr[0][:, :3]
        #         new_colors = new_pcd_attr[1]
        pcd.points = o3d.utility.Vector3dVector(new_pcd_cords)
        #         pcd.colors = o3d.utility.Vector3dVector(new_colors)
        vis.update_geometry()
        image = vis.capture_screen_float_buffer()
        plt.imshow(image)
        plt.savefig('images2gif/' + str(counter) + '.png')

        print(counter, num_frame)
        counter += 1
        return False

    key_to_callback = {}
    key_to_callback[ord("K")] = change_background_to_black
    key_to_callback[ord("R")] = load_render_option
    key_to_callback[ord(",")] = capture_depth
    key_to_callback[ord(".")] = capture_image
    key_to_callback[ord("Z")] = change_frame
    key_to_callback[ord("X")] = next_frame
    o3d.visualization.draw_geometries_with_key_callbacks([line_set, pcd], key_to_callback)





def save_as_bin(points_cord, filename = 'something.bin'):
    point_array = points_cord.astype(np.float32)
    point_array = point_array.reshape((-1))
    point_array.tofile(filename)
    return 'Saved'

def make_gif(directory):
    # images_listpath = glob.glob('images2gif/*.png')
    images_listpath = glob.glob(directory+'*.png')
    right_order = sorted(images_listpath, key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))

    images = []
    for filename in right_order:
        images.append(imageio.imread(filename))
    imageio.mimsave('images2gif/movie.gif', images)

    return 'Success'


# def histogram_equalization(frame):
    # return frame_equalised

# def labels(frame):



# fig,axs = plt.subplots(1,2, figsize = (10,10))
# axs[0].imshow(np.sum(frame[0,...,0],2))
# axs[1].imshow(np.sum(erosion,2))

