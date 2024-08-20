import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from vkitti.configuration import KittiLocations
from vkitti.frame import FrameDataLoader
from vkitti.frame import FrameTransformMatrix
from vkitti.frame.transformations import homogeneous_transformation

sys.path.append(os.path.abspath(os.path.join(__file__, '../..')))


def getCalib():
    intrisics = np.array([[640.000000, 0.000000, 640.000000], [0.000000, 640.000000, 320.000000], [0.0, 0.0, 1.0]])
    return intrisics


def rotateRect(rect, rotation_angle):
    x, y, w, l = rect
    # 将角度转换为弧度
    radians = math.radians(rotation_angle)

    # 计算矩形四个角相对于中心点的偏移量
    # 左上角 (-w/2, l/2)
    # 右上角 (w/2, l/2)
    # 右下角 (w/2, -l/2)
    # 左下角 (-w/2, -l/2)

    # 旋转矩阵
    # [[cos(theta), -sin(theta)],
    #  [sin(theta),  cos(theta)]]

    # 旋转左上角
    x1_offset = -w / 2 * math.cos(radians) + l / 2 * math.sin(radians)
    y1_offset = -w / 2 * math.sin(radians) - l / 2 * math.cos(radians)

    # 旋转右上角
    x2_offset = w / 2 * math.cos(radians) + l / 2 * math.sin(radians)
    y2_offset = w / 2 * math.sin(radians) - l / 2 * math.cos(radians)

    # 旋转右下角和左下角与右上角和左上角对称，不需要额外计算

    # 将旋转后的偏移量加回到中心点上
    x1, y1 = x + x1_offset, y + y1_offset
    x2, y2 = x + x2_offset, y + y2_offset

    # 由于我们只计算了左上角和右上角的坐标，但通常我们需要矩形的两个对角顶点
    # 假设我们要的是左上角和右下角作为对角顶点
    return [x1, y1, x + x2_offset, y - y2_offset]


def fov_mask(intrisics, img_width, map_extents, map_resolution):
    fu, cu = intrisics[0, 0], intrisics[0, 2]

    # Construct a grid of image coordinates
    x1, z1, x2, z2 = map_extents
    x, z = np.arange(x1, x2, map_resolution), np.arange(z1, z2, map_resolution)
    ucoords = x / z[:, None] * fu + cu

    ucoords = (ucoords >= 0) & (ucoords < img_width)
    # ucoords = np.flip(ucoords, axis=0)
    # ucoords = np.flip(ucoords, axis=1)

    # Return all points which lie within the camera bounds
    return ucoords


def vis_img(image):
    plt.imshow(image, cmap='coolwarm')
    plt.axis('on')
    plt.show()


def vis_pc_2d_img(mask, point_cloud, labels, map_extents, map_resolution, name_index):
    fig, ax = plt.subplots()
    ax.imshow(mask)
    x1, z1, x2, z2 = map_extents
    cols = np.where(point_cloud[:, 2] > z1)
    point_cloud_new = point_cloud[cols]
    cols = np.where(point_cloud_new[:, 2] < z2)
    point_cloud_new = point_cloud_new[cols]
    cols = np.where(point_cloud_new[:, 0] > x1)
    point_cloud_new = point_cloud_new[cols]
    cols = np.where(point_cloud_new[:, 0] < x2)
    point_cloud_new = point_cloud_new[cols]
    if labels:
        for label in labels:
            print(np.rad2deg(label['rotation']) - 90)
            # rect = Rectangle((label['x'] / map_resolution + 100, label['z'] / map_resolution), label['w'] / map_resolution,
            #                  label['l'] / map_resolution, np.rad2deg(label['rotation']) - 90,
            #                  linewidth=2, edgecolor='r', facecolor='r')
            rect = Rectangle((label['x'] / map_resolution + 100 - 5, label['z'] / map_resolution - 10), label['w'] / map_resolution,
                             label['l'] / map_resolution, np.rad2deg(label['rotation']),
                             linewidth=2, edgecolor='r', facecolor='r')
            # rect = Rectangle(((label['x'] - label['l'] / 2) / map_resolution + 100, (label['z'] - label['w'] / 2) / map_resolution),
            #                  label['w'] / map_resolution, label['l'] / map_resolution, np.rad2deg(label['rotation']) - 90,
            #                  linewidth=2, edgecolor='r', facecolor='r')
            ax.add_patch(rect)
    if run_radar:
        ax.scatter(point_cloud_new[:, 2] / map_resolution + 100, -point_cloud_new[:, 0] / map_resolution, s=1)
    else:
        ax.scatter(point_cloud_new[:, 0] / map_resolution + 100, point_cloud_new[:, 2] / map_resolution, s=1)
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 200)

    print("saving ...", './dualradar_lidar/' + format(int(name_index) - 1, f'05d') + '.png')
    # plt.savefig('./dualradar_lidar/' + format(int(name_index) - 1, f'05d') + '.png')
    plt.show()


def vis_pc_2d(mask, point_cloud, labels):
    fig, ax = plt.subplots()
    if run_radar:
        ax.scatter(point_cloud[:, 2], point_cloud[:, 0])
    else:
        ax.scatter(point_cloud[:, 0], point_cloud[:, 2])
    if labels:
        for label in labels:
            print(np.rad2deg(label['rotation']))
            rect = Rectangle((label['x'], label['z']), label['w'], label['l'], np.rad2deg(label['rotation']),
                             linewidth=2, edgecolor='r', facecolor='r')
            ax.add_patch(rect)
    plt.show()


def extractLabels(label_str):
    labels = []  # List to be filled
    for act_line in label_str:  # Go line by line to split the keys
        act_line = act_line.split()
        # type, truncated, occluded, alpha, bbox_x1, bbox_y1, bbox_x2, bbox_y2, h, w, l, x, y, z, rot, score, track_id
        type, truncated, occluded, alpha, bboxx1, bboxy1, bboxx2, bboxy2, h, w, l, x, y, z, rot = act_line
        h, w, l, x, y, z, rot = map(float, [h, w, l, x, y, z, rot])

        labels.append({'label_class': type,
                       'h': h,
                       'w': w,
                       'l': l,
                       'x': x,
                       'y': y,
                       'z': z,
                       'rotation': rot}
                      )
    return labels


run_radar = False

if __name__ == '__main__':

    # config = get_default_configuration()
    # config.merge_from_file('../configs/datasets/vod.yml')

    # Create an vod instance
    root_dir = r"/home/hosico/DataDisk/hdd2/DataGenerator-New/data/test/training"
    kitti_locations = KittiLocations(is_train=True,
                                     root_dir=root_dir,
                                     output_dir="example_output",
                                     frame_set_path="",
                                     pred_dir="",
                                     )

    # get train and val
    train_list = []
    val_list = []
    train_txt = os.path.join(root_dir, 'train.txt')
    val_txt = os.path.join(root_dir, 'val.txt')
    with open(train_txt, 'r') as file:
        lines = file.readlines()
        for line in lines:
            train_list.append(line.strip())

    with open(val_txt, 'r') as file:
        lines = file.readlines()
        for line in lines:
            val_list.append(line.strip())

    intrinsic = getCalib()
    for split in [train_list, val_list]:
        for frame_index in split:
            frame_data = FrameDataLoader(kitti_locations=kitti_locations, frame_number=frame_index)
            transforms = FrameTransformMatrix(frame_data)
            point_homo = np.hstack((frame_data.lidar_data[:, :3], np.ones((frame_data.lidar_data.shape[0], 1), dtype=np.float32)))
            points_camera_frame = homogeneous_transformation(point_homo, transform=transforms.t_camera_lidar)[:, :3]
            labels = extractLabels(frame_data.raw_labels)
            image_h, image_w, image_c = frame_data.image.shape
            mask = fov_mask(intrinsic, image_w, [-25, 0, 25, 50], 0.25)
            # vis_img(mask)
            vis_pc_2d_img(mask, points_camera_frame, labels, [-25, 0, 25, 50], 0.25, frame_index)

            # print(labels)