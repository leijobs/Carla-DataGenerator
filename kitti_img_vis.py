import matplotlib.pyplot as plt
import cv2
from kitti_object import kitti_object, show_image_with_boxes


if __name__ == '__main__':
    dataset = kitti_object('./data/test12', 'training')

    data_idx = 0
    for data_idx in range(0,100,1):
        print("data_idx is :", data_idx)
        objects = dataset.get_label_objects(data_idx)
        pc_velo = dataset.get_lidar(data_idx)
        calib = dataset.get_calibration(data_idx)
        img = dataset.get_image(data_idx)

        img_bbox2d, img_bbox3d = show_image_with_boxes(img, objects, calib)
        cv2.waitKey(0)

    

