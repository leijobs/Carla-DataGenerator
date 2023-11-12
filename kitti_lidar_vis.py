import mayavi.mlab as mlab
from kitti_object import kitti_object, show_image_with_boxes, show_lidar_on_image, \
    show_lidar_with_boxes, show_lidar_topview_with_boxes, get_lidar_in_image_fov, \
    show_lidar_with_depth
from viz_util import draw_lidar
import cv2
from PIL import Image
import time

class visualization:
    # data_idx: determine data_idx
    def __init__(self, root_dir=r'F:/CARLA_Simulator/CARLA_0.9.12/WindowsNoEditor/PythonAPI/examples/DataGenerator/data/test12/', data_idx=0):
        dataset = kitti_object(root_dir=root_dir)

        # Load data from dataset
        objects = dataset.get_label_objects(data_idx)
        print("There are {} objects.".format(len(objects)))
        img = dataset.get_image(data_idx)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_height, img_width, img_channel = img.shape
        pc_velo = dataset.get_lidar(data_idx)[:, 0:3]  # 显示bev视图需要改动为[:, 0:4]
        calib = dataset.get_calibration(data_idx)

        # init the params
        self.objects = objects
        self.img = img
        self.img_height = img_height
        self.img_width = img_width
        self.img_channel = img_channel
        self.pc_velo = pc_velo
        self.calib = calib

    # 1. 图像显示
    def show_image(self):
        Image.fromarray(self.img).show()
        cv2.waitKey(0)

    # 2. 图片上绘制2D bbox
    def show_image_with_2d_boxes(self):
        show_image_with_boxes(self.img, self.objects, self.calib, show3d=False)
        cv2.waitKey(0)

    # 3. 图片上绘制3D bbox
    def show_image_with_3d_boxes(self):
        show_image_with_boxes(self.img, self.objects, self.calib, show3d=True)
        cv2.waitKey(0)

    # 4. 图片上绘制Lidar投影
    def show_image_with_lidar(self):
        show_lidar_on_image(self.pc_velo, self.img, self.calib, self.img_width, self.img_height)
        mlab.show()

    # 5. Lidar绘制3D bbox
    def show_lidar_with_3d_boxes(self):
        show_lidar_with_boxes(self.pc_velo, self.objects, self.calib, True, self.img_width, self.img_height)
        mlab.show()

    # 6. Lidar绘制FOV图
    def show_lidar_with_fov(self):
        imgfov_pc_velo, pts_2d, fov_inds = get_lidar_in_image_fov(self.pc_velo, self.calib,
                                                                  0, 0, self.img_width, self.img_height, True)
        draw_lidar(imgfov_pc_velo)
        mlab.show()

    # 7. Lidar绘制3D图
    def show_lidar_with_3dview(self):
        draw_lidar(self.pc_velo)
        mlab.show()

    # 8. Lidar绘制BEV图
    def show_lidar_with_bev(self):
        from kitti_util import draw_top_image, lidar_to_top
        top_view = lidar_to_top(self.pc_velo)
        top_image = draw_top_image(top_view)
        cv2.imshow("top_image", top_image)
        cv2.waitKey(0)

    # 9. Lidar绘制BEV图+2D bbox
    def show_lidar_with_bev_2d_bbox(self):
        show_lidar_topview_with_boxes(self.pc_velo, self.objects, self.calib)
        mlab.show()


if __name__ == '__main__':
    kitti_vis = visualization()
    # kitti_vis.show_image()
    # kitti_vis.show_image_with_2d_boxes()
    # kitti_vis.show_image_with_3d_boxes()
    # kitti_vis.show_image_with_lidar()
    kitti_vis.show_lidar_with_3d_boxes()
    # kitti_vis.show_lidar_with_fov()
    # kitti_vis.show_lidar_with_3dview()
    # kitti_vis.show_lidar_with_bev()
    # kitti_vis.show_lidar_with_bev_2d_bbox()

    # print('...')
    # cv2.waitKey(0)
