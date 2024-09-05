import numpy as np
import mayavi.mlab as mlab

from vkitti.frame import FrameDataLoader, FrameTransformMatrix, FrameLabels, project_pcl_to_image, min_max_filter, homogeneous_transformation

from .helpers import plot_boxes, get_2d_label_corners, get_transformed_3d_label_corners
from .settings import label_color_palette_2d


def plot_3d_boxes(fig, labels, colors, draw_text=True, text_scale=1.0, line_width=1.0):
    for label, color in zip(labels, colors):
        b = label
        if draw_text:
            mlab.text3d(
                b[4, 0],
                b[4, 1],
                b[4, 2],
                label,
                scale=text_scale,
                color=color,
                figure=fig,
            )
        for k in range(0, 4):
            i, j = k, (k + 1) % 4
            mlab.plot3d(
                [b[i, 0], b[j, 0]],
                [b[i, 1], b[j, 1]],
                [b[i, 2], b[j, 2]],
                color=color,
                tube_radius=None,
                line_width=line_width,
                figure=fig,
            )

            i, j = k + 4, (k + 1) % 4 + 4
            mlab.plot3d(
                [b[i, 0], b[j, 0]],
                [b[i, 1], b[j, 1]],
                [b[i, 2], b[j, 2]],
                color=color,
                tube_radius=None,
                line_width=line_width,
                figure=fig,
            )

            i, j = k, k + 4
            mlab.plot3d(
                [b[i, 0], b[j, 0]],
                [b[i, 1], b[j, 1]],
                [b[i, 2], b[j, 2]],
                color=color,
                tube_radius=None,
                line_width=line_width,
                figure=fig,
            )
    return fig


class Visualization3D:
    """
    This class is responsible for plotting a frame from the set, and visualize
     its image with its point clouds (radar and/or LiDAR), annotations projected and overlaid.
    """
    def __init__(self,
                 frame_data_loader: FrameDataLoader,
                 classes_visualized: list = ['Cyclist', 'Pedestrian', 'Car', 'Van', 'Bus', 'Truck']
                 ):
        """
Constructor of the class, which loads the required frame properties, and creates a copy of the picture data.
        :param frame_data_loader: FrameDataLoader instance.
        :param classes_visualized: A list of classes to be visualized.
        """
        self.frame_data_loader = frame_data_loader
        self.frame_transformations = FrameTransformMatrix(self.frame_data_loader)

        self.classes_visualized = classes_visualized
        self.cam_centered = False

        self.lidar_points = self.frame_data_loader.lidar_data

    def plot_range(self, fig):
        center = [0., 0., 0.]
        radius_list = [10, 20, 30, 40, 50]  # 半径
        start_angle = -np.pi / 3  # 起始角度（弧度）
        end_angle = np.pi / 3  # 结束角度（弧度）
        num_points = 100  # 点的数量

        # # 使用 Mayavi 绘制点
        # mlab.figure(bgcolor=(1, 1, 1))  # 设置背景颜色为白色

        # 生成圆弧上的点
        for radius in radius_list:
            angles = np.linspace(start_angle, end_angle, num_points)
            x = center[0] + radius * np.cos(angles)
            y = center[1] + radius * np.sin(angles)
            z = np.zeros_like(x)  # 所有点都在 z=0 的平面上
            mlab.points3d(x, y, z, mode='sphere', color=(0, 0, 1), scale_factor=0.5)  # 绘制蓝色的点

        # 生成fov
        x_max_s = center[0] + radius_list[-1] * np.cos(start_angle)
        y_max_s = center[1] + radius_list[-1] * np.sin(start_angle)
        x = np.linspace(0, x_max_s, num_points)
        y = np.linspace(0, y_max_s, num_points)
        z = np.zeros_like(x)  # 所有点都在 z=0 的平面上
        mlab.points3d(x, y, z, mode='sphere', color=(0, 0, 1), scale_factor=0.5)  # 绘制蓝色的点

        x_max_e = center[0] + radius_list[-1] * np.cos(end_angle)
        y_max_e = center[1] + radius_list[-1] * np.sin(end_angle)
        x = np.linspace(0, x_max_e, num_points)
        y = np.linspace(0, y_max_e, num_points)
        z = np.zeros_like(x)  # 所有点都在 z=0 的平面上
        mlab.points3d(x, y, z, mode='sphere', color=(0, 0, 1), scale_factor=0.5)  # 绘制蓝色的点

        # 定义坐标轴的长度
        axis_length = 1.0

        # 在原点处绘制X轴箭头
        x_axis = np.array([[center[0], center[1], center[2]],
                           [center[0] + axis_length, center[1], center[2]]])
        mlab.quiver3d(x_axis[:, 0], x_axis[:, 1], x_axis[:, 2], color=(1, 0, 0), mode='arrow', scale_factor=1)

        # 在原点处绘制Y轴箭头
        y_axis = np.array([[center[0], center[1], center[2]],
                           [center[0], center[1] + axis_length, center[2]]])
        mlab.quiver3d(y_axis[:, 0], y_axis[:, 1], y_axis[:, 2], color=(0, 1, 0), mode='arrow', scale_factor=1)

        # 在原点处绘制Z轴箭头
        z_axis = np.array([[center[0], center[1], center[2]],
                           [center[0], center[1], center[2] + axis_length]])
        mlab.quiver3d(z_axis[:, 0], z_axis[:, 1], z_axis[:, 2], color=(0, 0, 1), mode='arrow', scale_factor=1)

        mlab.text3d(1, 0, 0, 'X', color=(0, 0, 1), scale=0.2)
        mlab.text3d(0, 1, 0, 'Y', color=(0, 0, 1), scale=0.2)
        mlab.text3d(0, 0, 1, 'Z', color=(0, 0, 1), scale=0.2)

    def plot_gt_labels(self, fig, max_distance_threshold):
        """
This method plots the ground truth labels on the frame.
        :param max_distance_threshold: The maximum distance where labels are rendered.
        """
        frame_labels_class = FrameLabels(self.frame_data_loader.raw_labels)
        frame_labels = frame_labels_class.labels_dict

        # Class filter
        filtered = list(filter(lambda elem: elem['label_class'] in self.classes_visualized, frame_labels))

        # Distance filter
        filtered = list(filter(lambda elem: elem['range'] < max_distance_threshold, filtered))

        colors = [label_color_palette_2d[v["label_class"]] for v in filtered]

        plot_3d_boxes(fig, filtered, colors)

    def plot_predictions(self, fig, score_threshold, max_distance_threshold):
        """
This method plots the prediction labels on the frame.
        :param score_threshold: The minimum score to be rendered.
        :param max_distance_threshold: The maximum distance where labels are rendered.
        """
        frame_labels_class = FrameLabels(self.frame_data_loader.raw_labels)
        frame_labels = frame_labels_class.labels_dict

        # Class filter
        filtered = list(filter(lambda elem: elem['label_class'] in self.classes_visualized, frame_labels))

        # Distance filter
        filtered = list(filter(lambda elem: elem['range'] < max_distance_threshold, filtered))

        colors = [label_color_palette_2d[v["label_class"]] for v in filtered]

        plot_3d_boxes(fig, filtered, colors)

    def plot_lidar_pcl(self, fig, max_distance_threshold, min_distance_threshold):
        """
This method plots the lidar pcl on the frame. It colors the points based on distance.
        :param max_distance_threshold: The maximum distance where points are rendered.
        :param min_distance_threshold: The minimum distance where points are rendered.
        """
        t_camera_lidar = self.frame_transformations.t_camera_lidar
        t_camera_lidar[1, 3] = -0.5  # lidar 1.7m - 2.2m
        point_homo = np.hstack((self.lidar_points[:, :3],
                                np.ones((self.lidar_points.shape[0], 1),
                                        dtype=np.float32)))

        points_camera_frame = homogeneous_transformation(point_homo,
                                                         transform=t_camera_lidar)

        x = self.lidar_points[:, 0]
        y = self.lidar_points[:, 1]
        z = self.lidar_points[:, 2]

        d = np.sqrt(x ** 2 + y ** 2)
        col = self.lidar_points[:, 2]

        mlab.points3d(x, y, z,
                      col,
                      scale_factor=5,
                      mode='point',
                      colormap='spectral',  # bone, copper, gnuplot
                      figure=fig,
                      )


    def draw_plot(self,
                  cam_centered: bool = True,
                  show_gt: bool = False,
                  show_pred: bool = False,
                  show_lidar: bool = False,
                  show_range: bool = False,
                  max_distance_threshold: float = 50.0,
                  min_distance_threshold: float = 0.0,
                  score_threshold: float = 0, ):
        """
This method can be called to draw the frame with the required information.
        :param show_gt: Should the ground truth be plotted.
        :param show_pred: Should the predictions be plotted.
        :param show_lidar: Should the lidar pcl be plotted.
        :param max_distance_threshold: Maximum distance of objects to be plotted.
        :param min_distance_threshold:  Minimum distance of objects to be plotted.
        :param score_threshold: Minimum score for objects to be plotted.
        """
        fig = mlab.figure(
            figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1600, 1000)
        )

        self.cam_centered = cam_centered

        if show_gt:
            self.plot_gt_labels(fig=fig, max_distance_threshold=max_distance_threshold)

        if show_pred:
            self.plot_predictions(fig=fig, max_distance_threshold=max_distance_threshold,
                                  score_threshold=score_threshold)

        if show_lidar:
            self.plot_lidar_pcl(fig=fig, max_distance_threshold=max_distance_threshold,
                                min_distance_threshold=min_distance_threshold)

        if show_range:
            self.plot_range(fig=fig)

        mlab.show()