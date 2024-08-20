import numpy as np
import mayavi.mlab as mlab


def show3D(pc_velo):
    # 可视化
    fig = mlab.figure(
                figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1600, 1000)
            )

    x = pc_velo[:, 0]
    y = pc_velo[:, 1]
    z = pc_velo[:, 2]

    d = np.sqrt(x ** 2 + y ** 2)
    col = pc_velo[:, 2]

    mlab.points3d(x, y, z,
                  col,
                  mode='point',
                  colormap='spectral',  # bone, copper, gnuplot
                  figure=fig
     )

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

    mlab.show()


def show3D_simple(pc_velo):
    # 可视化
    fig = mlab.figure(
                figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1600, 1000)
            )

    x = pc_velo[:, 0]
    y = pc_velo[:, 1]
    z = pc_velo[:, 2]

    col = pc_velo[:, 2]

    mlab.points3d(x, y, z,
                  col,
                  mode='point',
                  colormap='spectral',  # bone, copper, gnuplot
                  figure=fig
     )
    mlab.show()


if __name__ == '__main__':
    pc_dir = r"/home/hosico/DataDisk/hdd2/DataGenerator-New/data/test/training/velodyne/000012.bin"
    pointcloud = np.fromfile(pc_dir, dtype=np.float32, count=-1).reshape([-1, 4])
    show3D_simple(pointcloud)

