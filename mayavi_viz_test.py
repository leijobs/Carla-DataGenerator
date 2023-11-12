# import numpy as np
# from mayavi import mlab
#
# t=np.linspace(0,4*np.pi,125)
# x=np.sin(2*t)
# y=np.cos(t)
# z=2*t
# u=4+np.sin(3*t)
# v=3+2*np.cos(t)
# w=2+2*t
# s=2+np.sin(t/5)
#
# points=mlab.points3d(x,y,z,s,colormap='jet',scale_factor=.25)
# points=mlab.points3d(u,v,w,s,colormap='jet',scale_factor=.25)
# mlab.show()

# coding=utf-8
import numpy as np
import mayavi.mlab


# lidar_path换成自己的.bin文件路径
def main(lidar_path):
    pointcloud = np.fromfile(lidar_path, dtype=np.float32, count=-1).reshape([-1, 4])

    x = pointcloud[:, 0]  # x position of point
    y = pointcloud[:, 1]  # y position of point
    z = pointcloud[:, 2]  # z position of point

    r = pointcloud[:, 3]  # reflectance value of point
    d = np.sqrt(x ** 2 + y ** 2)  # Map Distance from sensor

    degr = np.degrees(np.arctan(z / d))

    vals = 'height'
    if vals == "height":
        col = z
    else:
        col = d

    fig = mayavi.mlab.figure(bgcolor=(0, 0, 0), size=(1080, 720))
    mayavi.mlab.points3d(x, y, z,
                         col,  # Values used for Color
                         mode="point",
                         colormap='spectral',  # 'bone', 'copper', 'gnuplot'
                         # color=(0, 1, 0),   # Used a fixed (r,g,b) instead
                         figure=fig,
                         )

    mayavi.mlab.show()


if __name__ == '__main__':
    lidar_path = './data/test11/training/velodyne/000017.bin'
    main(lidar_path)