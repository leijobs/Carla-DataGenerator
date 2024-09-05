from vkitti.configuration import KittiLocations
from vkitti.frame import FrameDataLoader
from vkitti.visualization import Visualization3D

kitti_locations = KittiLocations(is_train=True, root_dir=r"E:\Dataset\VKITTI",
                                output_dir="example_output")

for i in range(1, 240):
    frame_num = str(i).zfill(6)
    frame_data = FrameDataLoader(kitti_locations=kitti_locations,
                                 frame_number=frame_num)

    vis3d = Visualization3D(frame_data)

    vis3d.draw_plot(cam_centered=True,
                    show_gt=False,
                    show_pred=False,
                    show_range=True,
                    show_lidar=True,
                    min_distance_threshold=5,
                    max_distance_threshold=50)