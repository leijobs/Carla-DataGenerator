from vkitti.configuration import KittiLocations
from vkitti.frame import FrameDataLoader
from vkitti.visualization import Visualization2D

kitti_locations = KittiLocations(is_train=True, root_dir=r"E:\Dataset\VKITTI\Dataset2\data\test",
                                output_dir="example_output")

for i in range(5685, 5695):
    frame_num = str(i).zfill(6)
    frame_data = FrameDataLoader(kitti_locations=kitti_locations,
                                 frame_number=frame_num)

    vis2d = Visualization2D(frame_data)

    vis2d.draw_plot(show_lidar=True,
                    show_2d=True,
                    show_gt=True,
                    min_distance_threshold=5,
                    max_distance_threshold=70,
                    save_figure=False)
