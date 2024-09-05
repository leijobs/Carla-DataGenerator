import sys

sys.path.append("/home/hosico/DataDisk/hdd2/carla-0911/CARLA/PythonAPI/carla/dist/carla-0.9.11-py3.7-linux-x86_64.egg")

from config import config_to_trans
from export_utils import *
import numpy as np
import carla
from PIL import Image
import sys

sys.path.append("..")

SEM_COLORS = {

    6: (157, 234, 50),  # ROAD LINE
    7: (128, 64, 128),  # ROAD

}


class DataSave:
    def __init__(self, cfg):
        self.cfg = cfg
        self.OUTPUT_FOLDER = None
        self.LIDAR_PATH = None
        self.KITTI_LABEL_PATH = None
        self.CARLA_LABEL_PATH = None
        self.IMAGE_PATH = None
        self.IMAGE_SEG_PATH = None
        self.IMU_PATH = None
        self.GNSS_PATH = None
        self.CALIBRATION_PATH = None
        self._generate_path(self.cfg["SAVE_CONFIG"]["ROOT_PATH"])
        self.captured_frame_no = self._current_captured_frame_num()

    def _generate_path(self, root_path):
        """ 生成数据存储的路径"""
        PHASE = "training"
        self.OUTPUT_FOLDER = os.path.join(root_path, PHASE)
        folders = ['calib', 'image', 'image_seg', 'imu', 'gnss', 'label', 'velodyne']

        for folder in folders:
            directory = os.path.join(self.OUTPUT_FOLDER, folder)
            if not os.path.exists(directory):
                os.makedirs(directory)

        self.LIDAR_PATH = os.path.join(self.OUTPUT_FOLDER, 'velodyne/{0:06}.bin')
        self.KITTI_LABEL_PATH = os.path.join(self.OUTPUT_FOLDER, 'label/{0:06}.txt')
        self.CARLA_LABEL_PATH = os.path.join(self.OUTPUT_FOLDER, 'carla_label/{0:06}.txt')
        self.IMAGE_PATH = os.path.join(self.OUTPUT_FOLDER, 'image/{0:06}.png')
        self.IMAGE_SEG_PATH = os.path.join(self.OUTPUT_FOLDER, 'image_seg/{0:06}.png')
        self.CALIBRATION_PATH = os.path.join(self.OUTPUT_FOLDER, 'calib/{0:06}.txt')
        self.IMU_PATH = os.path.join(self.OUTPUT_FOLDER, 'imu/{0:06}.txt')
        self.GNSS_PATH = os.path.join(self.OUTPUT_FOLDER, 'gnss/{0:06}.txt')

    def _process_segmantic(self, image):
        image.convert(carla.ColorConverter.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        # [H,W,4] --> [H,W,3]
        # array = array[:, :, :3]
        # [0,1] --> [0,255]
        # array = array * 255
        # # Get the r channel
        sem = array[:, :, 2]
        return sem

    def _generate_segmantic(self, sem, labels=[]):
        canvas = np.zeros(sem.shape + (3,), dtype=np.uint8)
        # print("shape of canvas:",canvas.shape)
        for label in labels:
            # print(label)
            canvas[sem == label] = SEM_COLORS[label]
        return canvas

    def _current_captured_frame_num(self):
        """获取文件夹中存在的数据量"""
        label_path = os.path.join(self.OUTPUT_FOLDER, 'label/')
        num_existing_data_files = len(
            [name for name in os.listdir(label_path) if name.endswith('.txt')])
        print("当前存在{}个数据".format(num_existing_data_files))
        if num_existing_data_files == 0:
            return 0
        answer = input(
            "There already exists a dataset in {}. Would you like to (O)verwrite or (A)ppend the dataset? (O/A)".format(
                self.OUTPUT_FOLDER))
        if answer.upper() == "O":
            logging.info(
                "Resetting frame number to 0 and overwriting existing")
            return 0
        logging.info("Continuing recording data on frame number {}".format(
            num_existing_data_files))
        return num_existing_data_files

    def save_training_files(self, data):
        lidar_fname = self.LIDAR_PATH.format(self.captured_frame_no)
        kitti_label_fname = self.KITTI_LABEL_PATH.format(self.captured_frame_no)
        carla_label_fname = self.CARLA_LABEL_PATH.format(self.captured_frame_no)
        img_fname = self.IMAGE_PATH.format(self.captured_frame_no)
        img_seg_fname = self.IMAGE_SEG_PATH.format(self.captured_frame_no)
        calib_filename = self.CALIBRATION_PATH.format(self.captured_frame_no)
        imu_filename = self.IMU_PATH.format(self.captured_frame_no)
        gnss_filename = self.GNSS_PATH.format(self.captured_frame_no)

        for agent, dt in data["agents_data"].items():
            ego_pose = dt["ego_pose"]
            camera_transform = config_to_trans(self.cfg["SENSOR_CONFIG"]["RGB"]["TRANSFORM"])
            lidar_transform = config_to_trans(self.cfg["SENSOR_CONFIG"]["LIDAR"]["TRANSFORM"])

            save_ref_files(self.OUTPUT_FOLDER, self.captured_frame_no)
            save_image_data(img_fname, dt["sensor_data"][0])
            save_label_data(kitti_label_fname, dt["kitti_datapoints"])
            # save_label_data(carla_label_fname, dt['carla_datapoints'])
            save_calibration_matrices([camera_transform, lidar_transform], calib_filename, dt["intrinsic"])
            img_seg_index = self._process_segmantic(dt["sensor_data"][2])
            img_seg_fname_canvas_line = self._generate_segmantic(img_seg_index, labels=[6, 7])
            save_seg_image_data(img_seg_fname, img_seg_fname_canvas_line)
            save_imu_data(imu_filename, dt["sensor_data"][3])
            save_gnss_data(gnss_filename, dt["sensor_data"][4], ego_pose)
            save_lidar_data(lidar_fname, dt["sensor_data"][5])
        self.captured_frame_no += 1
