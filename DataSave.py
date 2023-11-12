from config import config_to_trans
from export_utils import *
import numpy as np
import carla
from PIL import Image
import sys
sys.path.append("..")
from Fisheye.fisheye_model import cameraModel

SEM_COLORS = {

    6: (157, 234, 50), # ROAD LINE
    7: (128, 64, 128), # ROAD

}


class DataSave:
    def __init__(self, cfg):
        self.cfg = cfg
        self.OUTPUT_FOLDER = None
        self.LIDAR_PATH = None
        self.KITTI_LABEL_PATH = None
        self.CARLA_LABEL_PATH = None
        self.IMAGE_RAW_PATH = None
        self.IMAGE_SEG_PATH = None
        self.IMAGE_SR_PATH = None
        self.IMAGE_SL_PATH = None
        self.IMAGE_FRONT_PATH = None
        self.IMAGE_RIGHT_PATH = None
        self.IMAGE_LEFT_PATH = None
        self.IMAGE_REAR_PATH = None
        self.CALIBRATION_PATH = None
        self._generate_path(self.cfg["SAVE_CONFIG"]["ROOT_PATH"])
        self.captured_frame_no = self._current_captured_frame_num()

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

    def _generate_path(self,root_path):
        """ 生成数据存储的路径"""
        PHASE = "training"
        self.OUTPUT_FOLDER = os.path.join(root_path, PHASE)
        # folders = ['calib', 'image_2','image_seg', 'image_stereoright', 'image_stereoleft', 'image_2','image_right','image_left','image_rear', 'kitti_label', 'carla_label', 'velodyne']
        folders = ['calib', 'image_2', 'image_seg', 'image_stereoright', 'image_stereoleft', 'kitti_label', 'carla_label', 'velodyne']

        for folder in folders:
            directory = os.path.join(self.OUTPUT_FOLDER, folder)
            if not os.path.exists(directory):
                os.makedirs(directory)

        self.LIDAR_PATH = os.path.join(self.OUTPUT_FOLDER, 'velodyne/{0:06}.bin')
        self.KITTI_LABEL_PATH = os.path.join(self.OUTPUT_FOLDER, 'kitti_label/{0:06}.txt')
        self.CARLA_LABEL_PATH = os.path.join(self.OUTPUT_FOLDER, 'carla_label/{0:06}.txt')
        self.IMAGE_RAW_PATH = os.path.join(self.OUTPUT_FOLDER, 'image_2/{0:06}.png')
        self.IMAGE_SEG_PATH = os.path.join(self.OUTPUT_FOLDER, 'image_seg/{0:06}.png')
        self.IMAGE_SR_PATH = os.path.join(self.OUTPUT_FOLDER, 'image_stereoright/{0:06}.png')
        self.IMAGE_SL_PATH = os.path.join(self.OUTPUT_FOLDER, 'image_stereoleft/{0:06}.png')
        # self.IMAGE_FRONT_PATH = os.path.join(self.OUTPUT_FOLDER, 'image_2/{0:06}.png')
        # self.IMAGE_RIGHT_PATH = os.path.join(self.OUTPUT_FOLDER, 'image_right/{0:06}.png')
        # self.IMAGE_LEFT_PATH = os.path.join(self.OUTPUT_FOLDER, 'image_left/{0:06}.png')
        # self.IMAGE_REAR_PATH = os.path.join(self.OUTPUT_FOLDER, 'image_rear/{0:06}.png')
        self.CALIBRATION_PATH = os.path.join(self.OUTPUT_FOLDER, 'calib/{0:06}.txt')


    def _current_captured_frame_num(self):
        """获取文件夹中存在的数据量"""
        label_path = os.path.join(self.OUTPUT_FOLDER, 'kitti_label/')
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
        img_raw_fname = self.IMAGE_RAW_PATH.format(self.captured_frame_no)
        img_seg_fname = self.IMAGE_SEG_PATH.format(self.captured_frame_no)
        img_sr_fname = self.IMAGE_SR_PATH.format(self.captured_frame_no)
        img_sl_fname = self.IMAGE_SL_PATH.format(self.captured_frame_no)
        # img_front_fname = self.IMAGE_FRONT_PATH.format(self.captured_frame_no)
        # img_right_fname = self.IMAGE_RIGHT_PATH.format(self.captured_frame_no)
        # img_left_fname = self.IMAGE_LEFT_PATH.format(self.captured_frame_no)
        # img_rear_fname = self.IMAGE_REAR_PATH.format(self.captured_frame_no)
        calib_filename = self.CALIBRATION_PATH.format(self.captured_frame_no)
        print("################ frame number ##################")
        print("                ", self.captured_frame_no)
        for agent, dt in data["agents_data"].items():

            kittiLabelSat = check_label_data(kitti_label_fname, dt["kitti_datapoints"])
            if not kittiLabelSat:
                print("no label info is recorded, end this loop !")
                self.captured_frame_no -= 1
                continue

            camera_raw_transform = config_to_trans(self.cfg["SENSOR_CONFIG"]["RGB_Raw"]["TRANSFORM"])
            camera_seg_transform = config_to_trans(self.cfg["SENSOR_CONFIG"]["RGB_Seg"]["TRANSFORM"])
            camera_sr_transform= config_to_trans(self.cfg["SENSOR_CONFIG"]["RGB_StereoRight"]["TRANSFORM"])
            camera_sl_transform = config_to_trans(self.cfg["SENSOR_CONFIG"]["RGB_StereoLeft"]["TRANSFORM"])
            lidar_transform = config_to_trans(self.cfg["SENSOR_CONFIG"]["LIDAR"]["TRANSFORM"])
            # fisheye_intrinsic = config_to_trans(self.cfg["FISHEYE_CONFIG"]["FISHEYEMODEL"]["INTRINSIC"])
            # fisheye_distort = config_to_trans(self.cfg["FISHEYE_CONFIG"]["FISHEYEMODEL"]["DISTORT"])

            save_ref_files(self.OUTPUT_FOLDER, self.captured_frame_no)

            # generate segmentation image
            save_image_data(img_sr_fname, dt["sensor_data"][0])
            save_image_data(img_sl_fname, dt["sensor_data"][1])

            save_image_data(img_raw_fname, dt["sensor_data"][2])

            img_seg_index = self._process_segmantic(dt["sensor_data"][3])
            img_seg_fname_canvas_line = self._generate_segmantic(img_seg_index, labels=[6])
            # img_seg_fname_canvas_road = self._generate_segmantic(img_seg_index, labels=[7])
            save_seg_image_data(img_seg_fname, img_seg_fname_canvas_line)
            # save_seg_image_data(img_seg_fname, img_seg_fname_canvas_road)



            # convert standard image into fisheye image
            # for i in range(2,5):
            #     dt["sensor_data"][i] = cameraModel.Fisheye(dt["sensor_data"][i],fisheye_intrinsic,fisheye_distort)

            # save_image_data(img_front_fname, dt["sensor_data"][4])
            # save_image_data(img_right_fname, dt["sensor_data"][5])
            # save_image_data(img_left_fname, dt["sensor_data"][6])
            # save_image_data(img_rear_fname, dt["sensor_data"][7])
            save_label_data(kitti_label_fname, dt["kitti_datapoints"])
            save_label_data(carla_label_fname, dt['carla_datapoints'])

            save_calibration_matrices(
                [camera_raw_transform, camera_seg_transform,camera_sr_transform,  camera_sl_transform, lidar_transform], calib_filename, dt["intrinsic"])
            # save_calibration_matrices(
            #     [camera_raw_transform, camera_seg_transform, camera_front_transform, camera_right_transform, camera_left_transform,
            #      camera_rear_transform, camera_sr_transform, camera_sl_transform, lidar_transform], calib_filename, dt["intrinsic"])
            save_lidar_data(lidar_fname, dt["sensor_data"][8])
        self.captured_frame_no += 1