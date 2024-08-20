import os


class KittiLocations:
    """
    This class contains the information regarding the locations of data for the dataset.
    """
    def __init__(self, is_train: bool, root_dir: str, output_dir: str = None, frame_set_path: str = None, pred_dir: str = None):
        """
Constructor which based on a few parameters defines the locations of possible data.
        :param root_dir: The root directory of the dataset.
        :param output_dir: Optional parameter of the location where output such as pictures should be generated.
        :param frame_set_path: Optional parameter of the text file of which output should be generated.
        :param pred_dir: Optional parameter of the locations of the prediction labels.
        """

        # Input parameters
        self.is_train: bool = is_train
        self.root_dir: str = root_dir
        self.output_dir: str = output_dir
        self.frame_set_path: str = frame_set_path

        # Automatically defined variables. The location of sub-folders can be customized here.
        # Current definitions are based on the recommended locations.
        sub_path = 'training'
        if not is_train:
            sub_path = 'testing'

        self.camera_dir = os.path.join(self.root_dir, sub_path, 'image')
        self.camera_seg_dir = os.path.join(self.root_dir, sub_path, 'image_seg')
        self.lidar_dir = os.path.join(self.root_dir, sub_path, 'velodyne')
        self.calib_dir = os.path.join(self.root_dir, sub_path, 'calib')
        self.gnss_dir = os.path.join(self.root_dir, sub_path, 'gnss')
        self.imu_dir = os.path.join(self.root_dir, sub_path, 'imu')
        self.label_dir = os.path.join(self.root_dir, sub_path, 'label')
