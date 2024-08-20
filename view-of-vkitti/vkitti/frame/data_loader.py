import os
import numpy as np
from typing import Optional, List
from matplotlib import pyplot as plt
import logging

from vkitti.configuration import KittiLocations


class FrameDataLoader:
    """
    This class is responsible for loading any possible data from the dataset for a single specific frame.
    """
    def __init__(self,
                 kitti_locations: KittiLocations,
                 frame_number: str):
        """
Constructor which creates the backing fields for the properties which can load and store data from the dataset
upon request
        :param kitti_locations: KittiLocations object.
        :param frame_number: Specific frame number for which the data should be loaded.
        """

        # Assigning parameters
        self.kitti_locations: KittiLocations = kitti_locations
        self.frame_number: str = frame_number

        # Getting filed id from frame number
        self.file_id: str = str(self.frame_number).zfill(6)

        # Creating properties for possible data.
        # Data is only loaded upon request, then stored for future use.
        self._image: Optional[np.ndarray] = None
        self._image_seg: Optional[np.ndarray] = None
        self._lidar_data: Optional[np.ndarray] = None
        self._raw_labels: Optional[np.ndarray] = None

    @property
    def image(self):
        """
Image information property in RGB format.
        :return: RGB image.
        """
        if self._image is not None:
            # When the data is already loaded.
            return self._image
        else:
            # Load data if it is not loaded yet.
            self._image = self.get_image()
            return self._image

    @property
    def image_seg(self):
            """
    Image information property in RGB format.
            :return: RGB image.
            """
            if self._image_seg is not None:
                # When the data is already loaded.
                return self._image_seg
            else:
                # Load data if it is not loaded yet.
                self._image_seg = self.get_image_seg()
                return self._image_seg


    @property
    def lidar_data(self):
        """
Ego-motion compensated 360 degree lidar data of a Nx4 array including x,y,z,reflectance values.
        :return: Lidar data.
        """
        if self._lidar_data is not None:
            # When the data is already loaded.
            return self._lidar_data
        else:
            # Load data if it is not loaded yet.
            self._lidar_data = self.get_lidar_scan()
            return self._lidar_data

    @property
    def raw_labels(self):
        """
The labels include the ground truth data for the frame in kitti format including:

* Class: Describes the type of object: 'Car', 'Pedestrian', 'Cyclist', etc.
* Truncated: Not used, only there to be compatible with KITTI format.
* Occluded: Integer (0,1,2) indicating occlusion state 0 = fully visible, 1 = partly occluded 2 = largely occluded.
* Alpha: Observation angle of object, ranging [-pi..pi]
* Bbox: 2D bounding box of object in the image (0-based index) contains left, top, right, bottom pixel coordinates.
* Dimensions: 3D object dimensions: height, width, length (in meters)
* Location: 3D object location x,y,z in camera coordinates (in meters)
* Rotation: Rotation around -Z axis of the LiDAR sensor [-pi..pi]
        :return: Label data in string format
        """
        if self._raw_labels is not None:
            # When the data is already loaded.
            return self._raw_labels
        else:
            # Load data if it is not loaded yet.
            self._raw_labels = self.get_labels()
            return self._raw_labels

    def get_image(self) -> Optional[np.ndarray]:
        """
This method obtains the image information from the location specified by the KittiLocations object. If the file
does not exist, it returns None.

        :return: Numpy array with image data.
        """
        try:
            img = plt.imread(
                os.path.join(self.kitti_locations.camera_dir, f'{self.frame_number}.png'))

        except FileNotFoundError:
            logging.error(f"{self.frame_number}.jpg does not exist at location: {self.kitti_locations.camera_dir}!")
            return None

        return img

    def get_image_seg(self) -> Optional[np.ndarray]:
        """
    This method obtains the image information from the location specified by the KittiLocations object. If the file
    does not exist, it returns None.
        :return: Numpy array with image data.
        """
        try:
            img = plt.imread(
                os.path.join(self.kitti_locations.camera_seg_dir, f'{self.frame_number}.png'))

        except FileNotFoundError:
            logging.error(f"{self.frame_number}.jpg does not exist at location: {self.kitti_locations.camera_seg_dir}!")
            return None

        return img

    def get_lidar_scan(self) -> Optional[np.ndarray]:
        """
This method obtains the lidar information from the location specified by the KittiLocations object. If the file
does not exist, it returns None.

        :return: Numpy array with lidar data.
        """

        try:
            lidar_file = os.path.join(self.kitti_locations.lidar_dir, f'{self.file_id}.bin')
            scan = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)

        except FileNotFoundError:
            logging.error(f"{self.file_id}.bin does not exist at location: {self.kitti_locations.lidar_dir}!")
            return None

        return scan

    def get_labels(self) -> Optional[List[str]]:
        """
This method obtains the label information from the location specified by the KittiLocations object. If the file
does not exist, it returns None.

        :return: List of strings with label data.
        """

        try:
            label_file = os.path.join(self.kitti_locations.label_dir, f'{self.file_id}.txt')
            with open(label_file, 'r') as text:
                labels = text.readlines()

        except FileNotFoundError:
            logging.error(f"{self.file_id}.txt does not exist at location: {self.kitti_locations.label_dir}!")
            return None

        return labels

