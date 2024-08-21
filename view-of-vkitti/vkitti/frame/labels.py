from typing import Optional, List


class FrameLabels:
    """
    This class is responsible for converting the label string list to a list of Python dictionaries.
    """

    def __init__(self,
                 raw_labels: List[str]):
        """
Constructor which creates the label property, given a list of strings containing the label data.
        :param raw_labels: List of strings containing label data.
        """
        self.raw_labels: List[str] = raw_labels

        self._labels_dict: Optional[List[dict]] = None

    @property
    def labels_dict(self):
        """
Label dictionary property.
        :return:
        """
        if self._labels_dict is not None:
            # When the data is already loaded.
            return self._labels_dict
        else:
            # Load data if it is not loaded yet.
            self._labels_dict = self.get_labels_dict()
            return self._labels_dict

    def get_labels_dict(self) -> List[dict]:
        """
This method returns a list of dictionaries containing the label data.
        :return: List of dictionaries containing label data.
        """

        labels = []  # List to be filled

        for act_line in self.raw_labels:  # Go line by line to split the keys
            act_line = act_line.split()
            """
            'Car', : type,
            '0', : truncated
            '2', : occluded
            '-2.486604605401256', : alpha
            '1919.0', '572.9676046174333', '1919.0', '703.078661586053', : bbox
            '1.7', '2.1', '5.0', : dimensions
            '20.81963268025056', '1.2099563877874295', '15.103961474799664', : locations
            '-3.089430081654608', : rotation_y
            '1', : score
            '853' :track_id
            """
            type, truncated, occluded, alpha, bbox_x1, bbox_y1, bbox_x2, bbox_y2, h, w, l, x, y, z, rot, score, track_id = act_line
            h, w, l, x, y, z, rot, score = map(float, [h, w, l, x, y, z, rot, score])
            bbox_x1, bbox_y1, bbox_x2, bbox_y2 = map(int, [bbox_x1, bbox_y1, bbox_x2, bbox_y2])
            if rot > 2:
                rot = -rot

            labels.append({'label_class': type,
                           'bbox_xmin': bbox_x1,
                           'bbox_ymin': bbox_y1,
                           'bbox_xmax': bbox_x2,
                           'bbox_ymax': bbox_y2,
                           'h': h,
                           'w': w,
                           'l': l,
                           'x': x,
                           'y': y,
                           'z': z,
                           'rotation': rot,
                           'score': score}
                          )
        return labels


