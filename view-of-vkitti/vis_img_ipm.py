import os
import cv2
import yaml
from math import sin, cos
import numpy as np


def calRotationMat(alpha, theta):
    _alpha = alpha / 180 * np.pi
    _theta = theta / 180 * np.pi

    Ra = np.array([cos(_alpha), -sin(_alpha), 0,
                    sin(_alpha), cos(_alpha), 0,
                    0, 0, 1]).reshape(3, 3)

    Rt = np.array([1, 0, 0,
        0, cos(_theta), -sin(_theta),
        0, sin(_theta), cos(_theta)]).reshape(3, 3)
    _R_c_cg = Ra @ Rt  # virtual camera
    _n = _R_c_cg * np.array([0, -1, 0])
    return _R_c_cg


if __name__ == "__main__":
    cam_file = r'E:\Dataset\VKITTI\config\config.yaml'
    with open(cam_file, 'r') as file:
        config = yaml.safe_load(file)
        file.close()

    fx = config['projection_parameters']['fx']
    fy = config['projection_parameters']['fy']
    cx = config['projection_parameters']['cx']
    cy = config['projection_parameters']['cy']

    cg = [config['cg_alpha'], config['cg_theta'], config['cg_h']]
    ipm = [config['IPM_WIDTH'], config['IPM_HEIGHT'], config['IPM_RESO']]

    theta = np.deg2rad(90)
    R = np.array([[cos(theta), -sin(theta), 0],
                  [sin(theta), cos(theta), 0],
                  [0, 0, 1]])

    # 定义平移向量
    t = np.array([0.0, 0.0, 0.0])  # 假设tx, ty, tz是已知的平移分量

    # 构造变换矩阵（齐次变换矩阵）
    T = np.hstack((R, t.reshape(3, 1)))
    T = np.vstack((T, [0, 0, 0, 1]))

    K = np.array([fx, 0, cx,
                  0, fy, cy,
                  0, 0, 1], dtype=np.float32).reshape(3, 3)

    Rc0c1 = np.array(calRotationMat(config['cg_alpha'], config['cg_theta']), dtype=np.float32)
    K_IPM_inv = np.array([ipm[2] / cg[2], 0, (-ipm[0] / 2) * ipm[2] / cg[2],
                 0, 0, 1,
                 0, -ipm[2] / cg[2], (ipm[1] - 1) * ipm[2] / cg[2]]).reshape(3, 3)

    IPM_homography = np.linalg.inv(cg[2] * K @ Rc0c1 @ K_IPM_inv)

    img_dir = r'E:\Dataset\VKITTI\training\image'
    img_list = os.listdir(img_dir)
    w = 400
    h = 800
    idx_thresh = 300

    for idx, img in enumerate(img_list):
        if idx < idx_thresh:
            continue
        print("processing img : ", str(idx).zfill(6) + '.png')
        img_abs = os.path.join(img_dir, img)
        img_src = cv2.imread(img_abs)
        img_ipm = cv2.warpPerspective(img_src, IPM_homography, (w, h))

        # center = (w / 2, h / 2)  # 绕图片中心进行旋转
        # angle = -180  # 旋转方向取（-180，180）中的随机整数值，负为逆时针，正为顺势针
        # scale = 1.0  # 将图像缩放为80%
        #
        # M = cv2.getRotationMatrix2D(center, angle, scale)
        # img_ipm = cv2.warpAffine(src=img_ipm, M=M, dsize=(w, h), borderValue=(255, 255, 255))

        cv2.imshow("ipm_transform", img_ipm)
        cv2.waitKey(0)
        cv2.destroyAllWindows()