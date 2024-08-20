import os
import json
import math
import numpy as np
import matplotlib.pyplot as plt


def read_json(json_file):
    result_dict = {}
    result_list = []
    with open(json_file, 'r') as file:
        for line in file:
            try:
                data = json.loads(line.strip())
                if "odomToCamera" in data:
                    result_dict["odomToCamera"] = data["odomToCamera"]
                    result_list = data["odomToCamera"]
            except json.JSONDecodeError as e:
                    print(f"error {e}")
    return result_dict, result_list


def extract_pose(json_info):
    r11, r12, r13, tx = json_info[0, :]
    r21, r22, r23, ty = json_info[1, :]
    r31, r32, r33, tz = json_info[2, :]
    rotation_matrix = json_info[:3, :3]
    # 先绕Z轴旋转（Yaw）
    cos_yaw = math.sqrt(r11 ** 2 + r21 ** 2)
    sin_yaw = math.sqrt(1 - cos_yaw ** 2)
    yaw = math.atan2(r21, r11) if cos_yaw > 0 else math.atan2(-r21, -r11) + math.pi

    # 去除Z轴旋转的影响
    rotation_y_z_removed = np.array([
        [r11 / cos_yaw, r12 / cos_yaw, r13 / cos_yaw],
        [r21 / cos_yaw, r22 / cos_yaw, r23 / cos_yaw],
        [r31 / cos_yaw, r32 / cos_yaw, r33 / cos_yaw]
    ])

    # 绕Y轴旋转（Pitch）
    cos_pitch = math.sqrt(rotation_y_z_removed[0, 0] ** 2 + rotation_y_z_removed[2, 0] ** 2)
    if(1 - cos_pitch ** 2) > 0.0:
        sin_pitch = math.sqrt(1 - cos_pitch ** 2)
    else:
        sin_pitch = 0
    pitch = math.atan2(-rotation_y_z_removed[1, 0], cos_pitch)

    # 绕X轴旋转（Roll）
    cos_roll = rotation_y_z_removed[2, 2]
    sin_roll = math.sqrt(1 - cos_roll ** 2)
    roll = math.atan2(rotation_y_z_removed[1, 2], rotation_y_z_removed[2, 2])

    return roll, pitch, yaw, tx, ty, tz


def cal_position(roll, pitch, yaw, tx, ty, tz, x, y, z):
    dt = 1 / 10
    Rx_rad_per_s, Ry_rad_per_s, Rz_rad_per_s = roll, pitch, yaw
    Tx_per_s, Ty_per_s, Tz_per_s = tx, ty, tz

    Rx = Rx_rad_per_s * dt
    Ry = Ry_rad_per_s * dt
    Rz = Rz_rad_per_s * dt

    # 计算当前时刻的平移距离
    Tx = Tx_per_s * dt
    Ty = Ty_per_s * dt
    Tz = Tz_per_s * dt

    # 更新位置
    x = x + Tx
    y = y + Ty
    z = z + Tz

    return x, y, z


def plot_path(path):
    imu_data_array = np.array(path)

    x = imu_data_array[:, 0]
    y = imu_data_array[:, 1]
    z = imu_data_array[:, 2]

    # 创建一个图形和坐标轴对象
    fig, ax = plt.subplots()

    # 绘制轨迹
    ax.plot(imu_data_array[:, 0], imu_data_array[:, 1], marker='o', linestyle='-', color='blue')

    # 创建一个3D坐标轴
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制点
    ax.scatter(x, y, z, c='red', marker='o')

    # 绘制连线
    # 注意：这里我们假设要连接所有的点，形成一个连续的线
    # 如果需要连接特定的点，你需要调整索引
    for i in range(len(x) - 1):
        ax.plot([x[i], x[i + 1]], [y[i], y[i + 1]], [z[i], z[i + 1]], color='blue')

    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 显示图形
    plt.show()


if __name__ == '__main__':
    json_dir = r'E:\VOD\vod\lidar\training\pose'
    json_list = os.listdir(json_dir)
    odomToCamera_list = []

    for json_file in json_list:
        json_abs = os.path.join(json_dir, json_file)
        json_dict, json_info = read_json(json_abs)
        if json_info:
            odomToCamera_list.append(np.array(json_info).reshape(4, 4))

    x, y, z = 0, 0, 0
    positions = []
    for json_info in odomToCamera_list:
        roll, pitch, yaw, tx, ty, tz = extract_pose(json_info)
        # 输出结果
        print(f"旋转角度（弧度）: {roll:.2f}, {pitch:.2f}, {yaw:.2f}")
        print(f"旋转角度（角度）: {math.degrees(roll):.2f}, {math.degrees(pitch):.2f}, {math.degrees(yaw):.2f}")
        print(f"平移距离 （m）: ({tx}, {ty}, {tz})")

        x, y, z = cal_position(roll, pitch, yaw, tx, ty, tz, x, y, z)
        positions.append([x, y, z])

    plot_path(positions)
