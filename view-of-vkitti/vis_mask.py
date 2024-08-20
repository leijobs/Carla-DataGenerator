import os

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def decode_image(image, n_class):
    # bits = np.ones(n_class).astype(int)  # 【修改项】
    bits = np.power(2, np.arange(n_class))  # 【修改项】
    out_array = (image & bits.reshape(-1, 1, 1)) > 0
    return out_array


def showColorImg(img_array):
    int_array = img_array.astype(int)
    plt.imshow(int_array, cmap='coolwarm', interpolation='nearest', vmin=0)  # , vmin=0, vmax=1
    plt.axis('off')
    plt.show()


def fov_mask():
    resolution = 0.25
    fu, cu = 1266.417203046554, 816.2670197447984
    image_width = 1600

    # Construct a grid of image coordinates
    x1, z1, x2, z2 = [-25, 1, 25, 50]
    x, z = np.arange(x1, x2, resolution), np.arange(z1, z2, resolution)
    ucoords = x / z[:, None] * fu + cu

    # Return all points which lie within the camera bounds
    ucoords = (ucoords >= 0) & (ucoords < image_width)
    ucoords2 = np.flip(ucoords, axis=0)
    ucoords3 = np.flip(ucoords2, axis=1)
    return ucoords3

def main():
    bev_dir = r'E:\DualRadar\view-of-dualradar\visualizeNuscenes\bev_maps'
    bev_imgs = os.listdir(bev_dir)
    for bev in bev_imgs:
        image = Image.open(os.path.join(bev_dir, bev))
        # 转换为整数数组以更清晰地显示（可选）
        image_np = np.array(image)
        int_array = image_np.astype(np.uint8)
        int_array += 1
        # showColorImg(int_array)

        masks = decode_image(image_np, 15)
        masks[-1] = ~masks[-1]
        # int_array *= masks[-1]
        # showColorImg(int_array)
        for i in range(int_array.shape[0]):
            for j in range(int_array.shape[1]):
                if int_array[i][j] == 34:
                    continue
                else:
                    int_array[i][j] *= masks[-1][i][j]

        int_array *= fov_mask()
        showColorImg(int_array)
        # for i in range(len(masks)):
        #     showColorImg(masks[i])


if __name__ == '__main__':
    main()