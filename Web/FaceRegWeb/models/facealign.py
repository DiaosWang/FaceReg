"""
这部分输入face5点位置和原图
输出aligned cropped face 
"""

import numpy as np
import cv2
import time
import os 

# image_data: src image   
# image_width, image_height,image_channels: width and height, channels of  src image
# src_x, src_y: 输出image每个像素对应的src image 中的像素位置.
def sampling(image_data, image_width, image_height, image_channels, src_x, src_y):
    ux = np.floor(src_x).astype(int)
    uy = np.floor(src_y).astype(int)
    
    # 创建一个与src_x形状相同的空数组，用于存储最终的像素值
    pixel = np.zeros((*src_x.shape, image_channels), dtype=np.uint8)

    # 创建一个掩码数组，标记有效的采样点
    valid_mask = (ux >= 0) & (ux < image_height - 1) & (uy >= 0) & (uy < image_width - 1)

    # 计算插值
    x = src_x - ux
    y = src_y - uy

    # 提取图像数据的各个通道
    image_data_reshape = image_data.reshape(-1, image_channels)  # (height * width, channels)
    ux_uy = ux * image_width + uy  # (height * width)
    ux_uy_next = ux_uy + 1
    ux_next = (ux + 1) * image_width + uy
    ux_next_next = ux_next + 1

    ux_uy[~valid_mask] = 0
    ux_uy_next[~valid_mask] = 0
    ux_next[~valid_mask] = 0
    ux_next_next[~valid_mask] = 0

    # 使用广播计算各个通道的插值
    top_left = image_data_reshape[ux_uy]
    top_right = image_data_reshape[ux_uy_next]
    bottom_left = image_data_reshape[ux_next]
    bottom_right = image_data_reshape[ux_next_next]

    # 计算插值
    interpolated_top = (1 - y[:, :, np.newaxis]) * top_left + y[:, :, np.newaxis] * top_right
    interpolated_bottom = (1 - y[:, :, np.newaxis]) * bottom_left + y[:, :, np.newaxis] * bottom_right
    interpolated_pixel = (1 - x[:, :, np.newaxis]) * interpolated_top + x[:, :, np.newaxis] * interpolated_bottom

    # 填充最终的像素值
    pixel[valid_mask] = np.clip(interpolated_pixel[valid_mask], 0, 255).astype(np.uint8)
    
    return pixel

def spatial_transform(image_data, image_width, image_height, image_channels,
                      crop_data, crop_width, crop_height, transformation,
                      pad_top=0, pad_bottom=0, pad_left=0, pad_right=0,
                      type='LINEAR', dtype='ZERO_PADDING', N=1):
    channels = image_channels
    dst_h = crop_height + pad_top + pad_bottom
    dst_w = crop_width + pad_left + pad_right

    for n in range(N):
        theta_data = transformation.reshape(-1)
        scale = np.sqrt(theta_data[0] ** 2 + theta_data[3] ** 2)
        
        bx, by = np.meshgrid(np.arange(dst_w) - pad_left, np.arange(dst_h) - pad_top)
        bx = bx.T
        by = by.T
        src_y = theta_data[0] * by + theta_data[1] * bx + theta_data[2]
        src_x = theta_data[3] * by + theta_data[4] * bx + theta_data[5]

        crop_data[:] = sampling(image_data, image_width, image_height, image_channels, src_x, src_y,)

    return True

def transformation_maker(crop_width, crop_height, points, mean_shape, mean_shape_width, mean_shape_height):
    points_num = len(points)  # point 个数 5 
    std_points = np.zeros((points_num, 2), dtype=np.float32)  # 标准点

    # 生成标准点的坐标
    for i in range(points_num):
        std_points[i, 0] = mean_shape[i * 2] * crop_width / mean_shape_width
        std_points[i, 1] = mean_shape[i * 2 + 1] * crop_height / mean_shape_height

    feat_points = np.array(points, dtype=np.float32).reshape(points_num, 2)
    
    # 初始化
    sum_x = 0.0
    sum_y = 0.0
    sum_u = 0.0
    sum_v = 0.0
    sum_xx_yy = 0.0
    sum_ux_vy = 0.0
    sum_vx_uy = 0.0

    for c in range(points_num):
        sum_x += std_points[c, 0]
        sum_y += std_points[c, 1]
        sum_u += feat_points[c, 0]
        sum_v += feat_points[c, 1]
        sum_xx_yy += std_points[c, 0] ** 2 + std_points[c, 1] ** 2
        sum_ux_vy += std_points[c, 0] * feat_points[c, 0] + std_points[c, 1] * feat_points[c, 1]
        sum_vx_uy += feat_points[c, 1] * std_points[c, 0] - feat_points[c, 0] * std_points[c, 1]

    if sum_xx_yy <= np.finfo(np.float32).eps:
        return False, None

    q = sum_u - sum_x * sum_ux_vy / sum_xx_yy + sum_y * sum_vx_uy / sum_xx_yy
    p = sum_v - sum_y * sum_ux_vy / sum_xx_yy - sum_x * sum_vx_uy / sum_xx_yy
    r = points_num - (sum_x ** 2 + sum_y ** 2) / sum_xx_yy

    if np.abs(r) <= np.finfo(np.float32).eps:
        return False, None

    a = (sum_ux_vy - sum_x * q / r - sum_y * p / r) / sum_xx_yy
    b = (sum_vx_uy + sum_y * q / r - sum_x * p / r) / sum_xx_yy
    c = q / r
    d = p / r

    transformation = np.zeros((2, 3), dtype=np.float64)
    transformation[0, 0] = transformation[1, 1] = a
    transformation[0, 1] = -b
    transformation[1, 0] = b
    transformation[0, 2] = c
    transformation[1, 2] = d

    return True, transformation


class FaceAlign:
    def __init__(self) -> None:
        self.crop_width, self.crop_height = 256, 256
        self.mean_shape_width, self.mean_shape_height = 256, 256
        self.mean_face = [ # 标准人脸的特征点的位置
            89.3095, 72.9025,
            169.3095, 72.9025,
            127.8949, 127.0441,
            96.8796, 184.8907,
            159.1065, 184.7601
        ]


    # landmarks5 = [  
    #     [268.99814285714285, 166.26619999999997], 
    #     [342.636625, 164.43359999999998], 
    #     [311.5448214285714, 221.24419999999998], 
    #     [272.2709642857143, 243.23539999999997], 
    #     [344.2730357142857, 241.40279999999996]
    # ]
    def align(self, image, landmarks5):  # 原图image  landmarks5
        success, transformation = transformation_maker(self.crop_width, self.crop_height, landmarks5, self.mean_face, self.mean_shape_width, self.mean_shape_height)
        if not success:
            print("Failed to compute transformation matrix.")
        
        img_height, img_width, img_channels = image.shape

        crop_data = np.zeros((self.crop_height, self.crop_width, 3), dtype=np.uint8)
        success = spatial_transform(image,  img_width, img_height, img_channels, 
                                    crop_data, self.crop_width, self.crop_height,
                                    transformation,
                                    )
        if success:
            if os.path.exists("./images/result1.jpg"):
                cv2.imwrite("./images/result2.jpg", crop_data, [cv2.IMWRITE_JPEG_QUALITY, 100])
            else: 
                cv2.imwrite("./images/result1.jpg", crop_data, [cv2.IMWRITE_JPEG_QUALITY, 100])
        else: 
            print("error when spatial_transform...")

        return crop_data

if __name__ == "__main__":
    fa = FaceAlign()
    landmarks5 = [(240.56920098163752, 111.91879640513824), 
                  (283.7146242409017, 93.30582481805237), 
                  (268.9820406889578, 129.202270021718), 
                  (259.51109411985107, 155.79222943184064), 
                  (296.34255299971073, 137.17925784475477)]
    landmarks5 = [ [ld5[0],ld5[1]] for ld5 in landmarks5]
    image = cv2.imread("/home/bns/seetaface6Python/seetaFace6Python/asserts/1.jpg")
    fa.align(image = image, landmarks5=landmarks5)

