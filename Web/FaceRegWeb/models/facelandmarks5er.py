"""
输入：原图，图中face框
输出：每张face的5点特征点的位置
"""
# from common.face_landmark_points_util import shape_index_process
# from common.face_detector_util import prior_box_forward, decode, nms_sorted


import cv2
import onnxruntime as ort 
import numpy as np

# 设置ONNX Runtime的日志级别为ERROR
ort.set_default_logger_severity(3)  # 3表示ERROR级别
m_origin_patch = [15, 15]
m_origin = [112, 112]

class HypeShape:
    def __init__(self, shape):
        self.m_shape = shape
        self.m_weights = [0]*len(self.m_shape)
        size = len(self.m_shape)
        self.m_weights[size - 1] = self.m_shape[size - 1]
        for times in range(size - 1):
             self.m_weights[size - 1 - times - 1] =  self.m_weights[size - 1 - times] * self.m_shape[size - 1 - times - 1]

    def to_index(self, coordinate):
        if len(coordinate) == 0:
            return 0
        size = len(coordinate)
        weight_start = len(self.m_weights) - size + 1
        index = 0
        for times in range(size - 1):
            index += self.m_weights[weight_start + times] * coordinate[times]
        index += coordinate[size - 1]
        return index


def shape_index_process(feat_data, pos_data):
    feat_h = feat_data.shape[2]
    feat_w = feat_data.shape[3]

    landmarkx2 = pos_data.shape[1]
    x_patch_h = int( m_origin_patch[0] * feat_data.shape[2] / float( m_origin[0] ) + 0.5 )
    x_patch_w = int( m_origin_patch[1] * feat_data.shape[3] / float( m_origin[1] ) + 0.5 )

    feat_patch_h = x_patch_h
    feat_patch_w = x_patch_w

    num = feat_data.shape[0]
    channels = feat_data.shape[1]

    r_h = ( feat_patch_h - 1 ) / 2.0
    r_w = ( feat_patch_w - 1 ) / 2.0
    landmark_num = int(landmarkx2 * 0.5)

    pos_offset = HypeShape([pos_data.shape[0], pos_data.shape[1]])
    feat_offset = HypeShape([feat_data.shape[0], feat_data.shape[1], feat_data.shape[2], feat_data.shape[3]])
    nmarks = int( landmarkx2 * 0.5 )
    out_shape = [feat_data.shape[0], feat_data.shape[1], x_patch_h, nmarks, x_patch_w]
    out_offset = HypeShape([feat_data.shape[0], feat_data.shape[1], x_patch_h, nmarks, x_patch_w])
    buff = np.zeros(out_shape)
    zero = 0

    buff = buff.reshape((-1))
    pos_data = pos_data.reshape((-1))
    feat_data = feat_data.reshape((-1))

    for i in range(landmark_num):
        for n in range(num):
            # coordinate of the first patch pixel, scale to the feature map coordinate
            y = int( pos_data[pos_offset.to_index( [n, 2 * i + 1] )] * ( feat_h - 1 ) - r_h + 0.5 )
            x = int( pos_data[pos_offset.to_index( [n, 2 * i] )] * ( feat_w - 1 ) - r_w + 0.5 )

            for c in range(channels):
                for ph in range(feat_patch_h):
                    for pw in range(feat_patch_w):
                        y_p = y + ph
                        x_p = x + pw
                        # set zero if exceed the img bound
                        if y_p < 0 or y_p >= feat_h or x_p < 0 or x_p >= feat_w:
                            buff[out_offset.to_index( [n, c, ph, i, pw] )] = zero
                        else:
                            buff[out_offset.to_index( [n, c, ph, i, pw] )] = feat_data[feat_offset.to_index( [n, c, y_p, x_p] )]

    return buff.reshape((1,-1,1,1)).astype(np.float32)


def crop_face(image:np.ndarray, face, H, W):
    """
    Crop a face from an image with padding if the face is out of bounds.

    Args:
        image (np.ndarray): The input image as a numpy array of shape (H, W, C).
        face (tuple): A tuple containing (x, y, w, h) for the face rectangle.
        padding (list): Padding data for padding the image.
    
    Returns:
        np.ndarray: Cropped and padded image.
    """
    x0, y0, x1, y1 = int(round(face[0])), int(round(face[1])), int(round(face[2])), int(round(face[3]))
    
    # Calculate padding
    pad_top = max(0, -y0)
    pad_bottom = max(0, y1 - H)
    pad_left = max(0, -x0)
    pad_right = max(0, x1 - W)

    # Apply padding
    padded_image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant', constant_values=0)

    # Update new coordinates after padding
    new_x0, new_y0 = x0 + pad_left, y0 + pad_top
    new_x1, new_y1 = x1 + pad_left, y1 + pad_top
    
    # Crop the face
    cropped_image = padded_image[new_y0:new_y1, new_x0:new_x1, :]

    return cropped_image, (x0,y0,x1,y1)


class Landmark5er():
    def __init__(self, onnx_path1, onnx_path2, num_threads=1) -> None:
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = num_threads
        # 初始化 InferenceSession 时传入 SessionOptions 对象
        self.ort_session1 = ort.InferenceSession(onnx_path1, session_options=session_options)
        self.ort_session2 = ort.InferenceSession(onnx_path2, session_options=session_options)
        self.first_input_name = self.ort_session1.get_inputs()[0].name
        self.second_input_name = self.ort_session2.get_inputs()[0].name

    def inference(self, image, box):
        # face_img = image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]  # 这种裁剪不对,不合适
        H,W,C = image.shape
        face_img, box = crop_face(image,box,H,W)

        x1,y1,x2,y2 = int(box[0]) ,int(box[1]), int(box[2]),int(box[3])
        if x1 < 0 or x1 > W-1 or x2 < 0 or x2 > W: 
            print("x超出边界")
        if y1 < 0 or y1 > H-1 or y2 < 0 or y2 > H: 
            print("y超出边界")

        face_img = cv2.resize(face_img,(112,112))

        gray_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        gray_img = gray_img.reshape((1, 1, 112, 112)).astype(np.float32)  # 输入必须是(1,1,112,112)
        # points5 net1
        results_1 = self.ort_session1.run([], {self.first_input_name: gray_img})

        # shape index process
        feat_data = results_1[0]
        pos_data = results_1[1]
        shape_index_results = shape_index_process(feat_data, pos_data)
        results_2 = self.ort_session2.run([], {self.second_input_name: shape_index_results})

        landmarks = (results_2[0] + results_1[1]) * 112
        # print("results_2[0] , results_1[1]: ",results_2[0], results_1[1])
        # print("in find_landmarks, landmarks:", landmarks)
        landmarks = landmarks.reshape((-1)).astype(np.int32)

        scale_x = (box[2] - box[0]) / 112.0
        scale_y = (box[3] - box[1]) / 112.0
        mapped_landmarks = []
        for i in range(landmarks.size // 2):
            x = box[0] + landmarks[2 * i] * scale_x
            y = box[1] + landmarks[2 * i + 1] * scale_y
            x = max(0.01, min(x,W-0.01))
            y = max(0.01, min(y,H-0.01))
            mapped_landmarks.append((x, y))

        return mapped_landmarks


if __name__ == "__main__":
    import sys 
    ld5 = Landmark5er(onnx_path1="./checkpoints/face_landmarker_pts5_net1.onnx", 
                      onnx_path2="./checkpoints/face_landmarker_pts5_net2.onnx", 
                      num_threads=1)

    if len(sys.argv) > 1:
        jpg_path =  sys.argv[1]
    else: 
        jpg_path = "asserts/1.jpg"

    image = cv2.imread(jpg_path)
    if image is None:
        print("Error: Could not load image.")
        exit()
    
    box = (201.633087308643, 42.78490193881931, 319.49375572419393, 191.68867463550623) 
    landmarks5 = ld5.inference(image,box)
    print(landmarks5)
