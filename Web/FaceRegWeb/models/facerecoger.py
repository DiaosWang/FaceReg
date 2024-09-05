# -*- coding: utf-8 -*-
"""
输入：aligned cropped face
输出：face feature
注意： 提供计算两个feature的相似度的函数
"""

import cv2
import onnxruntime as ort 
import numpy as np

# 设置ONNX Runtime的日志级别为ERROR
ort.set_default_logger_severity(3)  # 3表示ERROR级别

class FaceRecoger():
    def __init__(self, onnx_path, num_threads=1) -> None:
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = num_threads
        # 初始化 InferenceSession 时传入 SessionOptions 对象
        self.ort_session = ort.InferenceSession(onnx_path, session_options=session_options)
        
        output_node_dims = [out.shape for out in self.ort_session.get_outputs()]
        self.len_feat = output_node_dims[0][1] # feature 的长度为...

    def inference(self, crop_img):   # crop_img = cv2.imread(img_path) bgr
        input_feed = {}
        if crop_img.shape[:2] != (248,248):   # 这里还有另一种方式 ,[4:252, 4:252,...] 
            crop_img = cv2.resize(crop_img,(248,248))
        crop_img = crop_img[...,::-1]
        input_data = crop_img.transpose((2, 0, 1))
        input_feed['_input_123'] = input_data.reshape((1, 3, 248, 248)).astype(np.float32)
        pred_result = self.ort_session.run([], input_feed=input_feed)
        temp_result = np.sqrt(pred_result[0])
        norm = temp_result / np.linalg.norm(temp_result, axis=1)
        return norm.flatten()   # return normalize  feature

    @staticmethod
    def compute_sim(feat1,feat2):
        feat1, feat2 = feat1.flatten(), feat2.flatten()
        assert feat1.shape == feat2.shape
        sim = np.sum(feat1 * feat2)
        return sim 
        

if __name__ == "__main__":
    fr = FaceRecoger(onnx_path = "./checkpoints/face_recognizer.onnx", num_threads=1) 
    import sys  
    imgpath1 = sys.argv[1]
    imgpath2 = sys.argv[2]
    img1,img2 = cv2.imread(imgpath1),cv2.imread(imgpath2)
    feat1 = fr.inference(img1)
    feat2 = fr.inference(img2)

    print("sim: ", FaceRecoger.compute_sim(feat1, feat2))
