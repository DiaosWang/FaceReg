import cv2
import numpy as np
import onnxruntime as ort


class QualityOfClarity:
    def __init__(self, low_thresh=0.10, high_thresh=0.20):
        self.low_thresh = low_thresh
        self.high_thresh = high_thresh
    
    def reblur(self, data, width, height):
        data = np.array(data, dtype=np.float32).reshape((height, width))
        
        # 创建一维核
        kernel = np.ones((9,), np.float32) / 9.0
        
        # 垂直方向模糊处理
        BVer = cv2.filter2D(data, -1, kernel.reshape(-1, 1), borderType=cv2.BORDER_REPLICATE)
        
        # 水平方向模糊处理
        BHor = cv2.filter2D(data, -1, kernel.reshape(1, -1), borderType=cv2.BORDER_REPLICATE)
        
        s_FVer, s_FHor, s_Vver, s_VHor = 0.0, 0.0, 0.0, 0.0
    
        # 计算垂直方向的差分
        D_Fver = np.abs(data[1:, :] - data[:-1, :])
        D_BVer = np.abs(BVer[1:, :] - BVer[:-1, :])

        # 计算垂直方向的累积
        s_FVer = np.sum(D_Fver)
        s_Vver = np.sum(np.maximum(0.0, D_Fver - D_BVer))

        # 计算水平方向的差分
        D_FHor = np.abs(data[:, 1:] - data[:, :-1])
        D_BHor = np.abs(BHor[:, 1:] - BHor[:, :-1])

        # 计算水平方向的累积
        s_FHor = np.sum(D_FHor)
        s_VHor = np.sum(np.maximum(0.0, D_FHor - D_BHor))

        
        b_FVer = (s_FVer - s_Vver) / s_FVer
        b_FHor = (s_FHor - s_VHor) / s_FHor
        blur_val = max(b_FVer, b_FHor)
        
        return blur_val

    def grid_max_reblur(self, img, rows, cols):
        height, width = img.shape
        row_height = height // rows
        col_width = width // cols
        blur_val = float('-inf')
        
        for y in range(rows):
            for x in range(cols):
                grid = img[y * row_height: (y + 1) * row_height, x * col_width: (x + 1) * col_width]
                this_grad_blur_val = self.reblur(grid, col_width, row_height)
                if this_grad_blur_val > blur_val:
                    blur_val = this_grad_blur_val
        
        return max(blur_val, 0.0)

    def clarity_estimate(self, image):
        # x, y, w, h = rect
        # if w < 9 or h < 9:
        #     return 0.0
        
        src_data = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # src_data = gray_data[y:y+h, x:x+w]
        blur_val = self.grid_max_reblur(src_data, 2, 2)
        clarity = 1.0 - blur_val

        T1, T2 = 0.0, 1.0
        if clarity <= T1:
            clarity = 0.0
        elif clarity >= T2:
            clarity = 1.0
        else:
            clarity = (clarity - T1) / (T2 - T1)

        return clarity

    def check(self, image):
        clarity = self.clarity_estimate(image)
        if clarity < self.low_thresh:
            level = "LOW"
        elif self.low_thresh <= clarity < self.high_thresh:
            level = "MEDIUM"
        else:
            level = "HIGH"
        return level != 'LOW'
        # return {'level': level, "score": clarity}


class QualityChecker:  # check resolution and clarity of image
    def __init__(self, v0=70.0, v1=100.0, v2=210.0, v3=230.0, hw = (112,112)):
        self.bright_thresh0 = v0
        self.bright_thresh1 = v1
        self.bright_thresh2 = v2
        self.bright_thresh3 = v3
        self.middle_thresh = (self.bright_thresh1 + self.bright_thresh2) / 2
        self.rolu_thrds = hw  # （h，w）

    def get_bright_score(self, bright):
        bright_score = 1.0 / (abs(bright - self.middle_thresh) + 1)
        return bright_score

    def grid_max_bright(self, img, rows, cols):
        row_height = img.shape[0] // rows
        col_width = img.shape[1] // cols

        # 使用列表生成式获取所有网格的平均亮度
        grid_means = [
            np.mean(img[y * row_height:(y + 1) * row_height, x * col_width:(x + 1) * col_width])
            for y in range(rows)
            for x in range(cols)
        ]

        # 获取最大亮度值
        bright_val = max(grid_means)
        return max(bright_val, 0)

    def check_bright(self, face_image):
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY) if len(face_image.shape) == 3 else face_image
        bright_value = self.grid_max_bright(gray, 3, 3)

        if bright_value < self.bright_thresh0 or bright_value > self.bright_thresh3:
            level = "LOW"
        elif (self.bright_thresh0 <= bright_value < self.bright_thresh1) or (self.bright_thresh2 < bright_value <= self.bright_thresh3):
            level = "MEDIUM"
        else:
            level = "HIGH"

        return level == "HIGH"
    
    def check_resolution(self, face_image):
        H, W = face_image.shape[:2]
        if H < self.rolu_thrds[0] or W < self.rolu_thrds[1]:
            return False 
        return True


class QualityOfPose:
    def __init__(self, pad=0.3, yaw_thrd=30, pitch_thrd=25, var_onnx_path = './checkpoints/fsanet-var.onnx', conv_onnx_path='./checkpoints/fsanet-conv.onnx') -> None:
        self.pad = pad
        self.input_width = 64
        self.input_height = 64
        self.yaw_thrd = abs(yaw_thrd) 
        self.pitch_thrd = abs(pitch_thrd)
        self.var_fsanet = ort.InferenceSession(var_onnx_path)
        self.conv_fsanet = ort.InferenceSession(conv_onnx_path)

        self.var_input_names = [input_.name for input_ in self.var_fsanet.get_inputs()]
        self.var_output_names = [output.name for output in self.var_fsanet.get_outputs()]

        self.conv_input_names = [input_.name for input_ in self.conv_fsanet.get_inputs()]
        self.conv_output_names = [output.name for output in self.conv_fsanet.get_outputs()]

    def transform(self, mat):
        h, w = mat.shape[:2]
        nh = int(h + self.pad * h)
        nw = int(w + self.pad * w)
        nx1 = max(0, (nw - w) // 2)
        ny1 = max(0, (nh - h) // 2)

        # Create a padded canvas and copy the image into the center
        canvas = np.zeros((nh, nw, 3), dtype=np.uint8)
        canvas[ny1:ny1 + h, nx1:nx1 + w] = mat

        # Resize the image to the input dimensions
        canvas = cv2.resize(canvas, (self.input_width, self.input_height))

        # Normalize the image in-place
        canvas = canvas.astype(np.float32)
        mean = 127.5
        scale = 1.0 / 127.5
        canvas = (canvas - mean) * scale

        # Convert to CHW format
        canvas = np.transpose(canvas, (2, 0, 1))

        # Create a tensor
        input_tensor = np.expand_dims(canvas, axis=0).astype(np.float32)
        return input_tensor
    
    def detect_angle(self, img):
        input_tensor = self.transform(img)

        var_output = self.var_fsanet.run(
            self.var_output_names, {self.var_input_names[0]: input_tensor}
        )[0]
        conv_output = self.conv_fsanet.run(
            self.conv_output_names, {self.conv_input_names[0]: input_tensor}
        )[0]
        yaw, pitch, roll = np.mean(np.vstack((var_output,conv_output)), axis=0)
        
        return yaw, pitch, roll

    def check(self, image):
        yaw, pitch, roll = self.detect_angle(image)

        if abs(yaw) <= self.yaw_thrd and abs(pitch) <= self.pitch_thrd: 
            return "frontFace"
        elif yaw < -1.0 * self.yaw_thrd:
            return "rightFace" 
        elif yaw > self.yaw_thrd:
            return "leftFace"
        elif pitch > self.pitch_thrd:
            return "upFace"
        elif pitch < -1.0 * self.pitch_thrd:
            return "downFace"
        else:
            return "implementError"
    

if __name__ == "__main__":
    qp = QualityOfPose()
    img = cv2.imread("/home/bns/liteAitoolkit/Demos_onnx/examples/det_oeyecmouth.jpg")  
    angles = qp.check(img)
    print(f"ONNXRuntime Version yaw: {angles[0]} pitch: {angles[1]} roll: {angles[2]}")
    pass