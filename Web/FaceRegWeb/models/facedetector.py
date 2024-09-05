"""
输入：原图
输出：图片中face框
"""

import cv2
import numpy as np
import onnxruntime as ort


class Box:
    def __init__(self, x1, y1, x2, y2, score, label=1, label_text = 'face' ,flag=True):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.score = score
        self.label = label
        self.label_text = label_text
        self.flag = flag

    def iou_of(self, other):
        inter_x1 = max(self.x1, other.x1)
        inter_y1 = max(self.y1, other.y1)
        inter_x2 = min(self.x2, other.x2)
        inter_y2 = min(self.y2, other.y2)
        
        if inter_x1 < inter_x2 and inter_y1 < inter_y2:
            inter_area = (inter_x2 - inter_x1 + 1.0) * (inter_y2 - inter_y1 + 1.0)
            self_area = (self.x2 - self.x1 + 1.0) * (self.y2 - self.y1 + 1.0)
            other_area = (other.x2 - other.x1 + 1.0) * (other.y2 - other.y1 + 1.0)
            union_area = self_area + other_area - inter_area
            return inter_area / union_area
        else:
            return 0
        
    def area(self):
        return (self.x2 - self.x1 + 1) * (self.y2 - self.y1 + 1)

def hard_nms(boxes, iou_threshold, topk):
    if not boxes:
        return []
    boxes.sort(key=lambda x: x.score, reverse=True)
        
    merged = [0] * len(boxes)
    output = []

    count = 0
    for i in range(len(boxes)):
        if merged[i]:
            continue
        buf = [boxes[i]]
        merged[i] = 1

        for j in range(i + 1, len(boxes)):
            if merged[j]:
                continue

            iou = boxes[i].iou_of(boxes[j])
            if iou > iou_threshold:
                merged[j] = 1
                buf.append(boxes[j])

        output.append(buf[0])

        count += 1
        if count >= topk:
            break
    return output

def blending_nms(boxes, iou_threshold, topk):
    if not boxes:
        return []
    boxes.sort(key=lambda x: x.score, reverse=True)
    merged = [0] * len(boxes)
    output = []

    count = 0
    for i in range(len(boxes)):
        if merged[i]:
            continue
        buf = [boxes[i]]
        merged[i] = 1

        for j in range(i + 1, len(boxes)):
            if merged[j]:
                continue

            iou = boxes[i].iou_of(boxes[j])
            if iou > iou_threshold:
                merged[j] = 1
                buf.append(boxes[j])

        total = sum([np.exp(box.score) for box in buf])
        rects = Box(0, 0, 0, 0, 0)
        for box in buf:
            rate = np.exp(box.score) / total
            rects.x1 += box.x1 * rate
            rects.y1 += box.y1 * rate
            rects.x2 += box.x2 * rate
            rects.y2 += box.y2 * rate
            rects.score += box.score * rate
        rects.flag = True
        output.append(rects)

        count += 1
        if count >= topk:
            break
    return output

def offset_nms(boxes, iou_threshold, topk):
    if not boxes:
        return []
    boxes.sort(key=lambda x: x.score, reverse=True)
    merged = [0] * len(boxes)
    offset = 4096.0

    for box in boxes:
        box.x1 += box.label * offset
        box.y1 += box.label * offset
        box.x2 += box.label * offset
        box.y2 += box.label * offset

    output = []
    count = 0
    for i in range(len(boxes)):
        if merged[i]:
            continue
        buf = [boxes[i]]
        merged[i] = 1

        for j in range(i + 1, len(boxes)):
            if merged[j]:
                continue

            iou = boxes[i].iou_of(boxes[j])
            if iou > iou_threshold:
                merged[j] = 1
                buf.append(boxes[j])

        output.append(buf[0])

        count += 1
        if count >= topk:
            break

    for box in output:
        box.x1 -= box.label * offset
        box.y1 -= box.label * offset
        box.x2 -= box.label * offset
        box.y2 -= box.label * offset

    return output

def draw_rectface(img, box):
    x = max(0,int(box.x1))
    y = max(0,int(box.y1))
    w = min(img.shape[1]-x, int(box.x2-x+1))
    h = min(img.shape[0]-y, int(box.y2-y+1))
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
    # return img

def cut_rectface(img, box):
    x = max(0,int(box.x1))
    y = max(0,int(box.y1))
    w = min(img.shape[1]-x, int(box.x2-x+1))
    h = min(img.shape[0]-y, int(box.y2-y+1))
    return img[y:y+h,x:x+w]

def normalize_inplace(mat, mean, scale):
    mat = mat.astype(np.float32)
    mat -= mean
    mat *= scale
    return mat

def create_tensor(mat, tensor_dims, memory_info_handler, data_format):
    rows, cols, channels = mat.shape
    if len(tensor_dims) != 4:
        raise RuntimeError("dims mismatch.")
    if tensor_dims[0] != 1:
        raise RuntimeError("batch != 1")

    if data_format == "CHW":
        target_height = tensor_dims[2]
        target_width = tensor_dims[3]
        target_channel = tensor_dims[1]
        # target_tensor_size = target_channel * target_height * target_width
        if target_channel != channels:
            raise RuntimeError("channel mismatch.")
        
        if target_height != rows or target_width != cols:
            print("in create_tensor, resize mat...")
            mat = cv2.resize(mat, (target_width, target_height))
        
        mat = mat.transpose(2, 0, 1)  # HWC -> CHW   # 这儿存疑。 
        mat = np.expand_dims(mat, axis=0)
        return ort.OrtValue.ortvalue_from_numpy(mat, 'cpu')   
    
    elif data_format == "HWC":
        target_height = tensor_dims[1]
        target_width = tensor_dims[2]
        target_channel = tensor_dims[3]
        target_tensor_size = target_channel * target_height * target_width
        if target_channel != channels:
            raise RuntimeError("channel mismatch.")
        
        if target_height != rows or target_width != cols:
            mat = cv2.resize(mat, (target_width, target_height))
        
        return ort.OrtValue.ortvalue_from_numpy(mat, 'cpu')

class BasicOrtHandler:
    def __init__(self, onnx_path, num_threads=1):
        self.onnx_path = onnx_path
        self.num_threads = num_threads
        self.initialize_handler()

    def initialize_handler(self):
        # self.ort_env = ort.Env(ort.logging.ERROR)
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = self.num_threads
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # self.ort_session = ort.InferenceSession(self.onnx_path, session_options)
        # self.memory_info_handler = ort.OrtMemoryInfo("cpu", ort.OrtAllocatorType.ORT_ARENA_ALLOCATOR)

        # Initialize session
        self.ort_session = ort.InferenceSession(self.onnx_path, session_options)
        self.memory_info_handler = ort.OrtMemoryInfo("Cpu", ort.OrtAllocatorType.ORT_ARENA_ALLOCATOR, 0, ort.OrtMemType.DEFAULT)


        self.input_node_names = [self.ort_session.get_inputs()[0].name]
        self.input_node_dims = self.ort_session.get_inputs()[0].shape  # 获取输入张量的shape 
        self.input_tensor_size = np.prod(self.input_node_dims)

        self.output_node_names = [out.name for out in self.ort_session.get_outputs()]
        self.output_node_dims = [out.shape for out in self.ort_session.get_outputs()]
        self.num_outputs = len(self.output_node_names)

    def __del__(self):
        del self.ort_session

class FaceBoxesV2(BasicOrtHandler):
    def __init__(self, onnx_path, num_threads=1):
        super().__init__(onnx_path, num_threads)
        self.mean_vals = np.array([104.0, 117.0, 123.0], dtype=np.float32)
        self.scale_vals = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.variance = [0.1, 0.2]
        self.steps = [32, 64, 128]
        self.min_sizes = [
            [32, 64, 128],
            [256],
            [512]
        ]
        self.max_nms = 30000

    def transform(self, mat):
        canvas = cv2.resize(mat, (self.input_node_dims[3], self.input_node_dims[2]))
        canvas = normalize_inplace(canvas, self.mean_vals, self.scale_vals)
        return create_tensor(canvas, self.input_node_dims, self.memory_info_handler, "CHW")

    def detect(self, mat, score_threshold=0.35, iou_threshold=0.45, topk=300, nms_type=0):
        if mat is None or mat.size == 0:
            return
        
        img_height = float(mat.shape[0])
        img_width = float(mat.shape[1])

        # 1. make input tensor
        input_tensor = self.transform(mat)
        # 2. inference scores & boxes.
        output_tensors = self.ort_session.run(self.output_node_names, {self.input_node_names[0]: input_tensor})

        # 3. rescale & exclude.
        bbox_collection = []
        bbox_collection = self.generate_bboxes(output_tensors, score_threshold, img_height, img_width)

        # 4. hard|blend|offset nms with topk.  return detected_boxes
        return self.nms(bbox_collection, iou_threshold, topk, nms_type)
    
    def generate_bboxes(self, output_tensors, score_threshold, img_height, img_width):
        bboxes = output_tensors[0]  # e.g (1,n,4)
        probs = output_tensors[1]  # e.g (1,n,2) after softmax
        bbox_dims = self.output_node_dims[0]  # (1,n,4)
        bbox_num = bbox_dims[1]  # n = ?
        input_height = self.input_node_dims[2]  # e.g 640
        input_width = self.input_node_dims[3]  # e.g 640

        anchors = self.generate_anchors(input_height, input_width)

        num_anchors = len(anchors)
        if num_anchors != bbox_num:
            print(f"num_anchors={num_anchors} but detected bbox_num={bbox_num}")
            raise RuntimeError("mismatch num_anchors != bbox_num")

        bbox_collection = []
        count = 0
        for i in range(num_anchors):
            conf = probs[0, i, 1]
            if conf < score_threshold:
                continue  # filter first.

            # prior_cx = anchors[i].cx
            # prior_cy = anchors[i].cy
            # prior_s_kx = anchors[i].s_kx
            # prior_s_ky = anchors[i].s_ky
            prior_cx, prior_cy, prior_s_kx, prior_s_ky = anchors[i]

            dx = bboxes[0, i, 0]
            dy = bboxes[0, i, 1]
            dw = bboxes[0, i, 2]
            dh = bboxes[0, i, 3]

            cx = prior_cx + dx * self.variance[0] * prior_s_kx
            cy = prior_cy + dy * self.variance[0] * prior_s_ky
            w = prior_s_kx * np.exp(dw * self.variance[1])
            h = prior_s_ky * np.exp(dh * self.variance[1])  # norm coor (0.,1.)

            box = Box(
                x1=(cx - w / 2.0) * img_width,
                y1=(cy - h / 2.0) * img_height,
                x2=(cx + w / 2.0) * img_width,
                y2=(cy + h / 2.0) * img_height,
                score=conf,
                label=1,
                label_text="face",
                flag=True
            )
            bbox_collection.append(box)

            count += 1  # limit boxes for nms.
            if count > self.max_nms:
                break

        return bbox_collection
    
    def nms(self, input_boxes, iou_threshold, topk, nms_type):
        if nms_type == 1:
            output_boxes = blending_nms(input_boxes, iou_threshold, topk)
        elif nms_type == 2:
            output_boxes = offset_nms(input_boxes, iou_threshold, topk)
        elif nms_type == 0:
            output_boxes = hard_nms(input_boxes, iou_threshold, topk)
        else:
            raise NotImplementedError
        return output_boxes
   
    def generate_anchors(self, target_height, target_width):
        feature_maps = []
        for step in self.steps:
            feature_maps.append([
                int(np.ceil(target_height / step)),
                int(np.ceil(target_width / step))
            ])

        anchors = []
        for k, f_map in enumerate(feature_maps):
            tmp_min_sizes = self.min_sizes[k]
            f_h, f_w = f_map
            
            offset_32 = [0.0, 0.25, 0.5, 0.75]
            offset_64 = [0.0, 0.5]

            for i in range(f_h):
                for j in range(f_w):
                    for min_size in tmp_min_sizes:
                        s_kx = min_size / target_width
                        s_ky = min_size / target_height
                        
                        if min_size == 32:
                            for offset_y in offset_32:
                                for offset_x in offset_32:
                                    cx = (j + offset_x) * self.steps[k] / target_width
                                    cy = (i + offset_y) * self.steps[k] / target_height
                                    anchors.append([cx, cy, s_kx, s_ky])
                        elif min_size == 64:
                            for offset_y in offset_64:
                                for offset_x in offset_64:
                                    cx = (j + offset_x) * self.steps[k] / target_width
                                    cy = (i + offset_y) * self.steps[k] / target_height
                                    anchors.append([cx, cy, s_kx, s_ky])
                        else:
                            cx = (j + 0.5) * self.steps[k] / target_width
                            cy = (i + 0.5) * self.steps[k] / target_height
                            anchors.append([cx, cy, s_kx, s_ky])

        return anchors



# Usage example
if __name__ == "__main__":
    import sys 
    import os  
    img_path = sys.argv[1] 
    reta = FaceBoxesV2(r"./checkpoints/faceboxesv2-640x640.onnx",4)
    img = cv2.imread(img_path)
    detected_boxes = reta.detect(img)
    count = 0
    for box in detected_boxes:
        print(f"({box.x1:.3f},{box.y1:.3f},{box.x2:.3f},{box.y2:.3f})", end=" ")
        count += 1
    print("total face number:",count)

    for box in detected_boxes:
        draw_rectface(img, box)
    
    filename = os.path.basename(img_path)
    cv2.imwrite("./" + filename, img)
