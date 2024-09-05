# created on 2024/6/12  
# modified on 2024/6/12 
# description: tool file (.py)
import cv2
import os 
import hashlib
import pickle
import requests
import base64
import logging  
import numpy as np 
from datetime import datetime
import redis
from redisLock import RedisReadWriteLock
import onnxruntime 
import time 
import yaml
from models import FaceRecoger, FaceBoxesV2, Landmark5er, FaceAlign, QualityOfClarity, QualityOfPose, QualityChecker

so = onnxruntime.SessionOptions()
so.log_severity_level = 3  # 0=VERBOSE, 1=INFO, 2=WARNING, 3=ERROR, 4=FATAL


# 获取workers 
if "NUM_WORKERS" not in os.environ:
    raise RuntimeError("Environment variable NUM_WORKERS is required but not set.")
NUM_WORKERS = int(os.getenv("NUM_WORKERS", 10)) 

# max readers 
max_readers = int(os.getenv("MAX_READERS",60))

# 连接到 Redis
redis_host = str(os.getenv("REDIS_HOST", 'localhost')) 
redis_port = int(os.getenv("REDIS_PORT", 2012))
redis_password = str(os.getenv("REDIS_PASSWORD", 'Xjsfzb@Redis123!')) 
# connected
redis_client = redis.Redis(host=redis_host, port=redis_port, password=redis_password, db=0)

PID_id = None 
NUM_WORKERS = int(os.getenv("NUM_WORKERS", 10))
for i in range(NUM_WORKERS):
    if redis_client.setnx(f"worker_{i}", 0): # 设置为dirty
        PID_id = i 
        break

# create ReadWriteLock
rw_lock = RedisReadWriteLock(redis_client, max_readers=max_readers)

ErrorMsg = {
    
    "30003": "no face found in the database",
    "30004": "invalid file path",
    "30005": "invalid file suffix",

    "30006": "unsupported input file type; only supports: base64, URL, local path",
    "30007": "unsupported type for database update",

    "30000": "facial recognition of a known individual",
    "30001": "facial recognition of an unknown individual",
    "30002": "successful image processing",
    "30018": "no face detected in the image",
    "30008": "multiple faces found in the image",
    "30009": "poor bright conditions affecting facial quality in the image",
    "30010": "face partially out of frame, shifted left or right",
    "30011": "face partially out of frame, shifted top or bottom",
    "30012": "face rotated to the right in the image",
    "30013": "face rotated to the left in the image",
    "30014": "face rotated upwards in the image",
    "30015": "face rotated downwards in the image",
    "30016": "low resolution of a face in the image",
    "30017": "poor clarity of a face in the image",
    
    "30019": "identity already exists; for database protection, operation rejected at this time"
}

class FileError(Exception):
    def __init__(self, arg:str):
        self.code = arg
        self.args = [f"{str(self.__class__.__name__)} {str(arg)}: {ErrorMsg[arg]}"]

class NotImpltError(Exception):
    def __init__(self, arg:str):
        self.code = arg
        self.args = [f"{str(self.__class__.__name__)} {str(arg)}: {ErrorMsg[arg]}"]

class FaceError(Exception):
    def __init__(self, arg:str):
        self.code = arg
        self.args = [f"{str(self.__class__.__name__)} {str(arg)}: {ErrorMsg[arg]}"]

class UpdatedbError(Exception):
    def __init__(self, arg:str):
        self.code = arg
        self.args = [f"{str(self.__class__.__name__)} {str(arg)}: {ErrorMsg[arg]}"]

# setting Logger 
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_dir = f"{os.path.dirname(os.path.abspath(__file__))}/log"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=f'{log_dir}/{current_time}.log', level=logging.INFO,  
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')  
logger = logging.getLogger(__name__)  # @@@@
print(log_dir)
 
def list_images(path: str):
    """
    List images in a given path
    Args:
        path (str): path's location
    Returns:
        images (list): list of exact image paths
    """
    images = []
    for r, _, f in os.walk(path, followlinks=True):
        for file in f:
            exact_path = os.path.join(r, file)

            _, ext = os.path.splitext(exact_path)
            ext_lower = ext.lower()

            if ext_lower not in {".jpg", ".jpeg", ".png"}:
                continue
            images.append(exact_path)

            # with Image.open(exact_path) as img:  # lazy
            #     if img.format.lower() in ["jpeg", "png"]:
            #         images.append(exact_path)
    return images


def find_image_hash(file_path: str) -> str:
    """
    Find the hash of given image file with its properties
        finding the hash of image content is costly operation
    Args:
        file_path (str): exact image path
    Returns:
        hash (str): digest with sha1 algorithm
    """
    file_stats = os.stat(file_path)

    # some properties
    file_size = file_stats.st_size
    creation_time = file_stats.st_ctime
    modification_time = file_stats.st_mtime

    properties = f"{file_size}-{creation_time}-{modification_time}"

    hasher = hashlib.sha1()
    hasher.update(properties.encode("utf-8"))
    return hasher.hexdigest()

# 支持base64 local-path url 等多种检索图片的方式，返回 numpy
def load_img(img_path:str):
    image = None 
    try: 
        if img_path.startswith(("http","www")): # url 
            response = requests.get(url=img_path, stream=True, timeout=60, proxies={"http": None, "https": None})
            response.raise_for_status()
            image_array = np.asarray(bytearray(response.raw.read()), dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        elif img_path.startswith(("./","/","C:","D:","E:",".\\")) or os.path.isfile(img_path):  # local-path  
            if not os.path.isfile(img_path):
                raise FileError("30004")  # push: invalid file path 
            elif not img_path.lower().endswith((".jpg",'.jpeg','.png')):
                raise FileError("30005") # push: invaild file suffix
            else:
                image = cv2.imread(img_path)
        elif img_path.startswith("data:") and "base64" in img_path: # base64 
            encoded_data_parts = img_path.split(",")
            if len(encoded_data_parts) <= 0:
                raise FileError("104")  # push: base64 is empty 
                print( "base64 is empty" )  
            encoded_data = encoded_data_parts[-1]
            nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            raise NotImpltError("30006")  # push: input file type is not support, only support: base64, url, local-path
    except Exception as e:
        logger.info(f"{e}")
        raise e 
        # return e
    else: 
        return image
     

def encoder_img2base64(img:np.ndarray):
    success, encoded_img = cv2.imencode('.png', img)
    if success:
        img_base64 = base64.b64encode(encoded_img).decode("utf-8")

    return ",".join(["data:image/jpg;base64", img_base64])


# from seetaface.api import *
class FaceHelper:

    def __init__(self, db_dir, config_path = './config.yaml'):
        self.db_dir = os.path.abspath(db_dir)
        self.pid = PID_id
        self.db_embeddings = None 
        self.db_identities = None 

        # 根据config_path 读取ymal配置文件，然后进行初始化
        with open(config_path, 'r') as f: 
            config = yaml.safe_load(f) 

        self.sim_threshold = config['faceReg']['sim_threshold'] # 0.7 
        self.rotclsifer = onnxruntime.InferenceSession( config['ck_paths']['rotifer'], so) # "./checkpoints/model_gray_mobilenetv2_rotcls.onnx"
        self.db_path = os.path.join( db_dir, "seetaface6.pkl" ).lower()

        self.fd = FaceBoxesV2(config['ck_paths']['FcBx'], config['ck_paths']['num_threads'] ) # r"./checkpoints/faceboxesv2-640x640.onnx"  4
        self.ld5er = Landmark5er( onnx_path1 = config['ck_paths']['landmk1'],    # "./checkpoints/face_landmarker_pts5_net1.onnx", 
                                  onnx_path2 = config['ck_paths']['landmk2'],    # "./checkpoints/face_landmarker_pts5_net2.onnx", 
                                  num_threads=config['ck_paths']['num_threads'] # 4 
                                )
        self.fa = FaceAlign()
        self.fr = FaceRecoger(onnx_path = config['ck_paths']['FcReg'], num_threads= config['ck_paths']['num_threads'] )  # "./checkpoints/face_recognizer.onnx"  4 
        self.qc = QualityChecker(config['brightness']['v0'], config['brightness']['v1'], 
                                 config['brightness']['v2'], config['brightness']['v3'],
                                 hw = (config['resolution']['height'], config['resolution']['width']) 
                                ) # v0=70.0, v1=100.0, v2=210.0, v3=230.0
        self.qpose = QualityOfPose(yaw_thrd=config['pose']['yaw_thrd'], pitch_thrd=config['pose']['pitch_thrd'], 
                                   var_onnx_path = config['pose']['var_onnx_path'], # './checkpoints/fsanet-var.onnx', 
                                   conv_onnx_path = config['pose']['conv_onnx_path'], # './checkpoints/fsanet-conv.onnx'
                                   )
        self.qclarity = QualityOfClarity(low_thresh=config['clarity']['low_thrd'], high_thresh=config['clarity']['high_thrd'])

        # refresh the db  
        try: 
            self.updateDB(None, None, None, Onlyrefresh=True)
        except Exception as e: 
            # raise e 
            pass

        print(f"db_dir: {self.db_dir}; PID: {self.pid}")
        logger.info(f"db_dir: {self.db_dir}  ; PID: {self.pid}")

    # 读操作
    def faceRecognition(self, img_path:str):
        rw_lock.acquire_read()
        if int(redis_client.get(f"worker_{self.pid}")) == 0:  # 说明self中的db和磁盘中的db不同步
            with open(self.db_path, "rb") as f:
                representations = pickle.load(f)
            if representations  == []:
                self.db_embeddings, self.db_identities = None, None 
            else:
                self.db_embeddings = np.array([rep["embedding"] for rep in representations], dtype=np.float32) 
                self.db_identities = [os.path.splitext(os.path.basename(rep["identity"]))[0] for rep in representations]
            redis_client.set(f"worker_{self.pid}", 1) # 同步完毕

        try: 
            if self.db_embeddings is None: 
                raise FileError("30003")  # push: no face in the database
            
            image = load_img(img_path) # get bgr numpy image 

            start = time.time()
            unknown_embeddings, cropped_images, names = [], [], []
            image = self.rotadjust(image) # 调整角度
            detect_result = self.fd.detect(image)
            detect_result = [(box.x1, box.y1, box.x2, box.y2) for box in detect_result]
            if len(detect_result) == 0:
                logger.info(f"{img_path[:200]}: no face in the image")
                print(f"{img_path[:200]}: no face in the image")
                raise FaceError("30018") # push: no face in the image 
            for facebox in detect_result:
                landmarks5 = self.ld5er.inference(image, facebox)  #  return: [(),(),(),(),()]  左眼 右眼 鼻子 左嘴角 右嘴角
                # print("5点关键点：",landmarks5)
                # 输入image 和5点特征点位置(基于原图image的位置) , return all cropped aligned face （裁剪后的对齐后的人脸部分图像, 简写为aligned_faces）
                landmarks5 = [ [ld5[0],ld5[1]] for ld5 in landmarks5]
                cropped_face = self.fa.align(image, landmarks5=landmarks5)
                # 输入aligned_faces ，return all features of aligned_faces 
                feature = self.fr.inference(cropped_face)
                cropped_images.append(cropped_face)
                unknown_embeddings.append(feature)
            
            unknown_embeddings = np.vstack(unknown_embeddings)
            results = np.dot(unknown_embeddings, self.db_embeddings.T)

            max_values = np.max(results,axis=1)
            max_idxs = np.argmax(results,axis=1)

            for i, (idx, value) in enumerate(zip(max_idxs, max_values)):
                name = "unknown"
                if value > self.sim_threshold:
                    name = self.db_identities[idx]
                names.append(name)

            ret_data = []
            for i, (facebox, name) in enumerate(zip(detect_result, names)):
                if name != 'unknown':
                    ret_data.append({'code':"30000", 'msg': ErrorMsg["30000"], 'data':[name,facebox]})
                else:
                    code = self.check_face("None", image, facebox, prefix='facereg')
                    if code == "30002":
                        ret_data.append({'code':"30001",  'msg': ErrorMsg["30001"],  'data':[name,facebox]}) 
                    else:
                        ret_data.append({'code':code,  'msg': ErrorMsg[code],  'data':[name,facebox]})
            if len(ret_data) != 1:
                ret_data = {'code':"30008", 'msg': ErrorMsg["30008"], 'data': ret_data}
            else:
                ret_data = ret_data[0]

            print("facereg runtime:", time.time() - start)            

        except Exception as e:
            logger.info(f"{e}")
            rw_lock.release_read()
            raise e
        else:
            rw_lock.release_read()
            return ret_data
            # return names, [ encoder_img2base64(det) for det in cropped_images]

    # 只是提取人脸特征，不要对别的有任何影响
    def featureDetect(self, img_path:str):
        rw_lock.acquire_read()

        try:             
            image = load_img(img_path) # get bgr numpy image 

            unknown_embeddings = []
            image = self.rotadjust(image) # 调整角度
            detect_result = self.fd.detect(image)
            detect_result = [(box.x1, box.y1, box.x2, box.y2) for box in detect_result]
            if len(detect_result) == 0:
                logger.info(f"{img_path[:200]}: no face in the image")
                print(f"{img_path[:200]}: no face in the image")
                raise FaceError("30018") # push: no face in the image 
            for facebox in detect_result:
                landmarks5 = self.ld5er.inference(image, facebox)  #  return: [(),(),(),(),()]  左眼 右眼 鼻子 左嘴角 右嘴角
                # print("5点关键点：",landmarks5)
                # 输入image 和5点特征点位置(基于原图image的位置) , return all cropped aligned face （裁剪后的对齐后的人脸部分图像, 简写为aligned_faces）
                landmarks5 = [ [ld5[0],ld5[1]] for ld5 in landmarks5]
                cropped_face = self.fa.align(image, landmarks5=landmarks5)
                # 输入aligned_faces ，return all features of aligned_faces 
                feature = self.fr.inference(cropped_face)
                unknown_embeddings.append(feature)

            ret_data = []
            for _, (facebox, feature) in enumerate(zip(detect_result, unknown_embeddings)):
                code = self.check_face("None", image, facebox, prefix='featuredetect')
                ret_data.append({'code':code,  'msg': ErrorMsg[code],  'data': [f"{feature.tobytes()},{str(feature.dtype)}", facebox]})
                 
            if len(ret_data) != 1:
                ret_data = {'code':"30008", 'msg': ErrorMsg["30008"], 'data': ret_data}
            else:
                ret_data = ret_data[0]

        except Exception as e:
            logger.info(f"{e}")
            rw_lock.release_read()
            raise e
        else:
            rw_lock.release_read()
            return ret_data

    # opt in ['add','delete','replace'] identity作为检索的标识符，img_path只是提供文件路径
    # 写操作
    def updateDB(self, img_path :str, opt :str, identity :str, Onlyrefresh=False):
        global rw_lock
        rw_lock.acquire_write() # 写锁定
        print("come in the updatedb")
        try:        
            if not Onlyrefresh:
                if int(redis_client.get(f"worker_{self.pid}")) == 0:  # 说明self中的db和磁盘中的db不同步
                    with open(self.db_path, "rb") as f:
                        representations = pickle.load(f)
                    if representations  == []:
                        self.db_embeddings, self.db_identities = None, None 
                    else:
                        self.db_embeddings = np.array([rep["embedding"] for rep in representations], dtype=np.float32) 
                        self.db_identities = [os.path.splitext(os.path.basename(rep["identity"]))[0] for rep in representations]
                    redis_client.set(f"worker_{self.pid}", 1) # 同步完毕

                img = load_img(img_path)
                img = self.rotadjust(img) # 调整角度
                if opt in ["add","replace"]: 
                    if opt == "add" and self.db_identities is not None and identity in self.db_identities: 
                        raise UpdatedbError("30019") # push: identity has exist. to pretect the db, reject opt of this time 
                    else:
                        
                        detect_result = self.fd.detect(img)
                        if len(detect_result) == 0: # no face
                            logger.info(f"{img_path[:200]}: when update, no face in the image")
                            print(f"{img_path[:200]}: when update, no face in the image")
                            raise FaceError("30018") # push: no face in the image 
                        else: # 获取最大的face,然后进行check 
                            # H, W = img.shape[:2]
                            areas = [ box.area() for box in detect_result]
                            max_idx = areas.index(max(areas))
                            facebox = detect_result[max_idx]
                            facebox = (facebox.x1, facebox.y1, facebox.x2, facebox.y2) # top_left point,  bottom_right point
                            FaceError_number = self.check_face(img_path=img_path[:200], img=img, facebox=facebox, prefix='update')
                            if FaceError_number != "30002":
                                raise FaceError(FaceError_number)

                        cv2.imwrite(os.path.join(self.db_dir, identity+'.jpg'),img,[cv2.IMWRITE_JPEG_QUALITY, 100])  # 如果file已经存在，则会替换它
                            
                elif opt == "delete":
                    try: 
                        os.remove(os.path.join(self.db_dir, identity+'.jpg'))
                    except FileNotFoundError:
                        pass 
                else:
                    raise NotImpltError("30007")  # push: this updateDB type is not support
                
                print("end the updateDB")
                logger.info(f"end the updateDB")

            self.refresh_database(check = Onlyrefresh)  # 结束时刷新下db, 并通知别的进程，dirty
        except Exception as e:
            logger.info(f"{e}")
            rw_lock.release_write()
            raise e 
        else: 
            rw_lock.release_write()
            return 0

    def refresh_database(self, check = True):
        # ensure db exist 
        os.makedirs(self.db_dir, exist_ok=True)
        if not os.path.exists(self.db_path):
            with open(self.db_path, "wb") as f:
                pickle.dump([], f)
        representations = [] # representations 最后要储存在db中
        # Load the representations from the pickle file
        with open(self.db_path, "rb") as f:
            representations = pickle.load(f)

        # get identities of image 
        pickle_images = [rep["identity"] for rep in representations]

        # get the list of images on the dir 
        storage_images = list_images(self.db_dir)

        # transform all images in storage_images to `.jpg`
        for idx in range(len(storage_images)): 
            img_path = storage_images[idx]
            base_path, ext = os.path.splitext(img_path)
            if ext == '.jpg':
                continue 
            iimg = cv2.imread(img_path)
            cv2.imwrite(base_path+'.jpg', iimg, [cv2.IMWRITE_JPEG_QUALITY, 100])
            storage_images[idx] = base_path+'.jpg'

        must_save_pickle = False
        new_images = []; old_images = []; replaced_images = []

        new_images = list(set(storage_images) - set(pickle_images)) 
        old_images = list(set(pickle_images) - set(storage_images))

        for current_representation in representations: # 找到被替换的images
            identity = current_representation["identity"]
            if identity in old_images:
                continue
            alpha_hash = current_representation["hash"]
            beta_hash = find_image_hash(identity)
            if alpha_hash != beta_hash:
                # logger.debug(f"Even though {identity} represented before, it's replaced later.")
                replaced_images.append(identity) 
                
        new_images = new_images + replaced_images
        old_images = old_images + replaced_images

        # remove old images first
        if len(old_images) > 0:
            representations = [rep for rep in representations if rep["identity"] not in old_images]
            must_save_pickle = True

        # find representations for new images
        if len(new_images) > 0:
            print("find new images")
            new_representations = []
            for new_image in new_images:
                image = cv2.imread(new_image)
                image = self.rotadjust(image)  # 调整旋转角度
                detect_result = self.fd.detect(image)
                if len(detect_result) == 0:
                    logger.info(f"{new_image}: when refresh, no face in the image, delete")
                    print(f"{new_image}: when refresh, no face in the image, delete")
                else:
                    if  len(detect_result) > 1: 
                        logger.info(f"{new_image}: find too many face, get and extract the biggest face in them")
                    else: 
                        logger.info(f"{new_image}: find one face, perfect!")

                    areas = [ box.area() for box in detect_result]
                    max_idx = areas.index(max(areas))
                    facebox = detect_result[max_idx]
                    facebox = (facebox.x1, facebox.y1, facebox.x2, facebox.y2) # top_left point,  bottom_right point 
                    
                    if check:
                        FaceError_number = self.check_face(img_path=new_image[:200], img=image, facebox=facebox, prefix='refreshdb')
                        if FaceError_number != "30002":
                            continue 

                    landmarks5 = self.ld5er.inference(image, facebox)  #  return: [(),(),(),(),()]  左眼 右眼 鼻子 左嘴角 右嘴角
                    landmarks5 = [ [ld5[0],ld5[1]] for ld5 in landmarks5]
                    cropped_face = self.fa.align(image, landmarks5=landmarks5)
                    feature = self.fr.inference(cropped_face)

                    new_representations.append({
                        "identity": new_image, 
                        "hash": find_image_hash(new_image),
                        "embedding": feature,
                        "detected_face_base64": encoder_img2base64(cropped_face),
                    })
                
            representations += new_representations
            must_save_pickle = True    
        
        if must_save_pickle: 
            print("must save the pickle")
            with open(self.db_path, "wb") as f:
                pickle.dump(representations, f)
            global redis_client, NUM_WORKERS
            for i in range(NUM_WORKERS):
                redis_client.set(f"worker_{i}", 0)  # 通知别的进程db有更新 

        # 保证db_dir 中的图片和self.db["identity"] 一致
        remove_images = list(set(storage_images) - set([rep["identity"] for rep in representations]))
        for remove_img in remove_images: 
            try: 
                # os.remove(remove_img)
                fname = os.path.basename(remove_img) 
                # os.rename( remove_img, os.path.join(self.db_dir, "..","images","remove",fname) )
            except FileNotFoundError:
                pass         
        
        if int(redis_client.get(f"worker_{self.pid}")) == 0:
            empty = False
            if len(representations) <= 0:
                self.db_embeddings = None
                empty = True
                # raise FileError("30003")  # push: no face in db 
            else:
                self.db_embeddings = np.array([rep["embedding"] for rep in representations], dtype=np.float32)
                self.db_identities = [os.path.splitext(os.path.basename(rep["identity"]))[0] for rep in representations]
            redis_client.set(f"worker_{self.pid}", 1)  # 当前进程已更新
            if empty: 
                logger.info("no face in the database")
                raise FileError("30003")  # push: no face in db

    def rotadjust(self, img: np.ndarray):
        image = img.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # 转换为灰度图像
        image = cv2.resize(image, (256, 256)) # resize (256,256)
        
        # 中心裁剪到 224x224
        start = (256 - 224) // 2
        image = image[start:start+224, start:start+224]
        
        # 将单通道灰度图像转换为三通道
        image = np.stack((image,)*3, axis=-1)
        
        # 转换为符合 ONNX 需要的格式
        image = image.astype(np.float32) / 255.0  # 归一化
        image = image.transpose(2, 0, 1)  # 将图像从 HWC 格式转换为 CHW 格式
        image = np.expand_dims(image, axis=0)  # 增加一个批次维度

        inputs = {self.rotclsifer.get_inputs()[0].name: image}
        probs = self.rotclsifer.run(None, inputs)
        
        label = np.argmax(probs[0][0])  # 推理得到的逆时针旋转角度 [0,90,180,270] 
        if label == 1:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            logger.info("img turn left, use `cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)` to repair")
            print("img turn left, use `cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)` to repair")
        elif label == 2:
            img = cv2.rotate(img, cv2.ROTATE_180)
            logger.info("img flip the image vertically, use `cv2.rotate(img, cv2.ROTATE_180)` to repair")
            print("img flip the image vertically, use `cv2.rotate(img, cv2.ROTATE_180)` to repair")
        elif label == 3:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            logger.info("img turn right, use `cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)` to repair")
            print("img turn right, use `cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)` to repair")

        return img
    
    def get_feature(self, img: np.ndarray):
        time.sleep(0.08)
        # assert img.shape[0] == img.shape[1] and img.shape[0] == 256 and img.shape[2] == 3
        img = cv2.resize( img, (256,256) ) 
        input_feed = {}
        # crop_img = cv2.resize(img,(248,248))
        # crop_img = crop_img[...,::-1]
        crop_img = img[4:252, 4:252, :][...,::-1]  # 注意要考虑 长或宽 < 248的情况
        input_data = crop_img.transpose((2, 0, 1))
        # resize_img = cv2.resize(img, (248, 248))
        # input_data = resize_img.transpose((2, 0, 1))
        input_feed['_input_123'] = input_data.reshape((1, 3, 248, 248)).astype(np.float32)
        pred_result = self.FR.run([], input_feed=input_feed)
        # print(pred_result[0].shape)
        # post process
        # 1 sqrt feature
        temp_result = np.sqrt(pred_result[0])
        # 2 normalization feature
        norm = temp_result / np.linalg.norm(temp_result, axis=1)
        return norm.flatten()
    
    def check_face(self, img_path, img, facebox, prefix="update"):
        H, W = img.shape[:2]
        if facebox[0] < 0 or facebox[2] >= W:
            logger.info(f"{img_path}: when {prefix}, face shifted left/right")
            print(f"{img_path}: when {prefix}, face shifted left/right")
            return "30010"  # face shifted left/right, partially not captured.
        if facebox[1] < 0 or facebox[3] >= H:
            logger.info(f"{img_path}: when {prefix}, face shifted top/bottom")
            print(f"{img_path}: when {prefix}, face shifted top/bottom")
            return "30011" # face shifted top/bottom, partially not captured.
        face_img = img[ max(0,int(facebox[1])):int(facebox[3]), max(0,int(facebox[0])):int(facebox[2]) ]
        if not self.qc.check_bright(face_img):
            logger.info(f"{img_path}: when {prefix}, bad bright face in the image")
            print(f"{img_path}: when {prefix}, bad bright face in the image")
            return "30009" #  bad bright face in the image
        if not self.qc.check_resolution(face_img):
            logger.info(f"{img_path}: when {prefix}, too small resolution of face in the image")
            print(f"{img_path}: when {prefix}, too small resolution of face in the image")
            return "30016"  #  small face in the image
        pose = self.qpose.check(face_img)
        if pose != "frontFace":
            logger.info(f"{img_path}: when {prefix}, {pose} in the image")
            print(f"{img_path}: when {prefix}, {pose} in the image")
            dictt = {"rightFace": "30012", "leftFace": "30013", "upFace": "30014", "downFace": "30015"}
            return dictt[pose] # pose of face in the image
        if not self.qclarity.check(face_img):
            logger.info(f"{img_path}: when {prefix}, bad clarity of face in the image")
            print(f"{img_path}: when {prefix}, bad clarity of face in the image")
            return "30017" #  poor clarity of face in the image

        return "30002"