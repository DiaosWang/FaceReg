from fastapi import FastAPI
from pydantic import BaseModel
from process import FaceHelper, FileError, ErrorMsg 
import threading
import os 
import time 
app = FastAPI()
lock = threading.Lock()


facehelper = FaceHelper(
    db_dir="./dbface",
)

class face(BaseModel):
    img:str

class dbface(BaseModel):
    img:str
    optMode:str
    uniqueKey:str 

@app.post("/refreshdb")
@app.post("/refreshdb/")
def refresh():
    global facehelper
    try: 
        with lock:
            facehelper.updateDB(None, None, None, Onlyrefresh=True)
    except FileError as e: 
        # return {"status":e.code, "detail":f"{e}"}
        return {'code': e.code, 'msg': f"{e}", 'data': 'None'}
    else: 
        return {'code': "30002", 'msg': ErrorMsg['30002'], 'data': 'None'}

@app.post("/facerecognition")
@app.post("/facerecognition/")
def faceRecognition(input:face):
    start = time.time()
    global facehelper
    try: 
        ret_data = facehelper.faceRecognition(input.img)
        print("finished recognition...")
        end = time.time()
        print("runtime: ", end-start)
    except Exception as e:
        return {'code': e.code, 'msg': f"{e}", 'data': 'None'}
        # return {"status":f"{e.code}", "detail":f"{e}"} 
    else: 
        return ret_data
        return {"status":1, "name":identity, "resImg":res_img_base64}

@app.post("/featuredetect")
@app.post("/featuredetect/")
def featureDetect(input:face):
    start = time.time()
    global facehelper
    try: 
        ret_data = facehelper.featureDetect(input.img)
        print("finished featuredetect...")
        end = time.time()
        print("runtime: ", end-start)
    except Exception as e:
        return {'code': e.code, 'msg': f"{e}", 'data': 'None'}
        # return {"status":f"{e.code}", "detail":f"{e}"} 
    else: 
        return ret_data

@app.post("/updatedb")
@app.post("/updatedb/")
def updateDB(input:dbface):
    global facehelper
    # input.uniqueKey = os.path.splitext(os.path.basename(input.uniqueKey))[0]  

    try: 
        with lock:
            facehelper.updateDB(input.img, input.optMode, input.uniqueKey)
    except Exception as e:
        return {'code': e.code, 'msg': f"{e}", 'data': 'None'}
        # return {"status":f"{e.code}", "detail":f"{e}"}
    else: 
        return {'code': "30002", 'msg': ErrorMsg['30002'], 'data': 'None'}
    
