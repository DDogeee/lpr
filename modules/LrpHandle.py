import torch
import pickle
from flask_restful import Resource, reqparse
from flask import send_file
import base64
import numpy as np
import cv2
from flask import Flask, render_template, request, redirect, url_for
import os
from modules.MobilenetV3 import MobileNetV3
import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore, storage
from config import firebase_config
import uuid
from pymongo import MongoClient
from config import URI, PRIVATE_KEY
from PIL import Image
from io import BytesIO
import shutil
from datetime import datetime
import jwt
cred = credentials.Certificate("service_account.json")
firebase_admin.initialize_app(cred, firebase_config)

KNNClassifier = pickle.load(open('models/KNNClassifier', 'rb'))
lp_detect = torch.hub.load('yolov5', 'custom', path='models/LP_detector.pt', force_reload=True, source='local')
char_detect = torch.hub.load('yolov5', 'custom', path='models/char_detector.pt', force_reload=True, source='local')
cnn = MobileNetV3(pretrained = 'models/CNN.pt')

class ImageReg(Resource):
    def __init__(self) -> None:
        self.collection = MongoClient(URI).main.log_video
        args = reqparse.RequestParser()
        # args.add_argument("token", type=str, required=True, help="token is missing")
        args.add_argument("vid_id", type=str, required=True, help="video_id is missing")
        self.args = args
        
    def post(self):
        try:
            token = request.form['token']
            user = jwt.decode(token, PRIVATE_KEY, algorithms=["HS256"])
            if user['role']=='user':
                return {
                    "error": True,
                    "message": "Not enough permission",
                    "data" : None,
                }
            img_id = uuid.uuid4().hex
            filestr = request.files['file'].read()
            file_bytes = np.fromstring(filestr, np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
            data = pd.read_csv("data.csv")
    #         args = args.parse_args()
    #         im_b64 = args['b64']
    #         im_bytes = base64.b64decode(im_b64)
    #         im_arr = np.frombuffer(im_bytes, dtype=np.uint8)  # im_arr is one-dim Numpy array
    #         image = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
            plates = lp_detect(image, size = 1024).pandas().xyxy[0].values.tolist()
            r = []
            result = image.copy()
            for plate in plates:
                detect_conf = plate[4]
                plate = plate[:4]
                if detect_conf > 0.8:
                    crop_plate = get_crop_image(plate, image)
                    chars = char_detect(crop_plate, size = 1024).pandas().xyxy[0].values.tolist()
                    chars = [char[:4] for char in chars]
                    chars = sort_chars(chars)
                    labels = []
                    for char in chars:
                        crop_char = get_crop_image(char, crop_plate)
                        X = np.array(cnn(image = crop_char).detach())
                        labels.append(KNNClassifier.predict(X)[0])
                    reg_plate = ''.join(labels)
                    query = data.query(f"plate_number == '{reg_plate}'")
                    if len(query) > 0 :
                        name = query.iloc[0]['name']
                        province = query.iloc[0]['province']
                    else:
                        name = "Unknow"
                        province = "Unknow"
                    r.append({
                        "box" : plate,
                        "conf" : detect_conf,
                        "plate" : reg_plate,
                        "province" : province,
                        "name" : name
                    })
                    x0, y0, x1, y1 = plate
                    result = cv2.rectangle(result, (int(x0), int(y0)), (int(x1), int(y1)), (36,255,12), 1)
                    result = cv2.putText(result, reg_plate, (int(x0), int(y0-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            _, im_arr = cv2.imencode('.jpg', result)  # im_arr: image in Numpy one-dim array format.
            im_bytes = im_arr.tobytes()
            im_b64 = base64.b64encode(im_bytes)
            collection = MongoClient(URI).main.log_image
            item = {
                "img_id" : img_id,
                "data" : r,
                "img" : im_b64.decode()
            } 
            collection.insert_one(item)

            return {
                "error": False,
                "message": "Success",
                "img_id" : img_id,
                "data" : im_b64.decode(),
                "log" : r,
            }
        except Exception as e:
            return {
                "error": True,
                "message": e,
                "data" : "",
            }


def get_crop_image(box, image):
    x = int(box[0]) # xmin
    y = int(box[1]) # ymin
    w = int(box[2] - box[0]) # xmax - xmin
    h = int(box[3] - box[1]) # ymax - ymin
    return image[y:y+h, x:x+w]

def sort_chars(chars):
    chars = sorted(chars, key=lambda s: s[1])
    line1 = []
    line2 = []
    for i, char in enumerate(chars):
        if len(line1) ==0:
            line1.append(char)
            continue
        else:
          up = line1[len(line1)-1][3]
          down = line1[len(line1)-1][1]
          mid = (char[1]+char[3])/2
          if mid < up and mid > down:
            line1.append(char)
          else:
              line2.append(char)
    line1 = sorted(line1, key=lambda s: s[0])
    line2 = sorted(line2, key=lambda s: s[0])
    return [*line1, *line2]

class VideoReg(Resource):
    def post(self):
        try:
            token = request.form['token']
            user = jwt.decode(token, PRIVATE_KEY, algorithms=["HS256"])
            if user['role']=='user':
                return {
                    "error": True,
                    "message": "Not enough permission",
                    "data" : None,
                }
            vid_id = uuid.uuid4().hex
            file = request.files['file']
            file.save("response.mp4")
            os.system("python track.py --source response.mp4 --save-txt --save-vid")
            bucket = storage.bucket()
            blob = bucket.blob('response.mp4')
            outfile='inference/output/response.mp4'
            with open(outfile, 'rb') as my_file:
                blob.upload_from_file(my_file)
            blob.make_public()
            df = pd.read_csv("inference/output/response.txt", header=None, sep = ' ').drop([8], axis = 1)
            df.columns = ['frame_idx', 'object_id', 'top', 'left', 'width', 'height', 'class_id', 'speed']
            df = df[df['speed'].apply(lambda s : s>0 and s<=80)]
            mean_speed = df.groupby("object_id").agg(mean_speed = ('speed', "mean"))
            df = pd.merge(df, mean_speed, on = 'object_id')
            df['diff'] = df.apply(lambda s : abs(s['speed'] - s['mean_speed']), axis = 1)
            df = df.sort_values(by = ['object_id', 'diff']).drop_duplicates("object_id", keep="first")

            l = []
            for idx in df.index:
                frame = df.loc[idx, "frame_idx"]
                speed = df.loc[idx, "mean_speed"]
                object_id = df.loc[idx, "object_id"]
                x = df.loc[idx, "left"]
                y = df.loc[idx, "top"]
                w = df.loc[idx, "width"]
                h = df.loc[idx, "height"]
                frame = cv2.imread(os.path.join("raw", str(frame)+".jpg"))
                crop_img = frame[x:x+h, y:y+w]

                plates = lp_detect(crop_img, size = 1024).pandas().xyxy[0].values.tolist()
                if len(plates)!=0: 
                    plate = plates[0]
                    detect_conf = plate[4]
                    plate = plate[:4]
                    if detect_conf > 0.6:
                        crop_plate = get_crop_image(plate, crop_img)
                        chars = char_detect(crop_plate, size = 1024).pandas().xyxy[0].values.tolist()
                        chars = [char[:4] for char in chars]
                        chars = sort_chars(chars)
                        labels = []
                        for char in chars:
                            crop_char = get_crop_image(char, crop_plate)
                            X = np.array(cnn(image = crop_char).detach())
                            labels.append(KNNClassifier.predict(X)[0])
                        reg_plate = ''.join(labels)
                    else:
                        reg_plate = ''
                else:
                    reg_plate = '' 
                img = Image.fromarray(crop_img)
                im_file = BytesIO()
                img.save(im_file, format="JPEG")
                im_bytes = im_file.getvalue()  # im_bytes: image in binary format.
                im_b64 = base64.b64encode(im_bytes)

                if int(speed)>60:
                    l.append({
                        "image" : im_b64.decode(),
                        "speed" : speed,
                        "plate" : reg_plate,
                        "speeding" : True,
                        "date" : datetime.strftime(datetime.now(), "%Y-%m-%d")
                    })
                    collection = MongoClient(URI).main.vi_pham
                    collection.insert_one(l[-1])
                else:
                    l.append({
                        "image" : im_b64.decode(),
                        "speed" : speed,
                        "plate" : reg_plate,
                        "speeding" : False,
                    })

            item = {
                "vid_id" : vid_id,
                "list" : l,
                "count" : len(l),
                "date" : datetime.strftime(datetime.now(), "%Y-%m-%d")
            }
            print(item)
            collection = MongoClient(URI).main.log_video
            collection.insert_one(item)
            shutil.rmtree("raw")
            return  {
                "error": False,
                "message": "Success",
                "data" : {
                    "cloudPath" : blob.public_url,
                    "video_id" : vid_id,
                }
            }
        except Exception as e:
            return {
                "error": True,
                "message": e,
                "data" : "",
            }

class GetVidInfo(Resource):
    def __init__(self) -> None:
        self.collection = MongoClient(URI).main.log_video
        args = reqparse.RequestParser()
        # args.add_argument("token", type=str, required=True, help="token is missing")
        args.add_argument("vid_id", type=str, required=True, help="video_id is missing")
        self.args = args

    def post(self):
        try:
            args = self.args.parse_args()
            item = self.collection.find_one({
                "vid_id": args['vid_id']
            })
            return {
                "error": False,
                "message": "",
                "data" : {
                    "list" : item['list'],
                    "count" : item['count']
                },
            }
        
        except Exception as e:
            return {
                "error": True,
                "message": e,
                "data" : "",
            }