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
from config import URI
from PIL import Image
from io import BytesIO

cred = credentials.Certificate("service_account.json")
firebase_admin.initialize_app(cred, firebase_config)

KNNClassifier = pickle.load(open('models/KNNClassifier', 'rb'))
lp_detect = torch.hub.load('yolov5', 'custom', path='models/LP_detector.pt', force_reload=True, source='local')
char_detect = torch.hub.load('yolov5', 'custom', path='models/char_detector.pt', force_reload=True, source='local')
cnn = MobileNetV3(pretrained = 'models/CNN.pt')

class ImageReg(Resource):
    def post(self):
        try:
            filestr = request.files['file'].read()
            file_bytes = np.fromstring(filestr, np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)

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
                    r.append({
                        "box" : plate,
                        "conf" : detect_conf,
                        "plate" : reg_plate
                    })
                    x0, y0, x1, y1 = plate
                    result = cv2.rectangle(result, (int(x0), int(y0)), (int(x1), int(y1)), (36,255,12), 1)
                    result = cv2.putText(result, reg_plate, (int(x0), int(y0-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            _, im_arr = cv2.imencode('.jpg', result)  # im_arr: image in Numpy one-dim array format.
            im_bytes = im_arr.tobytes()
            im_b64 = base64.b64encode(im_bytes)

            return {
                "error": False,
                "message": "Success",
                "data" : im_b64.decode()
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
            cap = cv2.VideoCapture("inference/output/response.mp4")
            l = []
            for idx in df.index:
                frame = df.loc[idx, "frame_idx"]
                speed = df.loc[idx, "mean_speed"]
                object_id = df.loc[idx, "object_id"]
                x = df.loc[idx, "left"]
                y = df.loc[idx, "top"]
                w = df.loc[idx, "width"]
                h = df.loc[idx, "height"]
                cap.set(cv2.CAP_PROP_POS_FRAMES,frame)
                ret, frame = cap.read()
                crop_img = frame[x:x+h, y:y+w]
                img = Image.fromarray(crop_img)
                im_file = BytesIO()
                img.save(im_file, format="JPEG")
                im_bytes = im_file.getvalue()  # im_bytes: image in binary format.
                im_b64 = base64.b64encode(im_bytes)


                l.append({
                    "image" : im_b64.decode(),
                    "speed" : speed,
                })
            item = {
                "vid_id" : vid_id,
                "list" : l,
                "count" : len(l)
            }
            collection = MongoClient(URI).main.log
            collection.insert_one(item)
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
        