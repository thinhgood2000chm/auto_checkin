# -*- coding: UTF-8 -*-
import base64
import io
import json
from typing import List

import numpy as np
import time
from numpy.linalg import norm

import requests
from skimage import transform as trans
import cv2
# import torch
import threading, os
from multiprocessing import Process
from PIL import Image
from starlette import status

# from Read_message_consumer import ReadMessageConsumer
# from glob_var import minio_connect
# from glob_var import GlobVar

from config import MODEL, FACE_DETECT
from mongodb import get_all_embedding, query_user_info
from scrfd import SCRFD
#
# minio_address, minio_address1, bucket_name, client = minio_connect()
weights = os.getcwd() + '/weights/last.pt'
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_size = 640
conf_thres = 0.6
iou_thres = 0.5
imgsz = (640, 640)
# model_face_detect = load_model(weights, device)

scrfd_model = SCRFD(MODEL[FACE_DETECT])
scrfd_model.prepare(0)
import threading


src1 = np.array([[51.642, 50.115], [57.617, 49.990], [35.740, 69.007],
                 [51.157, 89.050], [57.025, 89.702]],
                dtype=np.float32)
# <--left
src2 = np.array([[45.031, 50.118], [65.568, 50.872], [39.677, 68.111],
                 [45.177, 86.190], [64.246, 86.758]],
                dtype=np.float32)

# ---frontal
src3 = np.array([[39.730, 51.138], [72.270, 51.138], [56.000, 68.493],
                 [42.463, 87.010], [69.537, 87.010]],
                dtype=np.float32)

# -->right
src4 = np.array([[46.845, 50.872], [67.382, 50.118], [72.737, 68.111],
                 [48.167, 86.758], [67.236, 86.190]],
                dtype=np.float32)

# -->right profile
src5 = np.array([[54.796, 49.990], [60.771, 50.115], [76.673, 69.007],
                 [55.388, 89.702], [61.257, 89.050]],
                dtype=np.float32)

src = np.array([src1, src2, src3, src4, src5])
src_map = {112: src, 224: src * 2}

arcface_src = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)

arcface_src = np.expand_dims(arcface_src, axis=0)
def convert_image_to_base64(image):
    buff = io.BytesIO()
    image = Image.fromarray(np.uint8(image))
    image.save(buff, format="JPEG")
    base64_string = base64.b64encode(buff.getvalue()).decode()
    return base64_string

def estimate_norm(lmk, image_size=112, mode='arcface'):
    assert lmk.shape == (5, 2)
    tform = trans.SimilarityTransform()
    lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
    min_M = []
    min_index = []
    min_error = float('inf')
    if mode == 'arcface':
        if image_size == 112:
            src = arcface_src
        else:
            src = float(image_size) / 112 * arcface_src
    else:
        src = src_map[image_size]
    for i in np.arange(src.shape[0]):
        tform.estimate(lmk, src[i])
        M = tform.params[0:2, :]
        results = np.dot(M, lmk_tran.T)
        results = results.T
        error = np.sum(np.sqrt(np.sum((results - src[i])**2, axis=1)))
        if error < min_error:
            min_error = error
            min_M = M
            min_index = i
    return min_M, min_index


def norm_crop(img, landmark, image_size=112, mode='arcface'):
    M, pose_index = estimate_norm(landmark, image_size, mode)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped

def convert_base64_to_image(base64_string: str):
    buffer = io.BytesIO(base64.b64decode(base64_string))
    image = Image.open(buffer)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image

def crop_face(face_data: dict, raw_image):
    try:
        # 0: xmin, 1: ymin, 2: xmax, 3: ymax
        image = raw_image
        image_in_type_array = np.array(image)
        right_eye = [float(face_data['right_eye']["x"]), float(face_data['right_eye']["y"])]
        left_eye = [float(face_data['left_eye']["x"]), float(face_data['left_eye']["y"])]
        nose = [float(face_data['nose']["x"]), float(face_data['nose']["y"])]
        mouth_right = [float(face_data['mouth_right']["x"]), float(face_data['mouth_right']["y"])]
        mouth_left = [float(face_data['mouth_left']["x"]), float(face_data['mouth_left']["y"])]
        kps = np.array([right_eye, left_eye, nose, mouth_right, mouth_left])
        face = norm_crop(image_in_type_array, landmark=kps, image_size=112)
        return face
    except (Exception,):
        return None


def find_cosine_distance(source_embedding, target_embedding):
    source_embedding = np.array(source_embedding).ravel()
    target_embedding = np.array(target_embedding).ravel()
    sim = np.dot(source_embedding, target_embedding) / (norm(source_embedding) * norm(target_embedding))
    return sim



def face_embedding(base64_images: List[str]):
        try:
            data = None
            response = requests.post(
                url=f'http://127.0.0.1:8082/embedding',
                json={
                    "data": base64_images
                },
                timeout=300
            )
            response_data = response.json()
            if response.status_code == status.HTTP_200_OK:
                data = response_data["data"]

            return data
        except (Exception, ):
            return None



def checkin_user(employee_code):
        try:
            data = None
            response = requests.post(
                url=f'http://127.0.0.1:9005/sworker/seatech/{employee_code}',
                json={
                    "lat": "10.793126",
                    "long": "106.6900929",
                    "app": "oncheck",
                    "type": "CHECKIN"
                },
                timeout=300
            )
            response_data = response.json()
            if response.status_code == status.HTTP_200_OK:
                return response

            return data
        except (Exception, ):
            return None

def verify_image(source_embedding, target_embedding, threshold=0.32):
    max_distance_verify = 1
    if threshold > max_distance_verify:
        threshold = max_distance_verify
    distance = find_cosine_distance(source_embedding, target_embedding)
    if distance > threshold:
        return True, distance
    else:
        return False, distance

def checkin(data_face_detect, frame):
    list_face_after_crop = []
    for faces_data in data_face_detect:
        faces_data = sorted(
            faces_data, key=lambda x: x["score"], reverse=True
        )
        # hiện tại chỉ cho phép 1 khuôn mặt trong 1 ảnh => ảnh detect ra 2 khuôn măt sẽ lấy khuôn mặt có score cao nhất
        # score thấp có thể là detect sai
        face_data = faces_data[0]
        face_after_crop = crop_face(face_data, frame)
        if not face_after_crop.any():
            print("error dont have face")
        else:

            img_after_crop_base64_string = convert_image_to_base64(
                face_after_crop
            )
            list_face_after_crop.append(img_after_crop_base64_string)

            data_embedding_for_verify = face_embedding(list_face_after_crop)

            if not data_embedding_for_verify:
                print("error when embedding face")

            all_data_embedding = get_all_embedding(
                corpcode="seatech"
            )

            all_data_embedding = list(all_data_embedding)
            for index, data_image in enumerate(data_embedding_for_verify):
                data_image = data_image[0]
                list_user_info = []
                for data_embedding in all_data_embedding:
                    for data_image_face in data_embedding["embedding"]:
                        data_image_in_db = data_image_face[0]
                        is_match, distance = verify_image(
                            data_image,
                            data_image_in_db,
                            0.32
                        )
                        if is_match:
                            user_info = query_user_info(
                                corp_code=data_embedding["corp_code"],
                                user_code=data_embedding["user_code"],
                            )
                            if user_info:
                                user_info["dob"] = str(user_info["dob"])
                                user_info["distance"] = distance
                                list_user_info.append(user_info)
                            break
                list_user_info = sorted(
                    list_user_info,
                    key=lambda x: x["distance"],
                    reverse=True,
                )
                is_checkin_success = checkin_user(list_user_info[0]["user_code"])
                if is_checkin_success:
                    cv2.putText(frame, "cham cong thanh cong", (7, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 0),
                                1,
                                cv2.LINE_AA)

                    frame = cv2.resize(frame, (400, 400))
                    cv2.imshow(f"{list_user_info[0]['user_code']}", frame)
                    cv2.waitKey(3000)
                # data_classify = {
                #         "users": list_user_info
                #         if list_user_info
                #         else None,
                #     }

    return "success"

catch_face = []

if __name__ == '__main__':
    frame_count = 0
    vid = cv2.VideoCapture(0)
    # URL = "http://192.168.6.82:8080/video"
    #
    # vid = cv2.VideoCapture(URL)
    error_frame = 0
    prev_frame_time = 0
    time_for_take_capture = 0
    check_face = False
    while True:
        timer = cv2.getTickCount()
        try:
            new_frame_time = time.time()

            ret, frame = vid.read()
            frame = cv2.resize(frame, (224, 224))
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time

            # converting the fps into integer
            fps = int(fps)

            # converting the fps to string so that we can display it on frame
            # by using putText function
            fps = str(fps)

            # putting the FPS count on the frame
            cv2.putText(frame, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX , 3, (100, 255, 0), 3, cv2.LINE_AA)
            list_response = []
            results = scrfd_model.detect_faces(np.array(frame))
            if not results:
                time_for_take_capture = 0
                check_face = False
            if results:
                if check_face is False:
                    time_for_take_capture = time.time()
                    check_face = True

                # đếm giây, nếu bỏ mặt ra ngoài đếm lại từ đầu
                cv2.putText(frame, str(int(time.time() - time_for_take_capture)) if check_face else 0, (7, 150), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (100, 255, 0), 1, cv2.LINE_AA)

                if check_face and time.time() - time_for_take_capture >= 5:

                    time_for_take_capture = 0
                    check_face = False
                    list_response.append(results)
                    p = Process(target=checkin, args=(list_response, frame))
                    # t1 = threading.Thread(target=checkin, args=(list_response, frame))
                    p.start()

                frame = cv2.rectangle(frame, (int(float(results[0]["facial_area"]['0'])), int(float(results[0]["facial_area"]['1']))), (int(float(results[0]["facial_area"]['2'])),int(float(results[0]["facial_area"]['3']))), (255,0,0), 2)
            cv2.imshow('frame', frame)
            key=cv2.waitKey(1)

            #

        except Exception as error:
            print("Error:", error)
            time.sleep(1)
            continue
    vid.release()
    cv2.destroyAllWindows()
    # time.sleep(0.01)