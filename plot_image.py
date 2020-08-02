import requests
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
import matplotlib.pyplot as plt
import re
import os
from openvino.inference_engine import IENetwork, IEPlugin

def resize_frame(frame, height):

    # resize image with keeping frame width
    scale = height / frame.shape[1]
    frame = cv2.resize(frame, dsize=None, fx=scale, fy=scale)

    return frame

def get_frame(image):
    
    resize_width = 640

    if re.match(r'http.?://', image):
        response = requests.get(image)
        frame = np.array(Image.open(BytesIO(response.content)))
    else:
        frame = cv2.imread(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if frame.shape[1] >= resize_width:
        frame = resize_frame(frame, resize_width)
    
    return frame

def plot_image(image):
    # plot setting
    rows = 6
    columns = 6

    # Read Image
    # image = "img_reged_person/hitoshi.jpg"

    frame = get_frame(image)
    frame_h, frame_w = frame.shape[:2]
    init_frame = frame.copy()

    print("frame_h, frame_w:{}".format(frame.shape[:2]))
    plt.figure(figsize=(8, 8))
    plt.imshow(frame)
    plt.show()

    return frame