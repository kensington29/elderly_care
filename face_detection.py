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



# plot setting
rows = 6
columns = 6
plt.rcParams['figure.figsize'] = (15.0, 15.0)


# Read Image
image = "img_reged_person/hitoshi.jpg"

# create image
#init_frame, frame = plot_image.plot_image(image)
frame = get_frame(image)
frame_h, frame_w = frame.shape[:2]
init_frame = frame.copy()

print("frame_h, frame_w:{}".format(frame.shape[:2]))
plt.figure(figsize=(8, 8))
plt.imshow(frame)
plt.show()
 
# ターゲットデバイスの指定 
plugin = IEPlugin(device="MYRIAD")
 
# モデルの読み込み 
# net = IENetwork(model='FP16/face-detection-retail-0004.xml', weights='FP16/face-detection-retail-0004.bin')
# 顔検出モデル'face-detection-adas-0001'を使用
net = IENetwork(model='FP16/face-detection-adas-0001.xml', weights='FP16/face-detection-adas-0001.bin')

# 3. Configure input & output
input_blob = next(iter(net.inputs))
out_blob = next(iter(net.outputs))
n, c, h, w = net.inputs[input_blob].shape
print("input:{}\noutput:{}".format(net.inputs,net.outputs))
print("input.shape:{}\noutput.shape:{}".format(
    net.inputs[input_blob].shape, net.outputs[out_blob].shape))

# 4. Load Model
exec_net = plugin.load(network=net, num_requests=2)

# 5. Create Async Request
in_frame = cv2.resize(frame, (w, h))
in_frame = in_frame.transpose((2, 0, 1))
in_frame = in_frame.reshape((n, c, h, w))
exec_net.start_async(request_id=0, inputs={input_blob: in_frame}) # res's shape: [1, 1, 200, 7]

# 6. Receive Async Request
if exec_net.requests[0].wait(-1) == 0:
    res = exec_net.requests[0].outputs[out_blob]
    # prob threshold : 0.5
    faces = res[0][:, np.where(res[0][0][:, 2] > 0.5)]

# 7. draw faces
frame = init_frame.copy()
face_frames = []

for face_id, face in enumerate(faces[0][0]):
    box = face[3:7] * np.array([frame_w, frame_h, frame_w, frame_h])
    (xmin, ymin, xmax, ymax) = box.astype("int")
    face_frame = frame[ymin:ymax, xmin:xmax]
    face_frames.append(face_frame)
    ax = plt.subplot(rows, columns, face_id + 1)
    ax.set_title("face_id:{}".format(face_id))
    plt.imshow(face_frame)
    face_id += 1
plt.show()