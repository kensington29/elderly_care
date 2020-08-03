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


def align_face(face_frame, landmarks):
   
    left_eye, right_eye, tip_of_nose, left_lip, right_lip = landmarks
    
    # compute the angle between the eye centroids
    dy = right_eye[1] - left_eye[1]     # right eye, left eye Y
    dx = right_eye[0] - left_eye[0]  # right eye, left eye X
    angle = np.arctan2(dy, dx) * 180 / np.pi
    # compute center (x, y)-coordinates (i.e., the median point)
    # between the two eyes in the input image
    ##eyes_center = ((right_eye[0] + left_eye[0]) // 2, (right_eye[1] + left_eye[1]) // 2)
    
    ## center of face_frame
    center = (face_frame.shape[0] // 2, face_frame.shape[1] // 2)
    h, w, c = face_frame.shape
    
    # grab the rotation matrix for rotating and scaling the face
    M = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    aligned_face = cv2.warpAffine(face_frame, M, (w, h))
    
    return aligned_face



# plot setting
rows = 6
columns = 6
plt.rcParams['figure.figsize'] = (15.0, 15.0)


# Read Image
image = "img_reged_person/hitoshi.jpg"
#image = "img_reged_person/cherry.jpg"
#image = "img_reged_person/mari_1.jpg"
#image = "img_reged_person/mari_2.jpg"
#image = "img_reged_person/banqet.jpg"


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

# Get facial landmarks to align faces
# 1.Read IR
# model_xml = fp_path + "FP16/landmarks-regression-retail-0009.xml"
#model_xml = fp_path + "facial-landmarks-35-adas-0001.xml"
# model_bin = os.path.splitext(model_xml)[0] + ".bin"
net = IENetwork(model="FP16/landmarks-regression-retail-0009.xml", weights="FP16/landmarks-regression-retail-0009.bin")


# 2. Configure input & putput
input_blob = next(iter(net.inputs))
out_blob = next(iter(net.outputs))
n, c, h, w = net.inputs[input_blob].shape
print("input:{}\noutput:{}".format(net.inputs,net.outputs))
print("input.shape:{}\noutput.shape:{}".format(
    net.inputs[input_blob].shape, net.outputs[out_blob].shape))

# 3. Load Model
exec_net = plugin.load(network=net, num_requests=2)

# 4. Create Async Request

# create ndarray (face_count, 5, 2) : shape of landmarks matrix (5, 2) 

facial_landmarks = np.zeros((faces.shape[2], 5, 2)) 

for face_id, face_frame in enumerate(face_frames):
    in_frame = cv2.resize(face_frame, (w, h))
    in_frame = in_frame.transpose((2, 0, 1))
    in_frame = in_frame.reshape((n, c, h, w))
    exec_net.infer(inputs={input_blob: in_frame})

    # 5. Get Response
    res = exec_net.requests[0].outputs[out_blob].reshape(1, 10)[0]

    # 6. draw Response
    lm_face = face_frame.copy()
    for i in range(res.size // 2):
        normed_x = res[2 * i]
        normed_y = res[2 * i + 1]
        x_lm = lm_face.shape[1] * normed_x
        y_lm = lm_face.shape[0] * normed_y
        cv2.circle(lm_face, (int(x_lm), int(y_lm)), 1 + int(0.03 * lm_face.shape[1]), (255, 255, 0), -1)
        # save landmarks
        facial_landmarks[face_id][i] = (x_lm, y_lm)
        ax = plt.subplot(rows, columns, face_id + 1)
        ax.set_title("face_id:{}".format(face_id))
    plt.imshow(lm_face)
plt.show()



# display aligned faces
aligned_faces = []

# display input faces
plt.figure()
for face_id, face_frame in enumerate(face_frames):
    ax = plt.subplot(rows, columns, face_id + 1)
    ax.set_title("before face:{}".format(face_id))
    plt.imshow(face_frame)
plt.show()

# display output faces
plt.figure()
for face_id, face_frame in enumerate(face_frames):
    aligned_face = face_frame.copy()
    aligned_face = align_face(aligned_face, facial_landmarks[face_id])
    aligned_faces.append(aligned_face)
    ax = plt.subplot(rows, columns, face_id + 1)
    ax.set_title("after face:{}".format(face_id))
    plt.imshow(aligned_face)
plt.show()