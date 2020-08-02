# Open cvのインポート　（cv2はバージョンが２という意味ではない）
import cv2
# numpyのインポート
import numpy as np
 
# モジュール読み込み (sys:Pythonのインタプリタや実行環境に関する情報を扱うためのモジュール)
import sys
#sys.path.append('/opt/intel/openvino/python/python3.5/armv7l')
from openvino.inference_engine import IENetwork, IEPlugin
 
# ターゲットデバイスの指定 
plugin = IEPlugin(device="MYRIAD")
 
# モデルの読み込み 
# net = IENetwork(model='FP16/face-detection-retail-0004.xml', weights='FP16/face-detection-retail-0004.bin')
# 顔検出モデル'face-detection-adas-0001'を使用
net = IENetwork(model='FP16/face-detection-adas-0001.xml', weights='FP16/face-detection-adas-0001.bin')
# exec_net = plugin.load(network=net)

# Configure input & output
input_blob = next(iter(net.inputs))
out_blob = next(iter(net.outputs))
n, c, h, w = net.inputs[input_blob].shape

# Load Model
exec_net = plugin.load(network=net, num_requests=2)
 
# カメラ準備 
cap = cv2.VideoCapture(0)
 
# メインループ 
while True:
    # video frameを1 frameずつ取得する
    # video frameを取得できなかった場合は、ret=False
    # 取得したvideo frameは'frame'に入る
    ret, frame = cap.read()
 
    # Reload on error 
    if ret == False:
        continue
 

    # 入力データフォーマットへ変換 
    # img = cv2.resize(frame, (300, 300))   # サイズ変更 
    # img = img.transpose((2, 0, 1))    # HWC > CHW 
    # img = np.expand_dims(img, axis=0) # 次元合せ 
 
    # 推論実行 
    # out = exec_net.infer(inputs={'data': img})

    
    # Create Async Request
    in_frame = cv2.resize(frame, (w, h))
    in_frame = in_frame.transpose((2, 0, 1))
    in_frame = in_frame.reshape((n, c, h, w))
    exec_net.start_async(request_id=0, inputs={input_blob: in_frame}) 
    # res's shape: [1, 1, 200, 7]
 
    # 出力から必要なデータのみ取り出し 
    out = out['detection_out']
    out = np.squeeze(out) #サイズ1の次元を全て削除 
 
    # 検出されたすべての顔領域に対して１つずつ処理 
    for detection in out:
        # conf値の取得 
        confidence = float(detection[2])
 
        # バウンディングボックス座標を入力画像のスケールに変換 
        xmin = int(detection[3] * frame.shape[1])
        ymin = int(detection[4] * frame.shape[0])
        xmax = int(detection[5] * frame.shape[1])
        ymax = int(detection[6] * frame.shape[0])
 
        # conf値が0.5より大きい場合のみバウンディングボックス表示 
        if confidence > 0.5:
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color=(240, 180, 0), thickness=3)
 
    # 画像表示 
    cv2.imshow('frame', frame)
 
    # 何らかのキーが押されたら終了 
    key = cv2.waitKey(1)
    if key != -1:
        break
 
# 終了処理 
cap.release()
cv2.destroyAllWindows()