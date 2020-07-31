import cv2
import numpy as np
 
# モジュール読み込み 
import sys
#sys.path.append('/opt/intel/openvino/python/python3.5/armv7l')
from openvino.inference_engine import IENetwork, IEPlugin
 
# ターゲットデバイスの指定 
plugin = IEPlugin(device="MYRIAD")
 
# モデルの読み込み 
net = IENetwork(model='FP16/face-detection-retail-0004.xml', weights='FP16/face-detection-retail-0004.bin')
exec_net = plugin.load(network=net)
 
# カメラ準備 
cap = cv2.VideoCapture(0)
 
# メインループ 
while True:
    ret, frame = cap.read()
 
    # Reload on error 
    if ret == False:
        continue
 
    # 入力データフォーマットへ変換 
    img = cv2.resize(frame, (300, 300))   # サイズ変更 
    img = img.transpose((2, 0, 1))    # HWC > CHW 
    img = np.expand_dims(img, axis=0) # 次元合せ 
 
    # 推論実行 
    out = exec_net.infer(inputs={'data': img})
 
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