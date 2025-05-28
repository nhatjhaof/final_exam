import time
import cv2
import pandas as pd
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
import base64
import os

try:
    model = YOLO('yolo11n.pt')
    cap = cv2.VideoCapture('Licence.mp4')
    if not cap.isOpened():
        raise RuntimeError("Không mở được Licence.mp4")

    class_list = model.names

    # Mouse callback (nếu muốn)
    def RGB(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            print([x, y])
    cv2.namedWindow('RGB')
    cv2.setMouseCallback('RGB', RGB)

    # Đếm frame và fps
    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Vị trí các vạch
    line_y_red    = 102
    line_y_blue   = 333
    line_y_yellow = 148

    # Theo dõi crossing và đếm
    crossed_red    = set()
    crossed_yellow = set()
    crossed_blue   = set()
    count_down     = defaultdict(int)
    count_up       = defaultdict(int)

    #Vùng lưu id ảnh xe
    save_blue_image = set()
    
    # Ngưỡng cho phép so với vạch xanh
    capture_threshold = 5

    # Lưu frame khi qua vạch để tính tốc độ
    speed_dict = dict()  # sẽ mapping: track_id -> {"yellow_frame":…, "red_frame":…}

    vehicle_classes = ["car", "truck", "bus", "motorbike"]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame = cv2.resize(frame, (1020, 500))

        results = model.track(frame, persist=True)

        # Lọc boxes, ids, class_indices, confidences
        boxes, track_ids, class_indices, confidences = [], [], [], []
        if results and results[0].boxes is not None and results[0].boxes.data is not None:
            for i in range(len(results[0].boxes.cls)):
                cls_idx  = int(results[0].boxes.cls[i].item())
                cls_name = class_list[cls_idx]
                if cls_name in vehicle_classes:
                    boxes.append(results[0].boxes.xyxy[i].cpu().tolist())
                    track_ids.append(int(results[0].boxes.id[i].item()) if results[0].boxes.id is not None else -1)
                    class_indices.append(cls_idx)
                    confidences.append(float(results[0].boxes.conf[i].item()))

        # Vẽ các vạch
        cv2.line(frame, (25, line_y_red),    (789, line_y_red),    (0, 0, 255),   3)
        cv2.putText(frame, 'Red Line',    (106, line_y_red - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        cv2.line(frame, (25, line_y_blue),  (857, line_y_blue),  (255,0,0),     3)
        cv2.putText(frame, 'Blue Line',   (106, line_y_blue - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        cv2.line(frame, (25, line_y_yellow),(830, line_y_yellow),(0,255,255),   3)
        cv2.putText(frame, 'Yellow Line', (104, line_y_yellow - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

        # Xử lý từng object
        for box, track_id, cls_idx in zip(boxes, track_ids, class_indices):
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2)//2, (y1 + y2)//2
            cls_name = class_list[cls_idx]

            # Khởi tạo entry trong speed_dict nếu chưa có
            if track_id not in speed_dict:
                speed_dict[track_id] = {"yellow_frame": None, "red_frame": None}

            # Vẽ bbox, tâm và ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.circle(frame, (cx, cy), 4, (0,0,255), -1)
            cv2.putText(frame, f"ID:{track_id} {cls_name}", (x1, y1-10),
                        cv2.FONT_HERSHEY_COMPLEX, 0.6, (0,255,255), 2)

            # Khi xe lần đầu qua vạch xanh
            if y1 <= line_y_blue <= y2 and track_id not in crossed_blue:
                crossed_blue.add(track_id)
            # Đếm xuống khi tâm gần vạch xanh
            if track_id in crossed_blue and abs(cy - line_y_blue) <= 5:
                count_down[cls_name] += 1

            # Khi xe qua vạch đỏ lần đầu
            if y1 <= line_y_red <= y2 and track_id not in crossed_red:
                crossed_red.add(track_id)
            # Ghi frame cắt vạch đỏ
            if y1 <= line_y_red <= y2 and speed_dict[track_id]["red_frame"] is None:
                speed_dict[track_id]["red_frame"] = frame_count

            # Khi xe qua vạch vàng lần đầu
            if y1 <= line_y_yellow <= y2 and track_id not in crossed_yellow:
                crossed_yellow.add(track_id)
            # Ghi frame cắt vạch vàng
            if y1 <= line_y_yellow <= y2 and speed_dict[track_id]["yellow_frame"] is None:
                speed_dict[track_id]["yellow_frame"] = frame_count

            # Tính tốc độ khi đã có cả hai lần cắt
            red_f   = speed_dict[track_id]["red_frame"]
            yellow_f= speed_dict[track_id]["yellow_frame"]
            if red_f is not None and yellow_f is not None:
                frame_diff = abs(yellow_f - red_f)
                if (frame_diff > 0):
                    real_distance_met = 5
                    time_taken = (frame_diff / fps) 
                    speed = (real_distance_met / time_taken) * 3.6  # m/s → km/h
                    
                    #Cảnh báo tốc độ
                    if (speed > 60):
                        warning_text = "Speed limit exceeded!"
                        cv2.putText(frame,warning_text, (x1, y1 - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    #Hiển thị tốc độ
                    speed_text = f"{speed:.1f} km/h"
                    cv2.putText(frame, speed_text, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        #Chụp lại hình ảnh xe
            if (track_id not in save_blue_image and abs(cy - line_y_blue) <= capture_threshold):
                
                car_image = frame[y1:y2, x1:x2]
                resized_car_image = cv2.resize(car_image, (500, 500))
                
                fileName = f"vehicle{track_id}_f{frame_count}_f{cls_name}.jpg" 
                filePath = f"./Capture/{fileName}"
        #Tạo thư mục capture nếu chưa dc khởi tạo
                os.makedirs("Capture", exist_ok=True)
        # Lưu ảnh
                cv2.imwrite(filePath, resized_car_image)
        #đánh dấu id đã được lưu

                save_blue_image.add(track_id)
        # Vẽ số liệu đếm xe đi xuống
        y_offset = 30
        for cls, cnt in count_down.items():
            cv2.putText(frame,
                        f'{cls} (Down): {cnt}',
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0,255,0), 2, cv2.LINE_AA)
            y_offset += 30

        cv2.imshow('RGB', frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Nhấn ESC để thoát
            break

finally:
    print("Đang dọn dẹp tài nguyên...")
    try:
        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Lỗi khi dọn dẹp: {e}")
