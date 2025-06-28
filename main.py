import os
import cv2
import time
import mysql.connector
from dotenv import load_dotenv
from ultralytics import YOLO
from collections import defaultdict
import base64
from detech_plate import process_and_save_plate 

## ssh -L 58763:127.0.0.1:3306 root@192.168.70.128

# Load biến môi trường từ file .env
load_dotenv()

DB_HOST = os.getenv('REMOTE_DB_HOST')
DB_PORT = int(os.getenv('REMOTE_DB_PORT'))
DB_USER = os.getenv('DB_USERNAME')
DB_PASS = os.getenv('DB_PASSWORD')
DB_NAME = os.getenv('DB_NAME')

print("→ Kết nối đến MySQL qua tunnel...")
conn = mysql.connector.connect(
    host=DB_HOST,
    port=DB_PORT,
    user=DB_USER,
    password=DB_PASS,
    database=DB_NAME,
    connection_timeout=5
)
cursor = conn.cursor()
print("✅ Đã kết nối tới MySQL")

try:
    model = YOLO('yolo11n.pt')
    cap = cv2.VideoCapture('License.mp4')
    if not cap.isOpened():
        raise RuntimeError("Không mở được License.mp4")

    class_list = model.names
    cv2.namedWindow('Camera vehical')

    def Camera_vehical(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            print([x, y])

    cv2.setMouseCallback('Camera vehical', Camera_vehical)

    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)

    line_y_red    = 102
    line_y_blue   = 333
    line_y_yellow = 148

    crossed_red    = set()
    crossed_yellow = set()
    crossed_blue   = set()
    count_down     = defaultdict(int)
    count_up       = defaultdict(int)
    save_blue_image = set()
    capture_threshold = 5

    speed_dict = dict()
    vehicle_classes = ["car", "truck", "bus", "motorbike"]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame = cv2.resize(frame, (1020, 500))
        results = model.track(frame, persist=True)

        boxes, track_ids, class_indices = [], [], []
        if results and results[0].boxes is not None and results[0].boxes.data is not None:
            for i in range(len(results[0].boxes.cls)):
                cls_idx  = int(results[0].boxes.cls[i].item())
                cls_name = class_list[cls_idx]
                if cls_name in vehicle_classes:
                    boxes.append(results[0].boxes.xyxy[i].cpu().tolist())
                    track_ids.append(int(results[0].boxes.id[i].item()) if results[0].boxes.id is not None else -1)
                    class_indices.append(cls_idx)

        cv2.line(frame, (25, line_y_red),    (789, line_y_red),    (0, 0, 255),   3)
        cv2.line(frame, (25, line_y_blue),   (857, line_y_blue),   (255, 0, 0),   3)
        cv2.line(frame, (25, line_y_yellow), (830, line_y_yellow), (0, 255, 255), 3)

        for box, track_id, cls_idx in zip(boxes, track_ids, class_indices):
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2)//2, (y1 + y2)//2
            cls_name = class_list[cls_idx]

            if track_id not in speed_dict:
                speed_dict[track_id] = {"yellow_frame": None, "red_frame": None, "speed": None}

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.circle(frame, (cx, cy), 4, (0,0,255), -1)
            cv2.putText(frame, f"ID:{track_id} {cls_name}", (x1, y1-10),
                        cv2.FONT_HERSHEY_COMPLEX, 0.6, (0,255,255), 2)

            if y1 <= line_y_blue <= y2 and track_id not in crossed_blue:
                crossed_blue.add(track_id)
            if track_id in crossed_blue and abs(cy - line_y_blue) <= 5:
                count_down[cls_name] += 1

            if y1 <= line_y_red <= y2 and track_id not in crossed_red:
                crossed_red.add(track_id)
            if y1 <= line_y_red <= y2 and speed_dict[track_id]["red_frame"] is None:
                speed_dict[track_id]["red_frame"] = frame_count

            if y1 <= line_y_yellow <= y2 and speed_dict[track_id]["yellow_frame"] is None:
                speed_dict[track_id]["yellow_frame"] = frame_count
            
            if y1 <= line_y_yellow <= y2 and track_id not in crossed_yellow:
                crossed_yellow.add(track_id)
            
            red_f = speed_dict[track_id]["red_frame"]
            yellow_f = speed_dict[track_id]["yellow_frame"]

            if red_f is not None and yellow_f is not None and speed_dict[track_id]["speed"] is None:
                frame_diff = abs(yellow_f - red_f)
                if frame_diff > 0:
                    real_distance_met = 7
                    time_taken = frame_diff / fps
                    speed = (real_distance_met / time_taken) * 3.6
                    speed_dict[track_id]["speed"] = speed

            # Luôn vẽ tốc độ nếu đã có
            speed = speed_dict[track_id].get("speed")
            if speed:
                speed_text = f"{speed:.1f} km/h"
                cv2.putText(frame, speed_text, (x1, y1 - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            if track_id not in save_blue_image and abs(cy - line_y_blue) <= capture_threshold and speed > 40:
                car_image = frame[y1:y2, x1:x2]
                resized_car_image = cv2.resize(car_image, (500, 500))
                fileName = f"vehicle{track_id}_f{frame_count}_{cls_name}.jpg"
                filePath = f"./Capture/{fileName}"
                os.makedirs("Capture", exist_ok=True)
                cv2.imwrite(filePath, resized_car_image)
                save_blue_image.add(track_id)

                # Gọi hàm xử lý biển số
                detected_plate, plate_roi = process_and_save_plate(resized_car_image, conn, cursor)
                if detected_plate and plate_roi is not None:
                    cv2.imwrite(f"Capture/plate_{frame_count}.jpg", plate_roi)

                try:
                    with open(filePath, "rb") as f:
                        binary_data = f.read()
                        encoded_data = base64.b64encode(binary_data).decode('utf-8')
                    cursor.execute(
                        "INSERT INTO vehicle_images (image_data, capture_time, speed) VALUES (%s, NOW(), %s)",
                        (encoded_data, speed)
                    )
                    conn.commit()    
                except Exception as e:
                    print(f"❌ Lỗi lưu ảnh vào DB: {e}")
                    # Gửi thông báo khi xe vi phạm tốc độ
    
                    


        y_offset = 30
        for cls, cnt in count_down.items():
            cv2.putText(frame,
                        f'{cls} (Down): {cnt}',
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0,255,0), 2, cv2.LINE_AA)
            y_offset += 30

        cv2.imshow('Camera vehical', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    print("Đang dọn dẹp tài nguyên...")
    try:
        cap.release()
        cv2.destroyAllWindows()
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals() and conn.is_connected():
            conn.close()
            print("→ Đã đóng kết nối MySQL.")
    except Exception as e:
        print(f"Lỗi khi dọn dẹp: {e}")
