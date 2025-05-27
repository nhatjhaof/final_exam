import time
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import base64

try:
    model = YOLO('yolo11n.pt')
    cap = cv2.VideoCapture('camera.mp4')
    class_list = model.names

    # Tạo biến để theo dõi ID đã lưu 
    saved_vehicle_ids = set()
    # Xử lý hình ảnh được cắt từ video từ bytes thành base64 string lưu vào dbdb
    def save_image_to_db(image, speed, vehicle_id):
        # Kiểm tra nếu ID này đã được lưu
        if vehicle_id in saved_vehicle_ids:
            return
        try:
             # Chuyển ảnh OpenCV sang định dạng bytes
            is_success, im_buf_arr = cv2.imencode(".jpg", image)
            byte_im = im_buf_arr.tobytes()
            
            # Chuyển bytes thành base64 string
            image_base64 = base64.b64encode(byte_im).decode('utf-8')

            # Thêm ID vào set đã lưu
            saved_vehicle_ids.add(vehicle_id)
            print(f"Đã lưu xe ID {vehicle_id} với tốc độ {speed} km/h")
        except Exception as e:
            print(f"Lỗi khi lưu ảnh: {e}")

            #Xác định tọa độ con trỏ chuột để vẽ cái đường line xác địnhđịnh
    def RGB(event, x, y, flags, param):
        if  event == cv2.EVENT_MOUSEMOVE :  
            colorsBGR = [x, y]
            print(colorsBGR)
    cv2.namedWindow('RGB')
    cv2.setMouseCallback('RGB', RGB)


    #đếm số frame tính fps 
    frame_count = 0
    #fps thuc te cua video
    fps = cap.get(cv2.CAP_PROP_FPS)
    # vị trí cố định của đường line theo trục oxy trong frame hình 
    line_y_red = 102  # Red line position
    line_y_blue = 333  # Blue line position
    line_y_yellow = 148  # Yellow line position

    # Theo dõi xe cán qua vạch 
    crossed_red = set()
    crossed_yellow = set()
    crossed_blue = set()
    
    # Lưu hình ảnh khi xe can qua vach xanh 
    save_blue_image = set()
    # Lặp từng khung hình videovideo
    while cap.isOpened():
        ret, frame = cap.read()
    
        if not ret:
            break
    frame_count += 1
    #điều chỉnh kích cỡ khung hìnhhình
    frame=cv2.resize(frame,(1020,500))    
    
    # chạy model theo dõi của yolo 
    results = model.track(frame, persist=True)

    vehicle_classes = ["car", "truck", "bus", "motorbike"]

    boxes = []
    track_ids = []
    class_indices = []
    confidences = []
      # Lọc chỉ giữ lại các phương tiện giao thông
    boxes, track_ids, class_indices, confidences = [], [], [], []
    
    if results and len(results) > 0 and results[0].boxes is not None and results[0].boxes.data is not None:
        for i in range(len(results[0].boxes.cls)):
            class_index = int(results[0].boxes.cls[i].item())  # Lấy index lớp
            class_name = class_list[class_index]  # Lấy tên lớp

            if class_name in vehicle_classes:  # Nếu thuộc nhóm xe cộ, mới thêm vào danh sách
                boxes.append(results[0].boxes.xyxy[i].cpu().tolist())
                track_ids.append(int(results[0].boxes.id[i].item()) if results[0].boxes.id is not None else -1)
                class_indices.append(class_index)
                confidences.append(float(results[0].boxes.conf[i].item()))
        # Vẽ đường line trên khung hình
    cv2.line(frame, (25, line_y_red), (789, line_y_red), (0, 0, 255), 3)
    cv2.putText(frame, 'Red Line', (106, line_y_red - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.line(frame, (70, line_y_blue), (857, line_y_blue), (255, 0, 0), 3)
    cv2.putText(frame, 'Blue Line', (106, line_y_blue - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.line(frame, (25, line_y_yellow), (830, line_y_yellow), (0, 255, 255), 3)
    cv2.putText(frame, 'Yellow Line', (104, line_y_yellow - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

finally:
    print("Đang dọn dẹp tài nguyên...") 
    try:
        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Lỗi khi dọn dẹp: {e}")