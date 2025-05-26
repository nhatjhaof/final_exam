from collections import defaultdict
from ultralytics import YOLO
import cv2
import time
import os

#Xác định tọa độ con trỏ chuột 
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Load model YOLO
model_vehicle = YOLO("yolo11n.pt")
class_list = model_vehicle.names
#real distance
real_distance_met = 5
# Save time for each vehicle when crossing lines
speed_dict = {}

# Open the video file
cap = cv2.VideoCapture('camera.mp4')

#Frame count
frame_count = 0
#fps thuc te cua video
fps = cap.get(cv2.CAP_PROP_FPS)
# Define line positions for counting
line_y_red = 102  # Red line position
line_y_blue = 333  # Blue line position
line_y_yellow = 148  # Yellow line position

# Variables to store counting and tracking information
counted_ids_red_to_blue = set()
counted_ids_blue_to_red = set()

# Count objects for each direction
count_red_to_blue = defaultdict(int)  # Moving downwards
count_blue_to_red = defaultdict(int)  # Moving upwards

# Track which line was crossed 
crossed_red = set()
crossed_yellow = set()

#Save image
save_yellow_image = set()

# Loop through video frames
while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break
    frame_count += 1
    #Resize frame 
    frame=cv2.resize(frame,(1020,500))    
    
    # Run YOLO tracking
    results = model_vehicle.track(frame, persist=True)

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

    # Draw the lines on the frame
    cv2.line(frame, (25, line_y_red), (789, line_y_red), (0, 0, 255), 3)
    cv2.putText(frame, 'Red Line', (106, line_y_red - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.line(frame, (70, line_y_blue), (857, line_y_blue), (255, 0, 0), 3)
    cv2.putText(frame, 'Blue Line', (106, line_y_blue - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.line(frame, (25, line_y_yellow), (830, line_y_yellow), (0, 255, 255), 3)
    cv2.putText(frame, 'Yellow Line', (104, line_y_yellow - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Loop through each detected object
    for box, track_id, class_index, conf in zip(boxes, track_ids, class_indices, confidences):
        x1, y1, x2, y2 = map(int, box)

        bottom_y = y2
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        class_name = class_list[class_index]

        # Draw dot at the center and display the tracking ID and class name
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
        cv2.putText(frame, f"ID: {track_id} {class_name}", (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 255), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Initialize speed tracking if not exist
        if track_id not in speed_dict:
            speed_dict[track_id] = {"yellow_frame": None, "red_frame": None}

        # Check crossing of red line
        if y1 <= line_y_red <= y2 and track_id not in crossed_red:
            crossed_red.add(track_id)
        
        # Ghi lại frame khi cắt vạch
        if y1 <= line_y_red <= y2 and speed_dict[track_id]["red_frame"] is None:
            speed_dict[track_id]["red_frame"] = frame_count
        
        # Check crossing of yellow line
        if y1 <= line_y_yellow <= y2 and track_id not in crossed_yellow:
            crossed_yellow.add(track_id)
        # Ghi lại frame khi cắt vạch
        if y1 <= line_y_yellow <= y2 and speed_dict[track_id]["yellow_frame"] is None:
            speed_dict[track_id]["yellow_frame"] = frame_count


        #Save image when cross the yellow line
        if y1 <= line_y_yellow <= y2 and track_id not in save_yellow_image:
            fileName = f"vehicle{track_id}_f{frame_count}_f{class_name}.jpg" 
            filePath = f"./Capture/{fileName}"
        # Cắt bounding box từ ảnh gốc
            vehicle_crop = frame[y1:y2, x1:x2]

        # Tạo thư mục nếu chưa tồn tại
            import os
            os.makedirs("Capture", exist_ok=True)

        # Lưu ảnh
            cv2.imwrite(filePath, vehicle_crop)

        # Đánh dấu đã lưu ảnh để không lưu trùng
            save_yellow_image.add(track_id)

        # # Check crossing of blue line
        # if  abs(bottom_y - line_y_blue) <= 5:
        #     if track_id not in crossed_blue_first:
        #         crossed_blue_first[track_id] = True

        # # Counting logic for downward direction (red → blue)
        # if track_id in crossed_red and track_id not in counted_ids_red_to_blue:
        #     if abs(bottom_y - line_y_blue) <= 5:
        #         counted_ids_red_to_blue.add(track_id)
        #         count_red_to_blue[class_name] += 1

        # # Counting logic for upward direction (blue → red)
        # if track_id in crossed_blue_first and track_id not in counted_ids_blue_to_red:
        #     if line_y_red - 5 <= cy <= line_y_red + 5:
        #         counted_ids_blue_to_red.add(track_id)
        #         count_blue_to_red[class_name] += 1

        # Speed calculation for both directions
        if speed_dict[track_id]["red_frame"] is not None and speed_dict[track_id]["yellow_frame"] is not None:
            frame_diff = abs(speed_dict[track_id]["yellow_frame"] - speed_dict[track_id]["red_frame"])
            # frame_interval_inverse = 1 / time_taken # so frame xu ly trong 1s dung de tinh van toc 
            if(frame_diff > 0):
                time_taken = (frame_diff / fps) 
                speed = (real_distance_met / time_taken) * 3.6  # m/s → km/h
                speed_text = f"{speed:.1f} km/h"
                cv2.putText(frame, speed_text, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # elif speed_dict[track_id]["yellow_time"] and speed_dict[track_id]["red_time"]:
        #     time_taken = abs(speed_dict[track_id]["red_time"] - speed_dict[track_id]["yellow_time"])
        #     speed = (100 / time_taken) * 3.6  # Convert to km/h
        #     speed_text = f"{speed:.1f} km/h"
        #     cv2.putText(frame, speed_text, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Display counts on the frame
    y_offset = 30
    for class_name, count in count_red_to_blue.items():
        cv2.putText(frame, f'{class_name} (Down): {count}', (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        y_offset += 30

    # y_offset += 20  # Add spacing for upward counts
    # for class_name, count in count_blue_to_red.items():
    #     cv2.putText(frame, f'{class_name} (Up): {count}', (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    #     y_offset += 30

    # Show the frame
    cv2.imshow("RGB", frame)

    # Exit loop if 'ESC' is pressed
    if cv2.waitKey(0) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
