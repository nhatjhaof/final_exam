import cv2  # Thư viện xử lý ảnh
import pytesseract  # Thư viện OCR để nhận diện ký tự trong ảnh
import numpy as np  # Thư viện xử lý dữ liệu số
import base64  # Thư viện mã hóa/giải mã base64
from ultralytics import YOLO  # Thư viện YOLO để phát hiện đối tượng

# Đường dẫn đến Tesseract-OCR (cần cài đặt Tesseract trước)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load mô hình YOLO đã được train cho nhận diện biển số
model = YOLO('best.pt')
# Hàm cải thiện nhận diện biển số
def improved_recognize_license_plate(image):
    try:
        # Sử dụng YOLO để nhận diện biển số xe với ngưỡng confidence là 0.5
        results = model.predict(image, conf=0.5)  # Giảm confidence để phát hiện được nhiều đối tượng hơn
        
        # Kiểm tra nếu có ít nhất một bounding box được phát hiện
        if len(results[0].boxes.data) > 0:
            boxes = results[0].boxes.data  # Lấy thông tin các bounding box
            
            # Lọc ra các bounding box có class là 0 (biển số xe)
            plate_boxes = [box for box in boxes if int(box[5]) == 0]
            
            # Nếu tìm thấy ít nhất một bounding box là biển số xe
            if len(plate_boxes) > 0:
                box = plate_boxes[0]  # Lấy bounding box đầu tiên (có thể chọn bounding box tốt nhất)
                x1, y1, x2, y2 = map(int, box[:4])  # Lấy tọa độ của bounding box
                
                # Cắt vùng ảnh biển số từ ảnh gốc
                roi = image[y1:y2, x1:x2]
                
                # Chuyển vùng biển số sang ảnh xám để xử lý tốt hơn
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                
                # Resize vùng biển số với tỷ lệ nhỏ hơn để tăng độ sắc nét
                roi_resized = cv2.resize(roi_gray, None, fx=3, fy=2.2, interpolation=cv2.INTER_CUBIC)
                
                # Áp dụng ngưỡng nhị phân Otsu để làm nổi bật ký tự
                _, roi_thresh = cv2.threshold(roi_resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # Sử dụng Tesseract để nhận diện ký tự từ vùng biển số
                custom_config = r'--oem 1 --psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                plate_text = pytesseract.image_to_string(roi_thresh, config=custom_config).strip()
                
                # Làm sạch kết quả OCR bằng cách loại bỏ các ký tự không hợp lệ
                plate_text = ''.join(filter(str.isalnum, plate_text))
                
                # Trả về kết quả nhận diện (text) và vùng biển số (ROI)
                return plate_text, roi
                
        # Trả về None nếu không phát hiện biển số
        return None, None
        
    except Exception as e:
        # Xử lý ngoại lệ nếu có lỗi
        print(f"Lỗi trong quá trình nhận diện: {e}")
        return None, None

# Hàm lưu biển số vào cơ sở dữ liệu
def save_license_plate_to_db(license_plate, image_license_data, conn, cursor):
    try:
        # Kiểm tra nếu kết nối hoặc con trỏ database không hợp lệ
        if conn is None or cursor is None:
            print("Kết nối database không hợp lệ")
            return False

        # Mã hóa ảnh vùng biển số sang định dạng base64
        _, buffer = cv2.imencode('.jpg', image_license_data)  # Nén ảnh thành định dạng JPEG
        byte_im = buffer.tobytes()  # Chuyển ảnh sang dạng byte
        image_base64 = base64.b64encode(byte_im).decode('utf-8')  # Mã hóa byte ảnh sang base64

        # Thêm dữ liệu biển số vào cơ sở dữ liệu
        sql = "INSERT INTO license_plates (license_plate, image_license_data) VALUES (%s, %s)"
        cursor.execute(sql, (license_plate, image_base64))  # Thực thi câu lệnh SQL
        conn.commit()  # Lưu thay đổi vào database
        print(f"Đã lưu biển số {license_plate} thành công")
        return True
    except Exception as e:
        # Xử lý ngoại lệ nếu có lỗi khi lưu
        print(f"Lỗi khi lưu biển số: {e}")
        if conn:
            conn.rollback()  # Hoàn tác nếu có lỗi
        return False

# Hàm xử lý và lưu biển số vào cơ sở dữ liệu
def process_and_save_plate(image, conn, cursor):
    try:
        # Kiểm tra nếu ảnh đầu vào rỗng
        if image is None:
            print("Ảnh đầu vào rỗng")
            return None, None

        # Nhận diện biển số từ ảnh
        detected_plate, plate_roi = improved_recognize_license_plate(image)

        # Nếu phát hiện được biển số và vùng ảnh biển số
        if detected_plate and plate_roi is not None:
            # Lưu biển số và vùng ảnh vào database
            if save_license_plate_to_db(detected_plate, plate_roi, conn, cursor):
                return detected_plate, plate_roi

        # Nếu không xử lý được biển số
        print("Không thể xử lý biển số")
        return None, None

    except Exception as e:
        # Xử lý ngoại lệ nếu có lỗi trong quá trình xử lý
        print(f"Lỗi trong quá trình xử lý: {e}")
        return None, None