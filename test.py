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
        sql = sql = "INSERT INTO license_plates (license_plate, image_license_data, capture_time) VALUES (%s, %s, NOW())"
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