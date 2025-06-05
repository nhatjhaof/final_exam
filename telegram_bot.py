import requests

# Cấu hình thông tin bot Telegram
TELEGRAM_TOKEN = "8043672123:AAEg7VcA-bLA1q3z88dG6giIORNkJBi8_yY"  # Thay bằng token bot của bạn
TELEGRAM_CHAT_ID = "5553535226"  # Thay bằng ID chat của bạn (hoặc nhóm)

# Hàm gửi thông báo qua Telegram
def sendTelegramMessage(message, image_path=None):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }
    # Nếu có hình ảnh, gửi ảnh qua phương thức sendPhoto
    files = None
    if image_path:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
        files = {
            'photo': open(image_path, 'rb')
        }
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "caption": message,
            "parse_mode": "HTML"
        }

    response = requests.post(url, data=payload, files=files)

    if response.status_code == 200:
        print("Thông báo đã gửi thành công!")
    else:
        print("Lỗi khi gửi thông báo:", response.text)

    # Đóng file ảnh nếu đã mở
    if files:
        files['photo'].close()
