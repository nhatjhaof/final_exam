import mysql.connector

try:
    print("→ Đang thử kết nối đến MySQL qua tunnel port 58763...")
    conn = mysql.connector.connect(
        host='127.0.0.1',
        port=58763,
        user='admin',
        password='123456',
        database='car_service',
        connection_timeout=5
    )
    print("✅ Kết nối thành công!")

    cursor = conn.cursor()
    cursor.execute("SELECT DATABASE();")
    print("📦 Database hiện tại:", cursor.fetchone()[0])

    conn.close()
    print("→ Kết nối đã đóng.")

except Exception as e:
    print("❌ Lỗi khi kết nối:", str(e))
