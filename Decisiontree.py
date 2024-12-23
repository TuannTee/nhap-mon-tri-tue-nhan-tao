import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Tải dữ liệu
data = pd.read_csv("housing_data.csv")

# Chọn các đặc trưng và biến mục tiêu
dac_trung = ['dien_tich', 'so_phong', 'tuoi_nha']  # Thay thế bằng tên đặc trưng thực tế
muc_tieu = 'gia'

X = data[dac_trung]
y = data[muc_tieu]

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Khởi tạo mô hình Cây Quyết Định
mo_hinh = DecisionTreeRegressor(random_state=42)

# Huấn luyện mô hình
mo_hinh.fit(X_train, y_train)

# Dự đoán
y_du_doan = mo_hinh.predict(X_test)

# Đánh giá mô hình
mse = mean_squared_error(y_test, y_du_doan)
print(f'Lỗi Bình Phương Trung Bình: {mse}')

# Dự đoán giá nhà trong tương lai
du_lieu_tuong_lai = pd.DataFrame({
    'dien_tich': [40],  # Diện tích (m2)
    'so_phong': [4],    # Số phòng
    'tuoi_nha': [4]     # Tuổi của ngôi nhà
})
du_doan_tuong_lai = mo_hinh.predict(du_lieu_tuong_lai)
print(f'Dự Đoán Giá Nhà Trong Tương Lai: {du_doan_tuong_lai}')