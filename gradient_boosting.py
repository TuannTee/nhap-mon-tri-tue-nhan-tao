# Import các thư viện cần thiết
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Bước 1: Đọc dữ liệu
# Tạo dữ liệu mẫu bao gồm cột 'year' để vẽ biểu đồ
data = pd.DataFrame({
    "year": [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019],
    "area": [1200, 1500, 800, 2000, 1000, 1800, 850, 1450, 1300, 1600],
    "bedrooms": [3, 4, 2, 5, 2, 4, 2, 3, 3, 4],
    "bathrooms": [2, 3, 1, 4, 2, 3, 1, 2, 2, 3],
    "location": ["City Center", "Suburb", "City Center", "Suburb", "City Center", "Suburb", "Suburb", "City Center", "City Center", "Suburb"],
    "price": [250000, 300000, 200000, 400000, 220000, 350000, 190000, 280000, 260000, 310000]
})

# Hiển thị thông tin cơ bản về dữ liệu
print(data.head())

# Bước 2: Chuẩn bị dữ liệu
X = data[['area', 'bedrooms', 'bathrooms']]
y = data['price']

# Chuyển đổi cột phân loại (categorical) thành dạng số nếu cần
if 'location' in data.columns:
    X = pd.concat([X, pd.get_dummies(data['location'], drop_first=True)], axis=1)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Bước 3: Xây dựng mô hình Gradient Boosting
model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Bước 4: Dự đoán giá nhà
y_pred = model.predict(X_test)

# Đánh giá mô hình
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("R-squared Score (R2):", r2_score(y_test, y_pred))

# Bước 5: Hiển thị biểu đồ thay đổi giá nhà theo năm
plt.figure(figsize=(10, 6))
plt.plot(data['year'], data['price'], marker='o', label='Actual Prices', color='blue')
plt.xlabel('Year')
plt.ylabel('House Prices')
plt.title('House Price Changes Over Years')
plt.grid(True)
plt.legend()
plt.show()
