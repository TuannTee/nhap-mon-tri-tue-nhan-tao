import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import tree

# Tạo dữ liệu mẫu
data = pd.DataFrame({
    "năm": [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019],
    "phòng ngủ": [3, 4, 2, 5, 2, 4, 2, 3, 3, 4],
    "phòng tắm": [2, 3, 1, 4, 2, 3, 1, 2, 2, 3],
    "địa điểm": ["City Center", "Suburb", "City Center", "Suburb", "City Center", "Suburb", "Suburb", "City Center", "City Center", "Suburb"],
    "giá nhà": [250000, 300000, 200000, 400000, 220000, 350000, 190000, 280000, 260000, 310000]
})

# Chuẩn bị đặc trưng (X) và biến mục tiêu (y)
X = data[['năm', 'phòng ngủ', 'phòng tắm', 'địa điểm']]
y = data['giá nhà']

# Chuyển đổi biến phân loại sang dạng số
X = pd.get_dummies(X)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tạo và huấn luyện mô hình Cây Quyết Định
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# Thực hiện dự đoán
y_pred = model.predict(X_test)

# Đánh giá mô hình
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Sai số bình phương trung bình: {mse}')
print(f'Điểm R2: {r2}')

# Ví dụ dự đoán
sample = X_test.iloc[0].values.reshape(1, -1)
predicted_price = model.predict(sample)
print(f'Giá dự đoán cho mẫu: ${predicted_price[0]:,.2f}')

#Hiển thị biểu đồ
import matplotlib.pyplot as plt
plt.figure(figsize=(20, 10))
tree.plot_tree(model, feature_names=X.columns, filled=True, rounded=True)
plt.title('Cây Quyết Định')
plt.show()