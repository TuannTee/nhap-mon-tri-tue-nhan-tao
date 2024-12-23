import pandas as pd
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Dữ liệu
nha_data = {
    "năm": [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019],
    "phòng ngủ": [3, 4, 2, 5, 2, 4, 2, 3, 3, 4],
    "phòng tắm": [2, 3, 1, 4, 2, 3, 1, 2, 2, 3],
    "địa điểm": ["City Center", "Suburb", "City Center", "Suburb", "City Center", "Suburb", "Suburb", "City Center", "City Center", "Suburb"],
    "giá nhà": [250000, 300000, 200000, 400000, 220000, 350000, 190000, 280000, 260000, 310000]
}

# Chuyển dữ liệu thành DataFrame
data = pd.DataFrame(nha_data)

# Biến đổi các biến phân loại thành số
location_mapping = {"City Center": 0, "Suburb": 1}
data["địa điểm"] = data["địa điểm"].map(location_mapping)

# Tách dữ liệu thành biến độc lập (X) và biến phụ thuộc (y)
X = data[["năm", "phòng ngủ", "phòng tắm", "địa điểm"]]
y = data["giá nhà"]

# Chia dữ liệu thành tập huấn và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tạo mô hình quyết định Decision Tree
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# Dự đoán và đánh giá
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print("Mean Absolute Error:", mae)

# Vẽ sơ đồ cây quyết định
plt.figure(figsize=(12, 8))
plot_tree(model, feature_names=X.columns, filled=True, rounded=True)
plt.show()

# Xuất cây quyết định dưới dạng text
rules = export_text(model, feature_names=list(X.columns))
print(rules)
