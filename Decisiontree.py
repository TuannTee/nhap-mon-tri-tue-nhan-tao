import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

data = pd.DataFrame({
    "year": [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019],
    "area": [1200, 1500, 800, 2000, 1000, 1800, 850, 1450, 1300, 1600],
    "bedrooms": [3, 4, 2, 5, 2, 4, 2, 3, 3, 4],
    "bathrooms": [2, 3, 1, 4, 2, 3, 1, 2, 2, 3],
    "location": ["City Center", "Suburb", "City Center", "Suburb", "City Center", "Suburb", "Suburb", "City Center", "City Center", "Suburb"],
    "price": [250000, 300000, 200000, 400000, 220000, 350000, 190000, 280000, 260000, 310000]
})
# Prepare features (X) and target variable (y)
X = data[['year', 'area', 'bedrooms', 'bathrooms', 'location']]
y = data['price']

# Convert categorical variables to numeric
X = pd.get_dummies(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Decision Tree model
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R2 Score: {r2}')

# Example prediction
sample = X_test.iloc[0].values.reshape(1, -1)
predicted_price = model.predict(sample)
print(f'Predicted price for sample: ${predicted_price[0]:,.2f}')