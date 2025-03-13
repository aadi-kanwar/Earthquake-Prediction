import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load dataset
data_path = 'D:/Projects/Earthquake Prediction/Datasets/earthquake_1995-2023.csv'
df = pd.read_csv(data_path)

# Display first few rows to understand structure
print(df.head())

# Selecting relevant features (Assuming column names, update if necessary)
features = ['latitude', 'longitude', 'nst']  # Adjust as per dataset
label = 'magnitude'  # Adjust if necessary

df = df.dropna(subset=features + [label])

# Splitting data into training and testing sets
X = df[features]
y = df[label]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the model
model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'MAE: {mae}')
print(f'RMSE: {rmse}')
print(f'R^2 Score: {r2}')


# Function to make predictions on user input
def predict_earthquake(latitude, longitude, nst):
    # scaler = StandardScaler()
    input_data = np.array([[latitude, longitude, nst]])
    # input_scaled = scaler.fit_transform(input_data)
    predicted_mag = model.predict(input_data)[0]
    return round(predicted_mag, 2)

# Function to take user input and predict earthquake magnitude
def user_input_prediction():
  latitude = float(input("Enter Latitude: "))
  longitude = float(input("Enter Longitude: "))
  nst = int(input("Enter Number of Stations (Nst): "))
  
  predicted_magnitude = predict_earthquake(latitude, longitude, nst)
  print(f"Predicted Earthquake Magnitude: {predicted_magnitude}")


  # print("Invalid input! Please enter numerical values for Latitude, Longitude, and Nst.")

# Call the function to allow user input
user_input_prediction()



# # Feature importance
# feature_importance = model.feature_importances_
# plt.bar(features, feature_importance)
# plt.xlabel('Features')
# plt.ylabel('Importance')
# plt.title('Feature Importance in Earthquake Prediction')
# plt.show()