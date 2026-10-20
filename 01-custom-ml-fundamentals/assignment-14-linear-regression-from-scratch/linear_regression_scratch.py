import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

data = pd.read_csv("dataset\\Student_Performance.csv")
#print(data.head())

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

ohe_extraculicular = OneHotEncoder(sparse_output=False,handle_unknown='ignore',dtype=int)
ohe_extraculicular.fit(data[['Extracurricular Activities']])
encoded_extraculicular = ohe_extraculicular.transform(data[['Extracurricular Activities']])

data.drop(['Extracurricular Activities'], axis=1, inplace=True)
data = pd.concat([data, pd.DataFrame(encoded_extraculicular, columns=ohe_extraculicular.get_feature_names_out(['Extracurricular Activities']))], axis=1)

target = data['Performance Index']
data = data.drop('Performance Index', axis=1)

x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(x_train, y_train)

prediction = model.predict(x_test)

print(mse(prediction, y_test))

class LinearRegression:
    def __init__(self, learning_rate=0.0001, epochs=1000):
        self.weights = np.zeros(x_train.shape[1])
        self.learning_rate = learning_rate
        self.epochs = epochs

    def predict(self, x):
        return np.dot(x, self.weights)

    def fit(self, x, y):
        #Gradient Descent
        x = np.array(x)
        y = np.array(y)

        for _ in range(self.epochs):
            #This is made by using the derivative of the loss function and by using vectorization
            y_pred = self.predict(x)
            derivative = (2/len(x)) * np.dot(x.T, (y_pred - y))
            self.weights -= self.learning_rate * derivative

    def __repr__(self):
        return f"Weights: {self.weights}"

custom_model = LinearRegression(learning_rate=0.0001, epochs=1000)
custom_model.fit(x_train, y_train)
#print(custom_model)

y_pred = custom_model.predict(x_test)

print(mse(y_pred, y_test))

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
data_norm = scaler_x.fit_transform(data)
target_norm = scaler_y.fit_transform(target.values.reshape(-1, 1))

custom_model_norm = LinearRegression(learning_rate=0.01, epochs=1000)
x_train_norm, x_test_norm, y_train_norm, y_test_norm = train_test_split(data_norm, target_norm, test_size=0.2, random_state=42)

custom_model_norm.fit(x_train_norm, y_train_norm.flatten())
y_pred_norm = custom_model_norm.predict(x_test_norm)

# Denormalize predictions back to original scale
y_pred_denorm = scaler_y.inverse_transform(y_pred_norm.reshape(-1, 1))
y_test_denorm = scaler_y.inverse_transform(y_test_norm)

print(mse(y_pred_denorm, y_test_denorm))


