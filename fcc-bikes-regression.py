import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler 
from imblearn.over_sampling import RandomOverSampler
import copy
import seaborn as sns
import tensorflow as tf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report
# ============== Formatting Model : Start  ===================
dataset_cols = ["bike_count", "hour", "temp", "humidity", "wind", "visibility", "dew_pt_temp", "radiation", "rain", "snow", "functional"]
df = pd.read_csv("SeoulBikeData.csv").drop(["Date", "Holiday", "Seasons"], axis = 1)
df.columns = dataset_cols
df["functional"] = (df["functional"] == "Yes").astype(int)
df = df[df["hour"] == 12]
df = df.drop(["hour", "wind", "visibility", "functional"], axis = 1)
# for label in df.columns[1:]:
#     plt.scatter(df[label], df["bike_count"])
#     plt.title(label)
#     plt.ylabel("Bike Count at Noon")
#     plt.xlabel(label)
#     plt.legend()
#     plt.show()
# Train, validation, test dataset
train, valid, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])
def get_xy(dataframe, y_label, x_labels = None):
    dataframe = copy.deepcopy(dataframe)
    if x_labels is None:
        X = dataframe[[c for c in dataframe.columns if c != y_label]].values
    else:
        if len(x_labels) == 1:
            X = dataframe[x_labels[0]].values.reshape(-1,1)
        else:
            X = dataframe[x_labels].values

    y = dataframe[y_label].values.reshape(-1,1)
    data = np.hstack((X, y))
    return data, X, y
_, X_train_temp, y_train_temp = get_xy(train, "bike_count", x_labels=["temp"])
_, X_valid_temp, y_valid_temp = get_xy(valid, "bike_count", x_labels=["temp"])
_, X_test_temp, y_test_temp = get_xy(test, "bike_count", x_labels=["temp"])
# ============== Formatting Model : End  ===================
# ============== Linear Regression  ===================
temp_reg = LinearRegression()
temp_reg.fit(X_train_temp, y_train_temp)
# print(temp_reg.coef_, temp_reg.intercept_)
# print(temp_reg.score(X_test_temp, y_test_temp))
# plt.scatter(X_train_temp, y_train_temp, label = "Data", color = "blue")
# x = tf.linspace(-20, 40, 100)
# plt.plot(x, temp_reg.predict(np.array(x).reshape(-1,1)), label = "Fit", color = "red", linewidth =3)
# plt.legend()
# plt.title("Bike vs Temp")
# plt.ylabel("Number of bikes")
# plt.xlabel("Temp")
# plt.show()
# ============== Multiple  Linear Regression  ===================
_, X_train_all, y_train_all = get_xy(train, "bike_count", x_labels=df.columns[1:])
_, X_valid_all, y_valid_all = get_xy(valid, "bike_count", x_labels=df.columns[1:])
_, X_test_all, y_test_all = get_xy(test, "bike_count", x_labels=df.columns[1:])
all_reg = LinearRegression()
all_reg.fit(X_train_all, y_train_all)
# print(all_reg.coef_, all_reg.intercept_)
# print(all_reg.score(X_test_all, y_test_all))
# ==============  Linear Regression with Neural Network  ===================
# Based on the training process, the linear regression in this case, may give different line
# here we use backpropagation
def plot_loss(history):
    plt.plot(history.history['loss'], label = 'loss')
    plt.plot(history.history['val_loss'], label = 'val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
    plt.show()
temp_normalizer = tf.keras.layers.Normalization(input_shape=(1,), axis=None) # Normalize the temprature
temp_normalizer.adapt(X_train_temp.reshape(-1))
temp_nn_model = tf.keras.Sequential([temp_normalizer, tf.keras.layers.Dense(1)])
temp_nn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.1), loss = 'mean_squared_error')
history = temp_nn_model.fit(X_train_temp.reshape(-1), y_train_temp, verbose=0, epochs=1000, validation_data=(X_valid_temp, y_valid_temp))
#plot_loss(history)
# plt.scatter(X_train_temp, y_train_temp, label = "Data", color = "blue")
# x = tf.linspace(-20, 40, 100)
# plt.plot(x, temp_nn_model.predict(np.array(x).reshape(-1,1)), label = "Fit", color = "red", linewidth =3)
# plt.legend()
# plt.title("Bike vs Temp")
# plt.ylabel("Number of bikes")
# plt.xlabel("Temp")
# plt.show()
# ============== Neural Network with one input  ===================
# In some cases, neural network, maybe the regression can be nonlinear as this case
# temp_normalizer = tf.keras.layers.Normalization(input_shape=(1,), axis=None) # Normalize the temprature
# temp_normalizer.adapt(X_train_temp.reshape(-1))
# nn_model = tf.keras.Sequential([
#     temp_normalizer, 
#     tf.keras.layers.Dense(32, activation= 'relu'),
#     tf.keras.layers.Dense(32, activation= 'relu'),
#     tf.keras.layers.Dense(32, activation= 'relu'),
#     tf.keras.layers.Dense(1),
# ])
# nn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.001), loss = 'mean_squared_error')
# history = nn_model.fit(X_train_temp, y_train_temp, verbose=0, epochs=100, validation_data=(X_valid_temp, y_valid_temp))
# plot_loss(history)
# plt.scatter(X_train_temp, y_train_temp, label = "Data", color = "blue")
# x = tf.linspace(-20, 40, 100)
# plt.plot(x, nn_model.predict(np.array(x).reshape(-1,1)), label = "Fit", color = "red", linewidth =3)
# plt.legend()
# plt.title("Bike vs Temp")
# plt.ylabel("Number of bikes")
# plt.xlabel("Temp")
# plt.show()
# ============== Neural Network with one input  ===================
# In some cases, neural network, maybe the regression can be nonlinear as this case
all_normalizer = tf.keras.layers.Normalization(input_shape=(6,), axis=-1) # Normalize the temprature
all_normalizer.adapt(X_train_all)
nn_model = tf.keras.Sequential([
    all_normalizer, 
    tf.keras.layers.Dense(32, activation= 'relu'),
    tf.keras.layers.Dense(32, activation= 'relu'),
    tf.keras.layers.Dense(32, activation= 'relu'),
    tf.keras.layers.Dense(1),
])
nn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.001), loss = 'mean_squared_error')
history = nn_model.fit(X_train_all, y_train_all, verbose=0, epochs=100, validation_data=(X_valid_all, y_valid_all))
# plot_loss(history)
# Calculate the MSE for both linear reg and nn_model
def MSE(y_pred, y_real):
    return (np.square(y_pred - y_real)).mean()
y_pred_lr = all_reg.predict(X_test_all)
y_pred_nn = nn_model.predict(X_test_all)
print(MSE(y_pred_lr, y_test_all))
print(MSE(y_pred_nn, y_test_all))
ax = plt.axes(aspect = "equal")
plt.scatter(y_test_all, y_pred_lr, label = "Lin Reg Preds")
plt.scatter(y_test_all, y_pred_nn, label = "NN Preds")
plt.xlabel("True Values")
plt.ylabel("Predictions")
lims = [0, 1800]
plt.xlim(lims)
plt.ylim(lims)
plt.legend()
plt.plot(lims, lims, c = "red")
plt.show()
