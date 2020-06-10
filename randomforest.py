# mae 는 1.5234948750000008
# pca -> 8개로 하면 mae 1.8588125000000022
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest 
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Conv2D, Dropout, MaxPool2D, Flatten
from sklearn.metrics import mean_absolute_error
# import xgboost as xgb
train = pd.read_csv('./vscode/dacon/train.csv', index_col=0)
test = pd.read_csv('./vscode/dacon/test.csv', index_col=0)
submission = pd.read_csv('./vscode/dacon/sample_submission.csv', index_col=0)

train = train.interpolate()
test = test.interpolate()

train = train.fillna(train.mean())
test = test.fillna(train.mean())

print(train.shape)
print(test.shape)

train = np.array(train)
x_predict = np.array(test)

x = train[:,:71]
y = train[:,71:]

# 전처리
from sklearn.preprocessing import RobustScaler, StandardScaler
scaler = RobustScaler()
scaler.fit(x) 
x = scaler.transform(x)
x_predict = scaler.transform(x_predict)

# from sklearn.decomposition import PCA
# pca = PCA(n_components=12)
# pca.fit(x)
# x = pca.transform(x)
# x_predict = pca.transform(x_predict)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=43)

model1 = RandomForestRegressor(n_estimators=300, criterion='mae',random_state=64,max_features = "auto", min_samples_leaf = 50)
model1.fit(x_train,y_train)
y_pred = model1.predict(x_test) # 회귀던 분류던 사용할 수 있음
mae = mean_absolute_error(y_pred, y_test)
print('랜덤포레스트 mae 는', mae)