import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators  # sklearn 0.20.1에서만 돌아감요
import warnings
warnings.filterwarnings('ignore')

# mae 는 1.5234948750000008
# pca -> 8개로 하면 mae 1.8588125000000022
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Conv2D, Dropout, MaxPool2D, Flatten
from sklearn.metrics import mean_absolute_error

train = pd.read_csv('./vscode/dacon/train.csv', index_col=0)
test = pd.read_csv('./vscode/dacon/test.csv', index_col=0)
submission = pd.read_csv('./vscode/dacon/sample_submission.csv', index_col=0)

train = train.interpolate()
test = test.interpolate()

train = train.fillna(train.mean())
test = test.fillna(train.mean())

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
y = y.reshape(-1)
# from sklearn.decomposition import PCA
# pca = PCA(n_components=20)
# pca.fit(x)
# x = pca.transform(x) 

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=66)

allAlgorithms = all_estimators(type_filter='regressor')

for (name, algorithm) in allAlgorithms :
    model = algorithm()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(name, '의 mae = ', mean_absolute_error(y_pred, y_test))



