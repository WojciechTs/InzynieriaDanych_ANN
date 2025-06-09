import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.engine import data_adapter
import matplotlib.pyplot as plt

def _is_distributed_dataset(ds):
    return isinstance(ds, data_adapter.input_lib.DistributedDatasetSpec)

data_adapter._is_distributed_dataset = _is_distributed_dataset

dataset = pd.read_csv('DataSet_for_ANN.csv')

x = dataset.iloc[:,3:-1].values
y = dataset.iloc[:,-1].values

le = LabelEncoder()

ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])],remainder='passthrough')
x = np.array(ct.fit_transform(x))


x[:,4] = le.fit_transform(x[:,4])

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])

y = le.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

sc = StandardScaler()
x_train[:,3:4] = sc.fit_transform(x_train[:,3:4])
x_test[:,3:4] = sc.fit_transform(x_test[:,3:4])
x_train[:,5:9] = sc.fit_transform(x_train[:,5:9])
x_test[:,5:9] = sc.fit_transform(x_test[:,5:9])
x_train[:,11:] = sc.fit_transform(x_train[:,11:])
x_test[:,11:] = sc.fit_transform(x_test[:,11:])

x_train = np.asarray(x_train).astype(np.float32)
y_train = np.asarray(y_train).astype(np.float32)
x_test = np.asarray(x_test).astype(np.float32)
y_test = np.asarray(y_test).astype(np.float32)
early_stop = EarlyStopping(monitor='val_loss', patience=25, mode='min')

model = Sequential()
model.add(Dense(units=x_train.shape[1], activation='relu'))
model.add(Dense(units=x_train.shape[1] // 2, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=64, epochs=1000, validation_data=(x_test, y_test), callbacks=[early_stop])

losses = pd.DataFrame(model.history.history)

print(losses[-50:])

losses.plot(figsize=(15, 5))
plt.show()





