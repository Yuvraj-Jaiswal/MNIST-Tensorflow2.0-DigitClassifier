from tensorflow.python.keras.datasets import mnist
import pandas as pd
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.callbacks import EarlyStopping
import numpy as np

#%%
data = mnist
(x_train,y_train) , (x_test,y_test) = data.load_data()
early_stoping = EarlyStopping(monitor='accuracy' , mode='max' , verbose=1)

model = Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(50,activation='relu'))
model.add(Dropout(0.5))
model.add( Dense(10,activation='softmax'))


model.compile(optimizer='adam' , loss='sparse_categorical_crossentropy' ,
              metrics=["accuracy"] )
model.fit(x_train,y_train,epochs=50,callbacks=[early_stoping])

#%%
dict = [0,1,2,3,4,5,6,7,8,9]
prediction = model.predict(x_test)
pred_dic = []
for i in range(10000):
    pred_dic.append(dict[np.argmax(prediction[i])])
#%%
import seaborn as sns
df_pred = pd.DataFrame(pred_dic , columns=['pred'])
df_orig = pd.DataFrame(y_test , columns=['orig'])

df_final = pd.concat([df_pred,df_orig] , axis=1 )
#%%
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,pred_dic)