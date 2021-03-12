import librosa
from librosa import display
import matplotlib.pyplot as plt
import os
import pandas as pd

import time
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import keras

import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Flatten, Dropout, Activation
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import optimizers

X = joblib.load('dataset_joblib/X.joblib')
y = joblib.load('dataset_joblib/y.joblib')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
x_traincnn = np.expand_dims(X_train, axis=2)
x_testcnn = np.expand_dims(X_test, axis=2)
x_traincnn.shape, x_testcnn.shape
#cnn
model = Sequential()

model.add(Conv1D(128, 5,padding='same',
                 input_shape=(40,1)))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(MaxPooling1D(pool_size=(8)))
model.add(Conv1D(128, 5,padding='same',))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(5))
model.add(Activation('softmax'))
# opt = tf.keras.optimizers.rmsprop(lr=0.00005, rho=0.9, epsilon=None, decay=0.0)
opt =optimizers.RMSprop(lr=0.00005, rho=0.9, epsilon=None, decay=0.0)
model.summary()
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
cnnhistory=model.fit(x_traincnn, y_train, batch_size=32, epochs=100000, validation_data=(x_testcnn, y_test))

#plot loss acc
plt.plot(cnnhistory.history['loss'])
plt.plot(cnnhistory.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig('loss.png', bbox_inches='tight')
plt.plot(cnnhistory.history['accuracy'])
plt.plot(cnnhistory.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig('acc.png', bbox_inches='tight')
#evaluation model
predictions = model.predict_classes(x_testcnn)
predictions
# y_test
new_Ytest = y_test.astype(int)
new_Ytest
report = classification_report(new_Ytest, predictions)
print(report)
matrix = confusion_matrix(new_Ytest, predictions)
print (matrix)

model_name = 'Emotion_Voice_Detection_Model.h5'
save_dir = 'model_res'
# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)
loaded_model = tf.keras.models.load_model('model_res/Emotion_Voice_Detection_Model.h5')
loaded_model.summary()
loss, acc = loaded_model.evaluate(x_testcnn, y_test)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))