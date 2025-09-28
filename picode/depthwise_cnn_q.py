import os
os.makedirs("result", exist_ok=True)

import uwb_dataset
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns, matplotlib.pyplot as plt
from keras.callbacks import CSVLogger
from keras.models import Model
from keras.layers import (Input, SeparableConv1D, BatchNormalization, Activation,
                          MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout,
                          Add, Conv1D, SeparableConv1D)
from keras.optimizers import Adam

# 1) 데이터 로드 & 전처리 
data = uwb_dataset.import_from_files()
csv_logger = CSVLogger('result/log.csv', append=False)

for item in data:
    item[15:] = item[15:] / float(item[2])

train = data[:30000, :]
np.random.shuffle(train)
x_train = train[:30000, 15:]; y_train = train[:30000, 0]
x_test  = data[30000:, 15:];   y_test  = data[30000:, 0]

x_val = x_train[25000:]; y_val = y_train[25000:]
x_train = x_train[:25000]; y_train = y_train[:25000]

def slice_window(arr, win=100, half=50):
    out=[]
    L = arr.shape[1]
    for row in arr:
        a = int(row.argmax())
        start = max(0, a-half)
        end   = start + win
        if end > L:
            end = L
            start = max(0, end - win)
        w = row[start:end]
        if w.shape[0] < win:
            w = np.pad(w, (0, win - w.shape[0]))
        out.append(w)
    return np.asarray(out)

x_train = slice_window(x_train)[..., None]  # (N,100,1)
x_val   = slice_window(x_val)[...,   None]
x_test  = slice_window(x_test)[...,  None]


# 2) Depthwise/Xception-1D
def relu_sepconv_bn(x, filters, k=3, name=None):
    x = Activation('relu')(x)
    x = SeparableConv1D(filters, k, padding='same', use_bias=False, name=name)(x)
    x = BatchNormalization()(x)
    return x

inp = Input(shape=(100,1))
x = Conv1D(32, 3, strides=2, padding='same', use_bias=False)(inp); x = Activation('relu')(x)
x = Conv1D(64, 3, padding='same', use_bias=False)(x);              x = Activation('relu')(x)

res = Conv1D(128, 1, strides=2, padding='same', use_bias=False)(x); res = BatchNormalization()(res)
x = relu_sepconv_bn(x, 128, 3); x = MaxPooling1D(3, strides=2, padding='same')(x); x = Add()([x, res])

res = Conv1D(256, 1, strides=2, padding='same', use_bias=False)(x); res = BatchNormalization()(res)
x = relu_sepconv_bn(x, 256, 3); x = MaxPooling1D(3, strides=2, padding='same')(x); x = Add()([x, res])

res = Conv1D(728, 1, strides=2, padding='same', use_bias=False)(x); res = BatchNormalization()(res)
x = relu_sepconv_bn(x, 728, 3); x = MaxPooling1D(3, strides=2, padding='same')(x); x = Add()([x, res])

for i in range(8):
    res = x
    x = relu_sepconv_bn(x, 728, 3, name=f"mid_sep_{i}_a")
    x = relu_sepconv_bn(x, 728, 3, name=f"mid_sep_{i}_b")
    x = relu_sepconv_bn(x, 728, 3, name=f"mid_sep_{i}_c")
    x = Add()([x, res])

res = Conv1D(1024, 1, strides=2, padding='same', use_bias=False)(x); res = BatchNormalization()(res)
x = relu_sepconv_bn(x, 728, 3); x = relu_sepconv_bn(x, 1024, 3)
x = MaxPooling1D(3, strides=2, padding='same')(x); x = Add()([x, res])

x = relu_sepconv_bn(x, 1536, 3); x = relu_sepconv_bn(x, 2048, 3)
x = GlobalAveragePooling1D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
out = Dense(1, activation='sigmoid')(x)

model = Model(inp, out)
model.compile(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()


# 3) 학습/평가

history = model.fit(x_train, y_train, epochs=10, batch_size=64,
                    validation_data=(x_val, y_val), callbacks=[csv_logger], verbose=1)
print('## evaluation ##')
print(model.evaluate(x_test, y_test, batch_size=64))


# 4) INT8 TFLite 변환 (입/출력 uint8)
# 대표 데이터(학습 분포와 동일)로 보정
rep = x_train[:512].astype(np.float32)
def representative_data_gen():
    for i in range(len(rep)):
        yield [rep[i:i+1]]

conv = tf.lite.TFLiteConverter.from_keras_model(model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.representative_dataset = representative_data_gen
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv.inference_input_type  = tf.uint8
conv.inference_output_type = tf.uint8

tflite_int8 = conv.convert()
with open("depthwise_cnn_q.tflite", "wb") as f:
    f.write(tflite_int8)
print("Saved: depthwise_cnn_q.tflite")
