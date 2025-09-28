# train_and_export_mlp_quant.py
# ==== 0) Imports ====
import os
os.makedirs("result", exist_ok=True)

import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from keras.callbacks import CSVLogger
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
import uwb_dataset

# ==== 1) 데이터 로드 & 프리앰블 정규화 ====
data = uwb_dataset.import_from_files()
csv_logger = CSVLogger('result/log.csv', append=False)
for item in data:
    item[15:] = item[15:] / float(item[2])

# ==== 2) Split ====
train = data[:30000, :]
np.random.shuffle(train)
x_train = train[:30000, 15:]
y_train = train[:30000, 0].astype(int)

x_test  = data[30000:, 15:]
y_test  = data[30000:, 0].astype(int)

# 검증셋
x_val = x_train[25000:]
y_val = y_train[25000:]
x_train = x_train[:25000]
y_train = y_train[:25000]

# ==== 3) Feature selection: argmax 윈도우(±50 → 길이 100) ====
def argmax_window(X, w=50):
    out = []
    L = X.shape[1]
    for row in X:
        m = int(row.argmax())
        s = max(0, m - w)
        e = s + 100
        if e > L:
            e = L; s = max(0, e - 100)
        wv = row[s:e]
        if wv.shape[0] < 100:
            wv = np.pad(wv, (0, 100 - wv.shape[0]))
        out.append(wv)
    return np.asarray(out)

x_train = argmax_window(x_train, 50)
x_val   = argmax_window(x_val,   50)
x_test  = argmax_window(x_test,  50)

# ==== 4) 사진과 유사한 MLP ====
model = Sequential()
model.add(Dense(30, input_shape=(100,), use_bias=True))
model.add(BatchNormalization())
model.add(Dense(16, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(16, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
print(model.summary())

# ==== 5) 학습 ====
model.fit(
    x_train, y_train,
    epochs=10, batch_size=64,
    validation_data=(x_val, y_val),
    callbacks=[csv_logger], verbose=1
)

# ==== 6) (기준) Keras 정확도 ====
keras_prob = model.predict(x_test, batch_size=256, verbose=0)
keras_pred = np.argmax(keras_prob, axis=1)
print("Keras accuracy :", accuracy_score(y_test, keras_pred))
print(classification_report(y_test, keras_pred, digits=4))

# ==== 7) TFLite 내보내기 ====
# 7-1) Dynamic-range (INT8 weights + float activations, I/O=float32)
conv_dr = tf.lite.TFLiteConverter.from_keras_model(model)
conv_dr.optimizations = [tf.lite.Optimize.DEFAULT]
# 대표데이터는 넣어도 되고 안 넣어도 됨(가중치 위주). 넣으면 조금 더 안정적.
calib = x_val[:500].astype(np.float32)
def representative_dataset():
    for i in range(calib.shape[0]):
        yield [calib[i:i+1]]
conv_dr.representative_dataset = representative_dataset
conv_dr.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
    tf.lite.OpsSet.TFLITE_BUILTINS  # float+int8 혼용 허용
]
tflite_int8w_floatact = conv_dr.convert()
open("mlp_int8w_floatact.tflite", "wb").write(tflite_int8w_floatact)

# 7-2) Full-INT8 (가중치/활성 INT8, I/O=uint8) — 라즈베리파이에서 더 작고 빠름
conv_full = tf.lite.TFLiteConverter.from_keras_model(model)
conv_full.optimizations = [tf.lite.Optimize.DEFAULT]
conv_full.representative_dataset = representative_dataset
conv_full.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv_full.inference_input_type  = tf.uint8
conv_full.inference_output_type = tf.uint8
tflite_int8_full = conv_full.convert()
open("mlp_int8_full.tflite", "wb").write(tflite_int8_full)

print("Saved: mlp_int8w_floatact.tflite (I/O=float32), mlp_int8_full.tflite (I/O=uint8)")
print("Sizes (KB): DR=", len(tflite_int8w_floatact)/1024, " | FullINT8=", len(tflite_int8_full)/1024)
