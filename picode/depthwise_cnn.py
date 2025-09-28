# train_and_export_fp32_tflite.py
import os
os.makedirs("result", exist_ok=True)

import uwb_dataset
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from keras.callbacks import CSVLogger
from keras.models import Model
from keras.layers import (Input, SeparableConv1D, BatchNormalization, Activation,
                          MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout,
                          Add, Conv1D, SeparableConv1D)
from keras.optimizers import Adam
import tensorflow as tf  # <-- TFLite 변환에 필요

# ------------------------
# 1) 데이터 로드 & 전처리 (동일)
# ------------------------
data = uwb_dataset.import_from_files()
csv_logger = CSVLogger('result/log.csv', append=False)

for item in data:
    item[15:] = item[15:]/float(item[2])

train = data[:30000, :]
np.random.shuffle(train)
x_train = train[:30000, 15:]
y_train = train[:30000, 0]
x_test  = data[30000:, 15:]
y_test  = data[30000:, 0]

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
        # 혹시 길이가 100이 아닐 경우 패딩(드물지만 경계에서 발생 가능)
        if w.shape[0] < win:
            pad = np.zeros((win - w.shape[0],), dtype=w.dtype)
            w = np.concatenate([w, pad], axis=0)
        out.append(w)
    return np.asarray(out)

x_train = slice_window(x_train)[..., np.newaxis]  # (N,100,1)
x_val   = slice_window(x_val)[...,   np.newaxis]
x_test  = slice_window(x_test)[...,  np.newaxis]

# ------------------------
# 2) Xception-1D (Depthwise CNN)
# ------------------------
def relu_sepconv_bn(x, filters, k=3, name=None):
    x = Activation('relu')(x)
    x = SeparableConv1D(filters, k, padding='same', use_bias=False, name=name)(x)
    x = BatchNormalization()(x)
    return x

inp = Input(shape=(100,1))

# Entry flow
x = Conv1D(32, 3, strides=2, padding='same', use_bias=False)(inp)
x = Activation('relu')(x)
x = Conv1D(64, 3, padding='same', use_bias=False)(x)
x = Activation('relu')(x)

# Block 1: 128, downsample + residual proj
res = Conv1D(128, 1, strides=2, padding='same', use_bias=False)(x)
res = BatchNormalization()(res)
x = relu_sepconv_bn(x, 128, 3)
x = MaxPooling1D(3, strides=2, padding='same')(x)
x = Add()([x, res])

# Block 2: 256
res = Conv1D(256, 1, strides=2, padding='same', use_bias=False)(x)
res = BatchNormalization()(res)
x = relu_sepconv_bn(x, 256, 3)
x = MaxPooling1D(3, strides=2, padding='same')(x)
x = Add()([x, res])

# Block 3: 728
res = Conv1D(728, 1, strides=2, padding='same', use_bias=False)(x)
res = BatchNormalization()(res)
x = relu_sepconv_bn(x, 728, 3)
x = MaxPooling1D(3, strides=2, padding='same')(x)
x = Add()([x, res])

# Middle flow: 8 times
for i in range(8):
    res = x
    x = relu_sepconv_bn(x, 728, 3, name=f"mid_sep_{i}_a")
    x = relu_sepconv_bn(x, 728, 3, name=f"mid_sep_{i}_b")
    x = relu_sepconv_bn(x, 728, 3, name=f"mid_sep_{i}_c")
    x = Add()([x, res])

# Exit flow
res = Conv1D(1024, 1, strides=2, padding='same', use_bias=False)(x)
res = BatchNormalization()(res)

x = relu_sepconv_bn(x, 728, 3)
x = relu_sepconv_bn(x, 1024, 3)
x = MaxPooling1D(3, strides=2, padding='same')(x)
x = Add()([x, res])

x = relu_sepconv_bn(x, 1536, 3)
x = relu_sepconv_bn(x, 2048, 3)

x = GlobalAveragePooling1D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
out = Dense(1, activation='sigmoid')(x)  # 이진 분류

model = Model(inp, out)
model.compile(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# ------------------------
# 3) 학습 / 평가 / 리포트
# ------------------------
history = model.fit(
    x_train, y_train, epochs=10, batch_size=64,
    validation_data=(x_val, y_val), callbacks=[csv_logger], verbose=1
)

print('## evaluation ##')
print(model.evaluate(x_test, y_test, batch_size=64))

y_pred = (model.predict(x_test) > 0.5).astype(int)
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1], yticklabels=[0,1])
plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig("result/confmat.png", dpi=150)

model.save("4.depthwise_cnn.h5")

# ------------------------
# 4) **Unquantized FP32 TFLite** 내보내기
# ------------------------
# 최적화/양자화 옵션 없이 그대로 변환 → FP32 모델
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_fp32 = converter.convert()
with open("depthwise_cnn_fp32.tflite", "wb") as f:
    f.write(tflite_fp32)

# 입출력 shape/dtype 메모(파이에서 확인 용도)
inp_shape = model.inputs[0].shape
out_shape = model.outputs[0].shape
print("Saved: depthwise_cnn_fp32.tflite")
print("Input:", inp_shape, "Output:", out_shape)
