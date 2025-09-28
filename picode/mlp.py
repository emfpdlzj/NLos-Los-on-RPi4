# train_and_export_mlp_fp32_tflite.py
import os
os.makedirs("result", exist_ok=True)

import uwb_dataset
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from keras.callbacks import CSVLogger
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
import tensorflow as tf  # <-- TFLite 변환용

# 1) 데이터 로드 & 프리앰블 정규화
data = uwb_dataset.import_from_files()
csv_logger = CSVLogger('result/log.csv', append=False)
for item in data:
    item[15:] = item[15:] / float(item[2])

# 2) train / test split
train = data[:30000, :]
np.random.shuffle(train)
x_train = train[:30000, 15:]; y_train = train[:30000, 0].astype(int)
x_test  = data[30000:, 15:];   y_test  = data[30000:, 0].astype(int)

# 3) 검증셋 분리
x_val = x_train[25000:]; y_val = y_train[25000:]
x_train = x_train[:25000]; y_train = y_train[:25000]

# 4) Feature selection: argmax 기준 앞뒤 50 → 길이 100
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
x_val   = argmax_window(x_val, 50)
x_test  = argmax_window(x_test, 50)

# 5) MLP 모델
model = Sequential([
    Dense(30, input_shape=(100,), use_bias=True),
    BatchNormalization(),
    Dense(16, activation='relu'),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dense(16, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
print(model.summary())

# 6) 학습
hist = model.fit(x_train, y_train, epochs=10, batch_size=64,
                 validation_data=(x_val, y_val), callbacks=[csv_logger])

# 7) 평가/리포트 (선택)
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=64)
print('## evaluation ##', loss_and_metrics)
y_prob = model.predict(x_test); y_pred = np.argmax(y_prob, axis=1)
print(classification_report(y_test, y_pred, digits=4))
cm = confusion_matrix(y_test, y_pred)
import seaborn as sns, matplotlib.pyplot as plt
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1], yticklabels=[0,1])
plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Confusion Matrix')
plt.tight_layout(); plt.savefig("result/mlp_confmat.png", dpi=150)

# (선택) Keras 저장
model.save("5.mlp.h5")

# 8) **Unquantized FP32 TFLite** 내보내기
converter = tf.lite.TFLiteConverter.from_keras_model(model)   # 최적화/양자화 옵션 X
tflite_fp32 = converter.convert()
with open("mlp_fp32.tflite", "wb") as f:
    f.write(tflite_fp32)
print("Saved: mlp_fp32.tflite  (input shape: (1,100), dtype=float32)")
