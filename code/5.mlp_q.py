# ==== 0) Imports ====
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

# ==== 3) Feature selection: argmax 윈도우(±50 → 길이 최대 100) ====
def argmax_window(X, w=50):
    out = []
    for row in X:
        m = int(row.argmax())
        s = max(0, m - w)
        e = m + w
        out.append(row[s:e])
    return np.asarray(out)

def pad_to_100(X):
    if X.shape[1] == 100:
        return X
    Z = np.zeros((X.shape[0], 100), dtype=X.dtype)
    L = min(100, X.shape[1])
    Z[:, :L] = X[:, :L]
    return Z

x_train = pad_to_100(argmax_window(x_train, 50))
x_val   = pad_to_100(argmax_window(x_val,   50))
x_test  = pad_to_100(argmax_window(x_test,  50))

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
    callbacks=[csv_logger],
    verbose=1
)

# ==== 6) (기준) Keras 정확도 ====
keras_prob = model.predict(x_test, batch_size=256, verbose=0)
keras_pred = np.argmax(keras_prob, axis=1)
print("Keras accuracy :", accuracy_score(y_test, keras_pred))
print(classification_report(y_test, keras_pred, digits=4))

# ==== 7) PTQ: 가중치 int8, activation float32 (논문 설명과 동일) ====
#    - 모델 저장/로딩 없이 from_keras_model(model) 사용
#    - calibration set: 검증셋에서 500개 사용
calib = x_val[:500].astype(np.float32)

def representative_dataset():
    for i in range(calib.shape[0]):
        yield [calib[i:i+1]]

# (참고) float32 TFLite (baseline)
conv_fp32 = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_fp32 = conv_fp32.convert()
open("mlp_fp32.tflite", "wb").write(tflite_fp32)

# PTQ (weights int8, activations float32 → dynamic-range quantization)
conv_int8w = tf.lite.TFLiteConverter.from_keras_model(model)
conv_int8w.optimizations = [tf.lite.Optimize.DEFAULT]
conv_int8w.representative_dataset = representative_dataset
# 지원 op을 넉넉히 허용(빌트인 float + int8 혼용)
conv_int8w.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
    tf.lite.OpsSet.TFLITE_BUILTINS
]
tflite_int8w = conv_int8w.convert()
open("mlp_int8w_floatact.tflite", "wb").write(tflite_int8w)

print("FP32 size (KB):", len(tflite_fp32)/1024)
print("INT8W size (KB):", len(tflite_int8w)/1024)

# ==== 8) TFLite 추론 유틸 ====
def tflite_predict(tflite_model_bytes, X):
    interpreter = tf.lite.Interpreter(model_content=tflite_model_bytes)
    interpreter.allocate_tensors()
    input_idx  = interpreter.get_input_details()[0]['index']
    output_idx = interpreter.get_output_details()[0]['index']
    # dynamic-range 모델은 입력/출력이 float32인 경우가 일반적
    X = X.astype(np.float32)
    preds = []
    for i in range(X.shape[0]):
        interpreter.set_tensor(input_idx, X[i:i+1])
        interpreter.invoke()
        out = interpreter.get_tensor(output_idx)  # shape (1,2)
        preds.append(out[0])
    return np.vstack(preds)

# ==== 9) FP32/INT8W-FLOATACT 정확도 비교 ====
tfl_fp32_prob = tflite_predict(tflite_fp32, x_test)
tfl_fp32_pred = np.argmax(tfl_fp32_prob, axis=1)

print("TFLite FP32 accuracy :", accuracy_score(y_test, tfl_fp32_pred))

tfl_int8w_prob = tflite_predict(tflite_int8w, x_test)
tfl_int8w_pred = np.argmax(tfl_int8w_prob, axis=1)
print("TFLite INT8-weights (float act) accuracy :", accuracy_score(y_test, tfl_int8w_pred))
print(classification_report(y_test, tfl_int8w_pred, digits=4))

cm = confusion_matrix(y_test, tfl_int8w_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1], yticklabels=[0,1])
plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Confusion Matrix (TFLite INT8W)')
plt.show()

model.save("5.mlp_q.h5");