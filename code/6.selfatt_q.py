# ==========================================
# Self-Attention-Assisted Classifier (UWB)
# - Argmax Feature Selection (±50 → len 100)
# - 2-stage training (FE freeze)
# - PTQ: int8 weights + float32 activations
# - TFLite inference & accuracy comparison
# ==========================================

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from keras.callbacks import CSVLogger
from keras.optimizers import Adam
from keras import layers as L, Model, Input
import uwb_dataset

# -----------------------------
# 0) 데이터 로드 & 프리앰블 정규화
# -----------------------------
data = uwb_dataset.import_from_files()
csv_logger = CSVLogger('result/log_sa_ptq.csv', append=False)

for item in data:
    # item[2] = preamble symbols count
    item[15:] = item[15:] / float(item[2])

# -----------------------------
# 1) Split
# -----------------------------
train = data[:30000, :]
np.random.shuffle(train)

x_train = train[:30000, 15:]
y_train = train[:30000, 0].astype(int)

x_test  = data[30000:, 15:]
y_test  = data[30000:, 0].astype(int)

# validation
x_val = x_train[25000:]
y_val = y_train[25000:]
x_train = x_train[:25000]
y_train = y_train[:25000]

# ---------------------------------------
# 2) Argmax Feature Selection & Padding
# ---------------------------------------
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

# -----------------------------
# 3) 모델 정의 (Functional API)
#     Feature Extractor → Self-Attention → Classifier
# -----------------------------
L_IN  = 100    # input length
D_FE  = 16     # feature dim after extractor
SCALE = 1.0 / np.sqrt(D_FE)

inp = Input(shape=(L_IN,), name='input_1')                 # (None,100)
x   = L.Lambda(lambda z: tf.expand_dims(z, -1), name='expand')(inp)  # (None,100,1)

# --- Feature Extraction (Dense(30)→BN→Dense(16,ReLU)) ---
fe = L.Conv1D(30, 1, use_bias=True, name='fe_dense30')(x)           # (None,100,30)
fe = L.BatchNormalization(name='fe_bn30')(fe)
fe = L.Conv1D(D_FE, 1, activation='relu', name='fe_dense16')(fe)    # (None,100,16)
fe_out = fe

# --- Self-Attention (Q/K/V: 16→16) ---
q = L.Conv1D(D_FE, 1, use_bias=True, name='sa_q')(fe_out)
q = L.BatchNormalization(name='sa_q_bn')(q)
k = L.Conv1D(D_FE, 1, use_bias=True, name='sa_k')(fe_out)
k = L.BatchNormalization(name='sa_k_bn')(k)
v = L.Conv1D(D_FE, 1, use_bias=True, name='sa_v')(fe_out)
v = L.BatchNormalization(name='sa_v_bn')(v)

scores  = L.Lambda(lambda t: tf.matmul(t[0], t[1], transpose_b=True) * SCALE,
                   name='attn_scores')([q, k])                      # (None,100,100)
weights = L.Softmax(axis=-1, name='attn_softmax')(scores)           # (None,100,100)
context = L.Lambda(lambda t: tf.matmul(t[0], t[1]),
                   name='attn_ctx')([weights, v])                   # (None,100,16)

# --- Classifier (Flatten→Dense(42)→Dropout→Dense(16)→Dropout→Dense(2,Softmax)) ---
flat = L.Flatten(name='flatten')(context)                            # (None,1600)
c    = L.Dense(42, activation='relu', name='cls_dense42')(flat)
c    = L.Dropout(0.5, name='drop1')(c)
c    = L.Dense(16, activation='relu', name='cls_dense16')(c)
c    = L.Dropout(0.5, name='drop2')(c)
out  = L.Dense(2, activation='softmax', name='output')(c)

model = Model(inp, out, name='SA_Assisted_Classifier')
model.compile(optimizer=Adam(1e-3),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
print(model.summary())

# -----------------------------
# 4) 학습 (1단계: end-to-end)
# -----------------------------
model.fit(
    x_train, y_train,
    epochs=8,
    batch_size=64,
    validation_data=(x_val, y_val),
    callbacks=[csv_logger],
    verbose=1
)

# -----------------------------
# 5) 학습 (2단계: Feature Extractor Frozen)
# -----------------------------
for name in ['fe_dense30', 'fe_bn30', 'fe_dense16']:
    model.get_layer(name).trainable = False

model.compile(optimizer=Adam(5e-4),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(
    x_train, y_train,
    epochs=4,
    batch_size=64,
    validation_data=(x_val, y_val),
    verbose=1
)
model.save("6.selfatt_q.h5");
# -----------------------------
# 6) Keras 기준 정확도
# -----------------------------
keras_prob = model.predict(x_test, batch_size=256, verbose=0)
keras_pred = np.argmax(keras_prob, axis=1)
print("[Keras] acc :", accuracy_score(y_test, keras_pred))
print(classification_report(y_test, keras_pred, digits=4))

# -----------------------------
# 7) PTQ (weights int8 + activations float32)
#     calibration set: 500 샘플
# -----------------------------
calib = x_val[:500].astype(np.float32)

def representative_dataset():
    for i in range(calib.shape[0]):
        yield [calib[i:i+1]]

# FP32 TFLite (baseline)
conv_fp32 = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_fp32 = conv_fp32.convert()
open("sa_cls_fp32.tflite", "wb").write(tflite_fp32)

# PTQ: int8 weights + float32 activations (dynamic-range)
conv_int8w = tf.lite.TFLiteConverter.from_keras_model(model)
conv_int8w.optimizations = [tf.lite.Optimize.DEFAULT]
conv_int8w.representative_dataset = representative_dataset
conv_int8w.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
    tf.lite.OpsSet.TFLITE_BUILTINS
]
tflite_int8w = conv_int8w.convert()
open("sa_cls_int8w_floatact.tflite", "wb").write(tflite_int8w)

print("FP32 size (KB):", len(tflite_fp32)/1024)
print("INT8W size (KB):", len(tflite_int8w)/1024)

# -----------------------------
# 8) TFLite 추론 유틸
# -----------------------------
def tflite_predict(tflite_model_bytes, X):
    interpreter = tf.lite.Interpreter(model_content=tflite_model_bytes)
    interpreter.allocate_tensors()
    in_det  = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]
    input_idx  = in_det['index']
    output_idx = out_det['index']

    # dynamic-range PTQ는 보통 float32 입출력
    if in_det['dtype'] == np.int8:
        # (full-int8 대비 안전용 처리)
        scale, zp = in_det['quantization']
        Xq = np.clip(np.round(X/scale + zp), -128, 127).astype(np.int8)
        feed = Xq
    else:
        feed = X.astype(np.float32)

    preds = []
    for i in range(X.shape[0]):
        interpreter.set_tensor(input_idx, feed[i:i+1])
        interpreter.invoke()
        out = interpreter.get_tensor(output_idx)
        if out_det['dtype'] == np.int8:
            os, oz = out_det['quantization']
            out = (out.astype(np.float32) - oz) * os
        preds.append(out[0])
    return np.vstack(preds)

# -----------------------------
# 9) 정확도 비교 (Keras vs TFLite FP32 vs TFLite PTQ)
# -----------------------------
tfl_fp32_prob = tflite_predict(tflite_fp32, x_test)
tfl_fp32_pred = np.argmax(tfl_fp32_prob, axis=1)
print("[TFLite FP32] acc :", accuracy_score(y_test, tfl_fp32_pred))

tfl_int8w_prob = tflite_predict(tflite_int8w, x_test)
tfl_int8w_pred = np.argmax(tfl_int8w_prob, axis=1)
print("[TFLite INT8-weights (float act)] acc :", accuracy_score(y_test, tfl_int8w_pred))
print(classification_report(y_test, tfl_int8w_pred, digits=4))

# -----------------------------
# 10) Confusion Matrix (PTQ 모델)
# -----------------------------
cm = confusion_matrix(y_test, tfl_int8w_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1], yticklabels=[0,1])
plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Confusion Matrix (SA-PTQ)')
plt.tight_layout()
plt.show()

# -----------------------------
# (옵션) Full-INT8 템플릿
# -----------------------------
# conv_fullint8 = tf.lite.TFLiteConverter.from_keras_model(model)
# conv_fullint8.optimizations = [tf.lite.Optimize.DEFAULT]
# conv_fullint8.representative_dataset = representative_dataset
# conv_fullint8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# conv_fullint8.inference_input_type  = tf.int8
# conv_fullint8.inference_output_type = tf.int8
# tflite_fullint8 = conv_fullint8.convert()
# open("sa_cls_full_int8.tflite", "wb").write(tflite_fullint8)
