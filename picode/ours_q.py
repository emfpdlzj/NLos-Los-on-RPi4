# sa_train_export_quant_tflite.py
# ==========================================
# Self-Attention-Assisted Classifier (UWB)
# - Argmax Feature Selection (±50 → len 100)
# - 2-stage training (FE freeze)
# - PTQ: (1) INT8 weights + float32 activations  (I/O=float32)
#        (2) Full-INT8 (I/O=int8)  *옵션
# ==========================================

import os
os.makedirs("result", exist_ok=True)

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
x_val = x_train[25000:]; y_val = y_train[25000:]
x_train = x_train[:25000]; y_train = y_train[:25000]

# ---------------------------------------
# 2) Argmax Feature Selection & Padding
# ---------------------------------------
def argmax_window(X, w=50):
    out=[]
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

# -----------------------------
# 3) 모델 정의 (Feature Extractor → Self-Attention → Classifier)
# -----------------------------
L_IN  = 100
D_FE  = 16
SCALE = 1.0 / np.sqrt(D_FE)

inp = Input(shape=(L_IN,), name='input_1')                  # (None,100)
x   = L.Lambda(lambda z: tf.expand_dims(z, -1), name='expand')(inp)  # (None,100,1)

# Feature Extractor (Conv1D with kernel=1 ≒ Dense per time-step)
fe = L.Conv1D(30, 1, use_bias=True, name='fe_dense30')(x)
fe = L.BatchNormalization(name='fe_bn30')(fe)
fe = L.Conv1D(D_FE, 1, activation='relu', name='fe_dense16')(fe)
fe_out = fe

# Self-Attention
q = L.Conv1D(D_FE, 1, use_bias=True, name='sa_q')(fe_out); q = L.BatchNormalization(name='sa_q_bn')(q)
k = L.Conv1D(D_FE, 1, use_bias=True, name='sa_k')(fe_out); k = L.BatchNormalization(name='sa_k_bn')(k)
v = L.Conv1D(D_FE, 1, use_bias=True, name='sa_v')(fe_out); v = L.BatchNormalization(name='sa_v_bn')(v)

scores  = L.Lambda(lambda t: tf.matmul(t[0], t[1], transpose_b=True) * SCALE,
                   name='attn_scores')([q, k])                      # (None,100,100)
weights = L.Softmax(axis=-1, name='attn_softmax')(scores)
context = L.Lambda(lambda t: tf.matmul(t[0], t[1]), name='attn_ctx')([weights, v])  # (None,100,16)

# Classifier
flat = L.Flatten(name='flatten')(context)                   # (None,1600)
c    = L.Dense(42, activation='relu', name='cls_dense42')(flat)
c    = L.Dropout(0.5, name='drop1')(c)
c    = L.Dense(16, activation='relu', name='cls_dense16')(c)
c    = L.Dropout(0.5, name='drop2')(c)
out  = L.Dense(2, activation='softmax', name='output')(c)

model = Model(inp, out, name='SA_Assisted_Classifier')
model.compile(optimizer=Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

# -----------------------------
# 4) 학습 (1단계: end-to-end)
# -----------------------------
model.fit(x_train, y_train, epochs=8, batch_size=64,
          validation_data=(x_val, y_val), callbacks=[csv_logger], verbose=1)

# -----------------------------
# 5) 학습 (2단계: Feature Extractor 동결 후 미세조정)
# -----------------------------
for name in ['fe_dense30','fe_bn30','fe_dense16']:
    model.get_layer(name).trainable = False
model.compile(optimizer=Adam(5e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=4, batch_size=64, validation_data=(x_val, y_val), verbose=1)

# -----------------------------
# 6) Keras 기준 정확도
# -----------------------------
prob = model.predict(x_test, batch_size=256, verbose=0)
pred = np.argmax(prob, axis=1)
print("[Keras] acc :", accuracy_score(y_test, pred))
print(classification_report(y_test, pred, digits=4))
cm = confusion_matrix(y_test, pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1], yticklabels=[0,1])
plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Confusion Matrix (SA-PTQ, Keras)')
plt.tight_layout(); plt.savefig("result/sa_ptq_confmat_keras.png", dpi=150)

# -----------------------------
# 7) PTQ 내보내기
# -----------------------------
# 7-1) Dynamic-range (INT8 weights + float32 activations, I/O=float32) — 기본 권장
calib = x_val[:500].astype(np.float32)
def representative_dataset():
    for i in range(calib.shape[0]):
        yield [calib[i:i+1]]

conv_dr = tf.lite.TFLiteConverter.from_keras_model(model)
conv_dr.optimizations = [tf.lite.Optimize.DEFAULT]
conv_dr.representative_dataset = representative_dataset
conv_dr.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
    tf.lite.OpsSet.TFLITE_BUILTINS
]
tflite_int8w_floatact = conv_dr.convert()
open("sa_cls_int8w_floatact.tflite", "wb").write(tflite_int8w_floatact)

# 7-2) Full-INT8 (I/O=int8) — 더 작고 빠름(모든 연산의 int8 커널 필요, 실패 시 DR 사용)
conv_full = tf.lite.TFLiteConverter.from_keras_model(model)
conv_full.optimizations = [tf.lite.Optimize.DEFAULT]
conv_full.representative_dataset = representative_dataset
conv_full.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv_full.inference_input_type  = tf.int8
conv_full.inference_output_type = tf.int8
try:
    tflite_fullint8 = conv_full.convert()
    open("sa_cls_full_int8.tflite", "wb").write(tflite_fullint8)
    print("Saved: sa_cls_full_int8.tflite (I/O=int8)")
except Exception as e:
    print("Full-INT8 conversion failed, use dynamic-range model. Reason:", e)

print("Saved: sa_cls_int8w_floatact.tflite (I/O=float32)")
