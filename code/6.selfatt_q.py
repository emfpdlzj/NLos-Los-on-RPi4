# Self-Attention Assisted Classifier (argmax feature selection 포함)
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from keras.callbacks import CSVLogger
from keras.optimizers import Adam
from keras import layers as L, Model, Input
import uwb_dataset

# 0) 데이터 로드 & 프리앰블 정규화
data = uwb_dataset.import_from_files()
csv_logger = CSVLogger('result/log_sa.csv', append=False)

for item in data:
    item[15:] = item[15:] / float(item[2])

# 1) Split
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

# 2) Feature selection: argmax 기준 ±50 (길이 100), 부족분은 0-padding
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

# 3) 모델 구성 (Functional API)
L_IN  = 100                 # 윈도우 길이
D_FE  = 16                  # feature extractor 출력 차원
SCALE = 1.0 / np.sqrt(D_FE) # attention scaling (1/sqrt(d_k))

inp = Input(shape=(L_IN,), name='input_1')     # (None, 100)
x   = L.Lambda(lambda z: tf.expand_dims(z, -1), name='expand')(inp)  # (None,100,1)

# --- Feature Extraction (Dense(30)→BN→Dense(16,ReLU)) ---
fe = L.Conv1D(30, 1, use_bias=True, name='fe_dense30')(x)           # (None,100,30)
fe = L.BatchNormalization(name='fe_bn30')(fe)
fe = L.Conv1D(D_FE, 1, activation='relu', name='fe_dense16')(fe)    # (None,100,16)
fe_out = fe                                                         # Feature map (Frozen 단계에서 고정)

# --- Self-Attention (Q/K/V: 16→16, BN 각자) ---
q = L.Conv1D(D_FE, 1, use_bias=True, name='sa_q')(fe_out)
q = L.BatchNormalization(name='sa_q_bn')(q)
k = L.Conv1D(D_FE, 1, use_bias=True, name='sa_k')(fe_out)
k = L.BatchNormalization(name='sa_k_bn')(k)
v = L.Conv1D(D_FE, 1, use_bias=True, name='sa_v')(fe_out)
v = L.BatchNormalization(name='sa_v_bn')(v)

# scores = softmax( (Q K^T)/sqrt(d) ), context = scores @ V
scores = L.Lambda(lambda t: tf.matmul(t[0], t[1], transpose_b=True) * SCALE,
                  name='attn_scores')([q, k])                         # (None,100,100)
weights = L.Softmax(axis=-1, name='attn_softmax')(scores)            # (None,100,100)
context = L.Lambda(lambda t: tf.matmul(t[0], t[1]), name='attn_ctx')([weights, v])  # (None,100,16)

# --- Classifier (Flatten→Dense(42)→Dropout→Dense(16)→Dropout→Dense(2,Softmax)) ---
flat = L.Flatten(name='flatten')(context)                            # (None, 100*16=1600)
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

# 4) 1단계: end-to-end 학습
model.fit(
    x_train, y_train,
    epochs=8, batch_size=64,
    validation_data=(x_val, y_val),
    callbacks=[csv_logger],
    verbose=1
)

# 5) 2단계: Feature Extractor 동결 후(그림의 Frozen) 미세조정
for layer_name in ['fe_dense30','fe_bn30','fe_dense16']:
    model.get_layer(layer_name).trainable = False

model.compile(optimizer=Adam(5e-4),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(
    x_train, y_train,
    epochs=4, batch_size=64,
    validation_data=(x_val, y_val),
    verbose=1
)

# 6) 평가
prob = model.predict(x_test, batch_size=256, verbose=0)
pred = np.argmax(prob, axis=1)
print("Test accuracy :", accuracy_score(y_test, pred))
print(classification_report(y_test, pred, digits=4))

cm = confusion_matrix(y_test, pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1], yticklabels=[0,1])
plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Confusion Matrix (Self-Attention)')
plt.show()

model.save("6.selfatt.h5")  # 원본 Keras 모델 저장

# === PTQ: 가중치 int8 / activation float32 (Dynamic Range Quantization) ===
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]   # DRQ 트리거 (rep dataset 불필요)
tflite_model = converter.convert()

with open("6.selfatt.drq.tflite", "wb") as f:
    f.write(tflite_model)

print("Saved TFLite (DRQ) → 6.selfatt.drq.tflite")
