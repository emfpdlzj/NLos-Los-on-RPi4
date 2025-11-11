import tensorflow as tf
import uwb_dataset
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from keras.callbacks import CSVLogger
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense, Dropout

"""
Input shape       : (100, 1)                                                         |
Conv1D layers     : 3 layers (filters: 16 → 32 → 64, kernel: 3)                      |
Pooling           : MaxPooling1D (pool=2, stride=2)                                  |
Final layers      : GlobalMaxPooling → Dense(128) + Dropout(0.5) → Dense(1, sigmoid) |
Optimizer         : Adam                                                             |
Loss              : Binary Crossentropy                                              |
Batch size        : 64                                                               |
Epochs            : 10                                                               |
Feature selection : CIR에서 신호 피크 기준 좌우 50개 (총 100개)                                   |
Dataset split     : Train: 25,000 / Val: 5,000 / Test: 12,000                        |
"""
# import raw data
data = uwb_dataset.import_from_files()
csv_logger = CSVLogger("result/log.csv", append=False)

# Preamble(프리앰블): 통신에서 신호의 시작을 알리고, 수신 동기를 맞추기 위해 보내는 특별한 비트 패턴
# divide CIR by RX preable count (get CIR of single preamble pulse)
# item[2] represents number of acquired preamble symbols
for item in data:
    item[15:] = item[15:] / float(item[2])

# test data
train = data[:30000, :]  # 앞 에서부터 30000개와 모든열을 가져와 잘라냄  (2차원)
np.random.shuffle(train)  # 랜덤으로 섞음.
x_train = train[
    :30000, 15:
]  # train의 앞에서부터 30000개 row, 각 row의 15번부터 끝까지 슬라이싱.
y_train = train[:30000, 0]  # 앞 에서부터 30000개와 0개의 열을 가져와 잘라냄 (1차원)
x_test = data[30000:, 15:]
y_test = data[30000:, 0]

# feed data 검증데이터
x_val = x_train[25000:]  # x_val은 x_train의 250000번째부터 끝까지
y_val = y_train[25000:]
x_train = x_train[:25000]  # x_train의 앞에서 24999까지만 남김.
y_train = y_train[:25000]

Nnew = []
for item in x_train:
    item = item[max([0, item.argmax() - 50]) : item.argmax() + 50]
    # 신호가 강한 구간 기준 앞 뒤로 50개씩 총100개 구간으로 자른다.
    Nnew.append(item)
x_train = np.asarray(Nnew)
# np.assary(): 리스트를 numpy배열로 변환하는 함수

Nnew = []
for item in x_test:
    item = item[max([0, item.argmax() - 50]) : item.argmax() + 50]
    Nnew.append(item)
x_test = np.asarray(Nnew)

Nnew = []
for item in x_val:
    item = item[max([0, item.argmax() - 50]) : item.argmax() + 50]
    Nnew.append(item)
x_val = np.asarray(Nnew)

x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
x_val = x_val.reshape((x_val.shape[0], x_val.shape[1], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))


# total 1016 data for CIR
# define model
model = Sequential()

# 입력 shape: (샘플 수, 100, 1) → reshape 필요
# 1개의 hidden layer를 가진 3개의 MLP 구조

model.add(Conv1D(filters=16, kernel_size=3, activation="relu", input_shape=(100, 1)))
model.add(MaxPooling1D(pool_size=2, strides=2))
model.add(Conv1D(filters=32, kernel_size=3, activation="relu"))
model.add(MaxPooling1D(pool_size=2, strides=2))
model.add(Conv1D(filters=64, kernel_size=3, activation="relu"))

model.add(GlobalMaxPooling1D())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))  # LoS/NLoS → binary classification

print(model.summary())
# CNN기반의 분류 모델 ?

# compile
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
hist = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=64,
    validation_data=(x_val, y_val),
    callbacks=[csv_logger],
)
# epochs: 전체 훈련 데이터셋이 훈련 중 네트워크를 통해 몇 번이나 실행될 것인지 지정.
# validation_data : 검증 데이터셋을 지정하는 인수
# SGD: 가장 기본적인 최적화 알고리즘, 미분 가능함 함수에서 기울기를 계산하여 가중치를 업데이트한다.

# evaluation
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=64)
# batch_size: 정확도를 측정하고 가중치와 편향을 업데이트 하기 전에 네트워크에 공급할 훈련 데이터의 수를 지정
# 16또는 32로 시작해서 효과적인값을 실험해보는게 좋음.
print("## evaluation loss and metrics ##")
print(loss_and_metrics)

y_pred = model.predict(x_test)
y_pred = (y_pred > 0.5).astype(int)
print(classification_report(y_test, y_pred))
# macro avg: 클래스마다 F1, Precision, Recall을 구한 뒤, 단순 평균
# weighted avg: 각 클래스의 F1, Precision, Recall에 해당 클래스의 샘플 수로 가중치를 줘서 평균 낸 것

cm = confusion_matrix(y_test, y_pred)

# 시각화 - confusion matrix
plt.figure(figsize=(5, 4))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

tflite_model_path = "1_cnn_model.tflite"

try:
    # 1. TFLite 변환기 생성 (훈련된 Keras 모델 객체에서 바로 변환)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # 2. TFLite 미지원 연산을 TensorFlow Op로 포함 (안정성 확보)
    # 이전 오류를 우회하기 위해 이 옵션을 반드시 사용합니다.
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    print("\n✨ TFLite 변환기 설정 완료 (SELECT_TF_OPS 포함).")

    # 3. TFLite 모델로 변환
    tflite_model = converter.convert()

    # 4. .tflite 파일로 저장
    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)

    print(f"✅ 모델 훈련 후 TFLite 변환 및 저장 성공: {tflite_model_path}")

except Exception as e:
    print(f"❌ TFLite 변환 중 오류 발생: {e}")
