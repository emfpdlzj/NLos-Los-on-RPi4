import uwb_dataset
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from keras.callbacks import CSVLogger
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam

# 1) 데이터 로드 & 프리앰블 정규화
data = uwb_dataset.import_from_files()
csv_logger = CSVLogger('result/log.csv', append=False)

for item in data:
    item[15:] = item[15:] / float(item[2])

# 2) train / test split
train = data[:30000, :]
np.random.shuffle(train)
x_train = train[:30000, 15:]
y_train = train[:30000, 0].astype(int)

x_test  = data[30000:, 15:]
y_test  = data[30000:, 0].astype(int)

# 3) 검증셋 분리
x_val = x_train[25000:]
y_val = y_train[25000:]
x_train = x_train[:25000]
y_train = y_train[:25000]

# 4) Feature selection: argmax 기준 앞뒤 50 → 길이 100
def argmax_window(X, w=50):
    out = []
    for row in X:
        m = int(row.argmax())
        s = max(0, m - w)
        e = m + w
        out.append(row[s:e])
    return np.asarray(out)

x_train = argmax_window(x_train, 50)
x_val   = argmax_window(x_val, 50)
x_test  = argmax_window(x_test, 50)

# (안전) 길이가 100이 아닐 수도 있는 케이스를 0-padding
def pad_to_100(X):
    if X.shape[1] == 100:
        return X
    Z = np.zeros((X.shape[0], 100), dtype=X.dtype)
    L = min(100, X.shape[1])
    Z[:, :L] = X[:, :L]
    return Z

x_train = pad_to_100(x_train)
x_val   = pad_to_100(x_val)
x_test  = pad_to_100(x_test)

# 5) 사진과 유사한 MLP 구조
#   Dense(30) → BN → Dense(16, ReLU) → Dense(64, ReLU) → BN
#   → Dense(16, ReLU) → BN → Dropout → Dense(2, Softmax)
model = Sequential()
model.add(Dense(30, input_shape=(100,), use_bias=True))  # (100×30)
model.add(BatchNormalization())                          # BN(30)
model.add(Dense(16, activation='relu'))                  # (30×16) + ReLU
model.add(Dense(64, activation='relu'))                  # (16×64) + ReLU
model.add(BatchNormalization())                          # BN(64)
model.add(Dense(16, activation='relu'))                  # (64×16) + ReLU
model.add(BatchNormalization())                          # BN(16)
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))                # (16×2) + Softmax

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(),
    metrics=['accuracy']
)

print(model.summary())

# 6) 학습
hist = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=64,
    validation_data=(x_val, y_val),
    callbacks=[csv_logger]
)

# 7) 평가
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=64)
print('## evaluation loss and metrics ##')
print(loss_and_metrics)

# 8) 예측 및 리포트
y_prob = model.predict(x_test)
y_pred = np.argmax(y_prob, axis=1)

print(classification_report(y_test, y_pred, digits=4))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1], yticklabels=[0,1])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

model.save("5.mlp.h5");