import uwb_dataset
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from keras.callbacks import CSVLogger
from keras.models import Model
from keras.layers import (Input, SeparableConv1D, BatchNormalization, Activation,
                          MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout,
                          Add, Conv1D)
from keras.optimizers import Adam

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

def slice_window(arr):
    out=[]
    for row in arr:
        a = int(row.argmax())
        out.append(row[max(0,a-50):a+50])
    return np.asarray(out)

x_train = slice_window(x_train)[..., np.newaxis]  # (N,100,1)
x_val   = slice_window(x_val)[...,   np.newaxis]
x_test  = slice_window(x_test)[...,  np.newaxis]

# ------------------------
# 2) Xception-1D (사진 구조 동일)
#    Entry → Middle(8회 반복) → Exit → GAP → FC → Sigmoid
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

# Middle flow: 8 times (no downsample, residual identity)
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
plt.show()

model.save("4.depthwise_cnn.h5");