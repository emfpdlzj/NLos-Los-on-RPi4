import uwb_dataset
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from keras.callbacks import CSVLogger
from keras.models import Model
from keras.layers import Input, Conv1D, BatchNormalization, Activation, MaxPooling1D
from keras.layers import GlobalMaxPooling1D, Dense, Dropout, Multiply, Softmax, Add

# 데이터 로드 및 전처리
data = uwb_dataset.import_from_files()
csv_logger = CSVLogger('result/log.csv', append=False)

for item in data:
    item[15:] = item[15:] / float(item[2]) 

# 학습/테스트 분할
train = data[:30000, :]
np.random.shuffle(train)
x_train = train[:30000, 15:]
y_train = train[:30000, 0]
x_test = data[30000:, 15:]
y_test = data[30000:, 0]

# 검증 데이터 분리
x_val = x_train[25000:]
y_val = y_train[25000:]
x_train = x_train[:25000]
y_train = y_train[:25000]

# argmax 기준 슬라이싱
def slice_by_peak(data):
    result = []
    for item in data:
        peak = item.argmax()
        start = max(0, peak - 50)
        end = peak + 50
        result.append(item[start:end])
    return np.asarray(result)

x_train = slice_by_peak(x_train)
x_val = slice_by_peak(x_val)
x_test = slice_by_peak(x_test)

# (샘플 수, 100) -> (샘플 수, 100, 1)
x_train = x_train[..., np.newaxis]
x_val = x_val[..., np.newaxis]
x_test = x_test[..., np.newaxis]

# Self-Attention Block 정의     
def self_attention_block(x):
    f1 = Conv1D(16, 1, padding='same')(x)  # Query
    f2 = Conv1D(16, 1, padding='same')(x)  # Key
    f3 = Conv1D(16, 1, padding='same')(x)  # Value

    f2_softmax = Softmax(axis=1)(f2)
    attn = Multiply()([f1, f2_softmax])
    out = Add()([attn, f3])  # Skip connection
    return out

# FCN + Attention 모델 정의 
inputs = Input(shape=(100, 1))
x = Conv1D(64, 9, strides=1, padding='valid')(inputs)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = Conv1D(64, 7, strides=1, padding='valid')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = MaxPooling1D(pool_size=2, strides=2)(x)

x = Conv1D(128, 5, strides=1, padding='valid')(x)
x = BatchNormalization()(x)

x = self_attention_block(x)  # ⭐ Self-Attention 추가

x = GlobalMaxPooling1D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs=inputs, outputs=outputs)
print(model.summary())

#모델 학습
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_val, y_val), callbacks=[csv_logger])

# 평가  
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=64)
print('## evaluation loss and metrics ##')
print(loss_and_metrics)

y_pred = model.predict(x_test)
y_pred = (y_pred > 0.5).astype(int)
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

model.save("3.fcn_sat.h5");