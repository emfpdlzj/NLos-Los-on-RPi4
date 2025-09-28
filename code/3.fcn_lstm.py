import uwb_dataset
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from keras.callbacks import CSVLogger
from keras.models import Model
from keras.layers import (Input, Conv1D, BatchNormalization, Activation,
                        GlobalAveragePooling1D, Dense, Dropout,
                          LSTM, Permute, Concatenate)
from keras.optimizers import Adam

#  데이터 로드 & 전처리 
data = uwb_dataset.import_from_files()
csv_logger = CSVLogger('result/log.csv', append=False)

for item in data:
    item[15:] = item[15:] / float(item[2])

train = data[:30000, :]
np.random.shuffle(train)
x_train = train[:30000, 15:]
y_train = train[:30000, 0]
x_test  = data[30000:, 15:]
y_test  = data[30000:, 0]

x_val = x_train[25000:]
y_val = y_train[25000:]
x_train = x_train[:25000]
y_train = y_train[:25000]

def slice_window(arr):
    Nnew=[]
    for item in arr:
        idx = int(item.argmax())
        start = max(0, idx - 50)
        end   = idx + 50
        Nnew.append(item[start:end])
    return np.asarray(Nnew)

x_train = slice_window(x_train)
x_val   = slice_window(x_val)
x_test  = slice_window(x_test)

# Conv1D/LSTM 입력 차원: (samples, timesteps, channels)
x_train = x_train[..., np.newaxis]  # -> (N, 100, 1)
x_val   = x_val[..., np.newaxis]
x_test  = x_test[..., np.newaxis]


# FCN-LSTM 모델 (Functional API)

inp = Input(shape=(100, 1))

# FCN 
x = Conv1D(128, 8, padding='valid', strides=1)(inp)
x = BatchNormalization()(x); x = Activation('relu')(x)

x = Conv1D(256, 5, padding='valid', strides=1)(x)
x = BatchNormalization()(x); x = Activation('relu')(x)

x = Conv1D(128, 3, padding='valid', strides=1)(x)
x = BatchNormalization()(x); x = Activation('relu')(x)

gap = GlobalAveragePooling1D(name="fcn_gap")(x)

# LSTM 
# Dimension shuffle: (timesteps, channels) -> (channels, timesteps)
p = Permute((2, 1), name="dim_shuffle")(inp)  # (1, 100)

# units는 데이터 크기에 맞춰 조절해야함. (작으면 underfit, 크면 overfit)
lstm_out = LSTM(64, name="lstm_block")(p)

# Concatenate & Classifier 
h = Concatenate(name="concat_gap_lstm")([gap, lstm_out])
h = Dense(128, activation='relu')(h)
h = Dropout(0.5)(h)
out = Dense(1, activation='sigmoid')(h)

model = Model(inputs=inp, outputs=out)
model.compile(loss='binary_crossentropy', optimizer=Adam(1e-3), metrics=['accuracy'])
model.summary()

# 학습
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=64,
    validation_data=(x_val, y_val),
    callbacks=[csv_logger],
    verbose=1
)

# 평가 & 리포트
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=64)
print('## evaluation loss and metrics ##')
print(loss_and_metrics)

y_pred = (model.predict(x_test) > 0.5).astype(int)
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1], yticklabels=[0,1])
plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Confusion Matrix')
plt.show()

model.save("3.fcn_lstm.h5");