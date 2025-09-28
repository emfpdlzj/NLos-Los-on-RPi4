        import uwb_dataset
        import numpy as np
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        import matplotlib.pyplot as plt
        from keras.callbacks import CSVLogger
        from sklearn.metrics import classification_report
        from keras.models import Sequential
        from keras.layers import Dense, Embedding, Dropout, Conv1D, MaxPooling1D, GlobalMaxPooling1D

        # import raw data
        data = uwb_dataset.import_from_files()
        csv_logger = CSVLogger('result/log.csv', append=False)

        # Preamble(프리앰블): 통신에서 신호의 시작을 알리고, 수신 동기를 맞추기 위해 보내는 특별한 비트 패턴
        # divide CIR by RX preable count (get CIR of single preamble pulse)
        # item[2] represents number of acquired preamble symbols
        for item in data:
            item[15:] = item[15:]/float(item[2])

        # test data
        train = data[:30000, :] #앞 에서부터 30000개와 모든열을 가져와 잘라냄  (2차원)
        np.random.shuffle(train) #랜덤으로 섞음.
        x_train = train[:30000, 15:] #train의 앞에서부터 30000개 row, 각 row의 15번부터 끝까지 슬라이싱. 
        y_train = train[:30000, 0] #앞 에서부터 30000개와 0개의 열을 가져와 잘라냄 (1차원)
        x_test = data[30000:, 15:]
        y_test = data[30000:, 0]

        # feed data 검증데이터 
        x_val = x_train[25000:] #x_val은 x_train의 25000번째부터 끝까지
        y_val = y_train[25000:]
        x_train = x_train[:25000] #x_train의 앞에서 24999까지만 남김. 
        y_train = y_train[:25000]

        Nnew=[]
        for item in x_train:
            item = item[max([0,item.argmax()-50]) : item.argmax()+50] 
            #신호가 강한 구간 기준 앞 뒤로 50개씩 총100개 구간으로 자른다. 
            Nnew.append(item)
        x_train = np.asarray(Nnew) 
        # np.assary(): 리스트를 numpy배열로 변환하는 함수

        Nnew=[]
        for item in x_test:
            item = item[max([0,item.argmax()-50]) : item.argmax()+50]
            Nnew.append(item)
        x_test = np.asarray(Nnew)

        Nnew=[]
        for item in x_val:
            item = item[max([0,item.argmax()-50]) : item.argmax()+50]
            Nnew.append(item)
        x_val = np.asarray(Nnew)

        # x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1)).transpose(2,0,1)
        # x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1)).transpose(2,0,1)
        # x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1)).transpose(2,0,1)

        # total 1016 data for CIR
        # define model
        model = Sequential() # 케라스를 사용하여 sequential 모델 생성, 
# Sequential은 입력층부터 출력층까지 쌓아감. 
        model.add(Embedding(2700, 128, input_length=100)) # 2700개의 단어를 128차원으로 변환, 입력의 길이는 100개의 정수
        model.add(Conv1D(filters=10, kernel_size=4, padding='valid', activation='relu'))
        #filter: 필터(출력 채널)의 수, kernel_size: 필터 크기, padding:경계처리방법, activation: 활성화 함수 
        model.add(Conv1D(20, 5, padding='valid', activation='relu'))
        model.add(MaxPooling1D(pool_size=2, strides=2, padding='valid')) 
        # maxpooling : 가져온 윈도우 값 중 가장 큰 값을 사용한다. 
        # 입력 데이터를 작게 만들면서도, 입력 데이터의 중요한 특징을 보존할 수 있다.
        # stride: 윈도우 생성 시, 몇 칸을 뛸 건지를 설정해주는 파라미터
        model.add(Conv1D(20, 4, padding='valid', activation='relu'))
        model.add(Conv1D(40, 4, padding='valid', activation='relu'))
        model.add(GlobalMaxPooling1D())#  1차원의 입력 데이터에서 가장 큰 값을 선택하여 출력하는 레이어. 
        model.add(Dense(128, activation='relu'))
        # dense:  입력 뉴런과 출력 뉴런이 모두 연결되어 있는 밀집(dense)구조의 fully connected layer
        # 입력 벡터와 가중치 행렬의 곱셈을 더하여 활성화 함수를 거친 출력 값을 계산한다. 
        # 활성화 함수로는 보통 ReLU, sigmoid, tanh가 사용된다. 
        model.add(Dropout(0.5)) 
        #regularization 기법 중 하나로, 무작위로 일부 뉴런을 선택해 출력을 0으로 만듬.여기서는 50%의 뉴런을 0으로 만든다.
        model.add(Dense(1, activation='sigmoid'))
        # unit이 1개이고 활성화 함수가 sigmoid인 dense layer
        print(model.summary())
        #CNN기반의 분류 모델 ?

        # compile
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # binary_crossentropy:  0 또는 1을 맞히는 이진 분류 문제에서 모델의 예측이 얼마나 틀렸는지 측정하는 함수
        # metrics=['accuray']: 모델을 정확도 기준으로 평가함  
        # optimizer = adam. adam: RMsprop과 모멘텀을 결합하여 만든 최적화 알고리즘. 기울기 제곱값과 모멘텀의 이동평균 값을 조절하여 학습률을 조정한다. 
        hist = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_val, y_val),callbacks=[csv_logger])
        # x_train, y_train: 모델의 입력 데이터 
        # epochs: 전체 훈련 데이터셋이 훈련 중 네트워크를 통해 몇 번이나 실행될 것인지 지정.
        # validation_data : 검증 데이터셋을 지정하는 인수

        # evaluation
        loss_and_metrics = model.evaluate(x_test, y_test, batch_size=64)
        # evaluate: 학습된 모델을 사용하여 주어진 데이터셋에 대한 손실값과 지정된 평가 지표를 계산한다. 
        # batch_size: 정확도를 측정하고 가중치와 편향을 업데이트 하기 전에 네트워크에 공급할 훈련 데이터의 수를 지정
        # 16또는 32로 시작해서 효과적인값을 실험해보는게 좋음.
        print('## evaluation loss and metrics ##')
        print(loss_and_metrics)

        model.save("sliced_cnn.h5");
        
        y_pred = model.predict(x_test)
        #model.predict() : 모델의 예측 결과를 반환한다. 
        y_pred = (y_pred > 0.5).astype(int)
        #astype() : 열의 요소의 dtype을 변경하는 메소드.
        print(classification_report(y_test, y_pred))
        # macro avg: 클래스마다 F1, Precision, Recall을 구한 뒤, 단순 평균 구함.
        # weighted avg: 각 클래스의 F1, Precision, Recall에 해당 클래스의 샘플 수로 가중치를 줘서 평균 낸 것

        cm = confusion_matrix(y_test, y_pred)

        # 시각화 - confusion matrix
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1], yticklabels=[0,1])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()

