## Self-Attention-Assisted TinyML for UWB NLoS Identification 재현 및 확장 실험 기록

---

## 0) 참고 베이스라인 (GitHub CNN)

* **Repo**: [LOS-NLOS-Classification-CNN](https://github.com/tycheyoung/LOS-NLOS-Classification-CNN)
* **차이점(Feature Selection)**

  * **논문**: `fp_index − 2 … + 47` → 총 **50** 길이
  * **본 실험(기본)**: `argmax` 기준 **±50** → 총 **100** 길이

```python
Nnew = []
for item in x_train:
    item = item[max([0, item.argmax()-50]) : item.argmax()+50]
    Nnew.append(item)
x_train = np.asarray(Nnew)
```

* **데이터 분할**

  * 논문 표준: **60 / 20 / 20**
  * 실험 편의: **25000 / 12000 / 5000** (고정 샘플 카운트)도 병행

<div align="center">
  <img src="./matrix/0.github.png" alt="Confusion Matrix (GitHub CNN)" width="55%" />
  <img src="./code/result/0.github.png" alt="Scores (GitHub CNN)" width="40%" />
</div>

---

## 1) CNN (Baseline)

**참고 논문**: *Improving Indoor Localization Using CNNs on Computationally Restricted Devices*

### 구조 / 학습 설정

* **입력**: 1D CIR(슬라이스) · **활성화**: ReLU
* **공간 축소**: MaxPooling (특징의 위치 민감도 완화)
* **가중치 규모 일정 유지** 가이드 반영(채널 수 조정)
* **출력**: FC → Softmax(2-class)
* **학습**: Adam, batch=**256**, Dropout=**0.5**

> **입력 구간**: 내부 라디오 알고리즘의 `first path index` 기준 **152 bins**(원 논문 설정). 본 실험에서는 위 전처리 옵션과 병행 비교.

<div align="center">
  <img src="./matrix/1.cnn.png" alt="Confusion Matrix (CNN)" width="55%" />
  <img src="./code/result/1.cnn.png" alt="Scores (CNN)" width="40%" />
</div>

---

## 2) CNN + LSTM 계열

**목적**: 시간적 상관을 LSTM으로 보완

### 공통 설정

* **CNN feature extractor** 이후에 LSTM 계열 부착
* LSTM hidden=**32**, lr=**1e-3**, Dropout=0.5, batch=64 또는 256
* Padding="valid", FC=128, Epochs=10~50 (논문·재현 난도에 따라 조정)

### 층수별 CNN 변형 (1~4층)

* 표기: Conv(p, q)=**p개 필터**, **kernel=q** / MaxPooling(a, b)=**pool=a, stride=b**

<div align="center">
  <img src="./matrix/2.cnn_1.png" alt="CNN-1층" width="48%" />
  <img src="./code/result/2.cnn_1.png" alt="Scores" width="48%" />

  <img src="./matrix/2.cnn_2.png" alt="CNN-2층" width="48%" />
  <img src="./code/result/2.cnn_2.png" alt="Scores" width="48%" />

  <img src="./matrix/2.cnn_3.png" alt="CNN-3층" width="48%" />
  <img src="./code/result/2.cnn_3.png" alt="Scores" width="48%" />

  <img src="./matrix/2.cnn_4.png" alt="CNN-4층" width="48%" />
  <img src="./code/result/2.cnn_4.png" alt="Scores" width="48%" />
</div>

### LSTM 변형 비교

* **CNN-LSTM** (단층)
* **CNN-Stacked-LSTM**
* **CNN-BiLSTM**

> 관찰: Stacked-LSTM이 소폭 향상, BiLSTM은 파라미터 증가로 학습 난도가 커져 오히려 약간 저하되는 경향(원문 서술과 일치).

<div align="center">
  <img src="./matrix/2.cnn_lstm.png" alt="CNN-LSTM Matrix" width="48%" />
  <img src="./code/result/2.cnn_lstm.png" alt="CNN-LSTM Scores" width="48%" />

  <img src="./matrix/2.cnn_stackedlstm.png" alt="CNN-Stacked-LSTM Matrix" width="48%" />
  <img src="./code/result/2.cnn_stackedlstm.png" alt="CNN-Stacked-LSTM Scores" width="48%" />
</div>

---

## 3) FCN & FCN-Attention & FCN-LSTM

### FCN 블록

* (Conv → BN → ReLU) × 3, 중간 **MaxPooling**으로 크기 축소

### Self-Attention 부착 (FCN-Attention)

* 동일 입력에서 **Q=K=V**(Self-Attention)로 특징 상호작용 강화 → FC로 투영

<div align="center">
  <img src="./matrix/3.fcn.png" alt="FCN Matrix" width="48%" />
  <img src="./code/result/3.fcn.png" alt="FCN Scores" width="48%" />

  <img src="./matrix/3.fcn_sat.png" alt="FCN-Attention Matrix" width="48%" />
  <img src="./code/result/3.fcn_sat.png" alt="FCN-Attention Scores" width="48%" />

  <img src="./matrix/3.fcn_lstm.png" alt="FCN-LSTM Matrix" width="48%" />
  <img src="./code/result/3.fcn_lstm.png" alt="FCN-LSTM Scores" width="48%" />
</div>

> **비고**: 일부 재현 결과에서 `CNN-LSTM > FCN > FCN-Attention` 순. 전처리 구간, 정규화, 초기화, 학습 스케줄러 등 영향 가능. 추가 점검 예정.

---

## 4) Depthwise CNN (Xception 스타일)

* Depthwise Separable Conv + Residual 연결
* UWB NLoS 분야 적용 리포트가 드물어 **동일 조건**으로 비교
* 양자화는 본 논문(타깃) 방식을 따름

---

## 5) MLP (경량 선형 계열)

* 전처리(특징 선택) 품질이 충분할 경우 **MLP도 강력한 베이스라인** 가능
* PC 측에서 다른 베이스라인 대비 경쟁력 확인 사례 존재

---

## 6) 타깃 논문 방식 (Self-Attention-Assisted TinyML)

### 방법 개요

* **사전학습 분류기**: FC×5 + BN×3 (초기 3개 층 **Freeze**)
* **재학습 블록**: Self-Attention + 축소된 Classifier (Adam, CE, batch=256, epochs=350)

### 양자화·배포

* **PTQ (weights-only INT8)**: 용량 1/4, 속도 향상
* **Full-INT8**: activation 포함 INT8 (캘리브레이션 샘플 **500**)
* **TFLite-Micro** 변환 및 C/C++ 소스 임베딩 옵션

---
