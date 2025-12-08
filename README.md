# TinyML LoS/NLoS Classification on Raspberry Pi 4 

#### target papger:Self-Attention-Assisted TinyML for UWB NLoS Identification <br>
UWB **CIR (Channel Impulse Response)** 데이터를 이용해 **LoS / NLoS**를 분류하고, 학습된 모델을 **Raspberry Pi 4**에서 **TinyML (TensorFlow Lite / TFLite-Micro)** 로 구동하는 연구용 저장소입니다.  
베이스라인은 1D-CNN이며, **FCN, CNN-LSTM, CNN-stacked-LSTM, CNN-bi-LSTM, FCN-Attention**, 그리고 타깃 논문 방식의 **Self-Attention-Assisted TinyML**까지 비교합니다.

> **진행 상태:** 연구 기록은 `report.md`에, 타깃 논문 요약은 `+타깃논문.md`에 정리되어 있습니다.  

기간: 25.7.8 ~ 25.8.31
---

## ✨ 핵심 요약 (TL;DR)

- **데이터**: eWINE UWB CIR (LoS/NLoS) → 정규화 & 슬라이싱 → 특성 행렬
- **모델**: 1D-CNN(기본), FCN, CNN-LSTM/Stacked/Bi-LSTM, FCN-Attention, Self-Attention-Assisted MLP
- **전처리**:  
  - (기본) `argmax` 기준 ±50(총 100) 윈도우  
  - (논문) `fp_index −2 … +47`(총 50) 윈도우  
- **분할**: 60/20/20(논문) 또는 고정 개수(예: 25k/12k/5k) 실험 병행
- **경량화**: TFLite **PTQ** (weights-only int8, **Full-INT8**: 500 샘플 캘리브레이션)  
- **배포**: Raspberry Pi 4 + `tflite-runtime`로 실시간 추론

---

## 📂 저장소 구조

tinymllab/ <br>
├─ code/        # PC 학습·평가·변환 스크립트 (전처리, 학습, TFLite 변환 등). <br>
├─ dataset/     # (비공개) 원천/가공 데이터. <br>
├─ image/       # 아키텍처 도식, 논문 인용 그림 등<br>
├─ matrix/      # confusion matrix<br>
├─ picode/      # Raspberry Pi 4 추론 스크립트 (tflite-runtime)<br>
├─ report.md    # 실험 기록/메모 (모델별 결과 스크린샷 포함)<br>
├─ +타깃논문.md  # TinyML 리뷰 논문 요약 (세미나 정리본)<br>
└─ README.md<br>

---

## 🧪 데이터셋

- **출처**: eWINE 프로젝트 — *UWB LOS/NLOS Data Set* (CC-BY-4.0)  
- **특징**: 7개 실내 환경에서 수집한 UWB CIR (LoS/NLoS)  

### 전처리 방식

1) Argmax 기반 (일반적 방법)
```python
# 강한 신호 지점(argmax) 기준으로 앞뒤 50씩 총 100 길이
Nnew = []
for item in x_train:
    item = item[max([0, item.argmax()-50]) : item.argmax()+50]
    Nnew.append(item)
x_train = np.asarray(Nnew)
```

2. **논문 기준 (fp_index)**

	•	fp_index − 2 … fp_index + 47 → 총 50 길이 <br>
	•	최종 실험 6번 파트에서 fp_index 기준도 별도 비교

데이터 분할  
	•	논문 기본: 60 / 20 / 20  
	•	실험 반복성: 25,000 / 12,000 / 5,000 샘플로도 병행 평가  

⸻

🏗️ 모델 구성
```
베이스라인 및 변형들
1D-CNN
	• Conv1D + ReLU, MaxPooling(공간 축소), FC + Softmax(2-class)
	• 가중치 수를 층 간 일정하게 유지하도록 채널 수 조절 (논문 권고)
	• 최적화: Adam, 배치 256, Dropout 0.5

CNN-LSTM / CNN-Stacked-LSTM / CNN-Bi-LSTM
	• 동일 CNN feature extractor 뒤에 LSTM (hidden=32, lr=1e-3)
	• 논문에 구체 레이어 스펙은 없어 관례적 설계로 구현, 층 수(1~4) 비교

FCN / FCN-Attention
	• [FCN] Conv-BN-ReLU 블록 ×3 + 중간 MaxPooling
	• [FCN-Attention] FCN feature 뒤 Self-Attention 블록 추가

Depthwise CNN (Xception 스타일)
	• Depthwise Separable Conv + Residual

MLP 

Self-Attention-Assisted TinyML - 타깃 논문 방식
	•사전학습 분류기(FC×5 + BN×3)에서 초기 3개 층 Freeze
	•그 위에 Self-Attention + 축소된 분류기를 재학습
	•최적화 Adam, CE loss, batch 256, epochs 350
	•PTQ + Full-INT8(QAT 대체 가능)로 임베디드 추론 최적화
```

report.md에 각 모델의 혼동행렬(matrix/*.png)과 점수 그래프(code/result/*.png)가 포함되어 있습니다.

---

⚙️ 환경

PC (학습/변환)<br>
	•	Python 3.10+<br>
	•	주요 패키지: numpy, pandas, scikit-learn, tensorflow (2.13~2.15 권장), matplotlib <br>

Raspberry Pi 4 (추론)<br>

```
python3 -m pip install --upgrade tflite-runtime
```

---

🔗 참고 자료 (References) <br>
	•	데이터셋: eWINE — UWB LOS/NLOS Data Set (CC-BY-4.0)<br>
	•	GitHub: https://github.com/ewine-project/UWB-LOS-NLOS-Data-Set<br>
	•	베이스라인 CNN 구현: https://github.com/tycheyoung/LOS-NLOS-Classification-CNN <br>
	•	Self-Attention-Assisted TinyML for UWB NLoS Identification (타깃 논문) <br>
	•	TinyML Review: A review on TinyML: State-of-the-art and prospects (Partha Pratim Ray, 2021)<br>

논문/그림 인용은 원 저작권을 따르며, 본 저장소의 코드/노트는 연구 재현을 목적으로 합니다.



