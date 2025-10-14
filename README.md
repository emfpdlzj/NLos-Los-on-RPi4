# TinyML LoS/NLoS Classification on Raspberry Pi 4 (WIP)(미완)

UWB CIR(Channel Impulse Response) 데이터를 이용해 **LoS / NLoS** 를 분류하고, 학습된 모델을 **Raspberry Pi 4** 에서 **TinyML(TensorFlow Lite)** 로 구동하는 연구 저장소입니다.

> 목표: 임베디드 환경에서 **가벼운 모델**로 **실시간 분류**를 달성하고, **전처리–학습–경량화–배포**를 일관된 파이프라인으로 정리하기.

기본 베이스라인은 **1D-CNN**. 추가 실험으로 **FCN, CNN-LSTM, CNN-stacked-LSTM, CNN-bi-LSTM, FCN-Attention** 등을 비교합니다.

---

## TL;DR

* 데이터: UWB CIR → 정규화/슬라이싱 → 특성 행렬
* 모델: CNN 기반(추가 실험: FCN, CNN-LSTM, FCN-Attention 등)
* 경량화: TFLite(PTQ), 필요 시 QAT
* 배포: RPi4 + `tflite-runtime` 로 실시간 추론
* 진행상태: **연구 진행중(WIP)** – 실험 로그/보고서는 `report.md` 

---

## Repository Structure

```
tinymllab/
├─ code/        # PC 학습/평가/변환 스크립트 (전처리, 학습, tflite 변환 등)
├─ dataset/     # (로컬 비공개) 원천 데이터 또는 다운로드/가공 결과 위치
├─ image/       # 보고서용 결과 그래프/혼동행렬/아키텍처 그림 등
├─ matrix/      # 전처리 후 특성 행렬(NPY/CSV 등) 저장 위치
├─ picode/      # Raspberry Pi 4 추론 스크립트 (tflite-runtime 사용)
├─ report.md    # 실험 요약/회의록/메모
└─ README.md
```

---

## Environment

### PC (학습/변환)

* Python 3.10+
* 주요 패키지: `numpy`, `pandas`, `scikit-learn`, `tensorflow` (2.13~2.15 권장), `matplotlib`


### Raspberry Pi 4 (추론)

* Raspberry Pi 4 OS
* `tflite-runtime` (TensorFlow 대신 경량 런타임)

```bash
python3 -m pip install --upgrade tflite-runtime
```

---

## Evaluation

```bash
python code/eval.py \
  --data_dir matrix/ \
  --model_path code/ckpt/best_model.h5 \
  --out_dir image/
```

지표(예시): Accuracy, Precision/Recall/F1, ROC-AUC, Confusion Matrix

---

## Model Compression & TFLite Export

우선 **Post-Training Quantization(PTQ)** 로 용량/지연 줄이기. 필요 시 **QAT**.

```bash
# FP32 → TFLite (PTQ int8 예시)
python code/export_tflite.py \
  --model_path code/ckpt/best_model.h5 \
  --calib_dir matrix/val/ \
  --tflite_path code/ckpt/model_int8.tflite \
  --quantize int8
```

---

## Reference
추후 수정예정

- Self-Attention-Assisted_TinyML_With_Effective_Representation_for_UWB_NLOS_Identification



---

## License

연구용(WIP). 추후 명시.

---

## Data Source / Dataset 출처

본 연구에서는 **eWINE 프로젝트의 UWB LOS/NLOS 데이터셋**을 사용했습니다.

* GitHub 저장소: [ewine-project/UWB-LOS-NLOS-Data-Set](https://github.com/ewine-project/UWB-LOS-NLOS-Data-Set) ([GitHub][1])
* 이 데이터셋은 7개의 실내 환경에서 LOS / NLOS 조건 하에서 수집된 UWB CIR 데이터를 포함합니다. ([GitHub][1])
* CC-BY-4.0 라이선스 하에 제공되며, 이용 시 저작자 표기를 권장합니다. ([GitHub][1])
