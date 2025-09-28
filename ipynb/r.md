# 파이썬 ->주피터 코드 구분기준 

1. 환경/라이브러리 세팅 : import, 경로설정, 랜덤시드 고정

2. 데이터 로드 & 전처리 : 
- 데이터 셋 불러오기
- train/test split, scaling, feature selection

3. 모델정의
- ex) 모델 클래스: keras sequential,...
- optimizer, loss function

4. 학습과정
- model.fit, epoch
- 로그/콜백

5. 평가 및 시각화
- model.evaluate classification_report, confusion_matrix
- matplotlib, seaborn 시각화

6.추론 , 저장
- model.predict로 샘플입력
- 모델 저장, 불러오기 : .save, torch.save