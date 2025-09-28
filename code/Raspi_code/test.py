import numpy as np
import tflite_runtime.interpreter as tflite
import csv

# 1. TFLite 모델 로드
interpreter = tflite.Interpreter(model_path="/home/pi/Desktop/cnn_classify_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 2. CSV 파일에서 한 줄 읽기
with open('/home/pi/Desktop/uwb_dataset_part1.csv', 'r') as f:
    reader = csv.reader(f)
    header = next(reader)  # 헤더 스킵
    row = next(reader)     # 첫 번째 샘플만 사용 (여러 줄 반복하려면 for 루프로 바꾸면 됨)

# 3. 데이터 전처리 (학습 코드 스타일 그대로)
row = [float(x) for x in row]
preamble_count = row[2]
cir_data = np.array(row[15:], dtype=np.float32) / preamble_count

# 4. 입력 형태 맞추기 (1, 1016)
input_data = np.expand_dims(cir_data, axis=0)  # shape: (1, 1016)

# 5. 추론
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# 6. 출력 결과
output_data = interpreter.get_tensor(output_details[0]['index'])
print("예측 결과 (NLOS 확률):", output_data[0][0])
