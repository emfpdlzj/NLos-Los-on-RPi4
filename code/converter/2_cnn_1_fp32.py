# convert_2_cnn_1_fp32.py
import tensorflow as tf
from tensorflow import keras

H5_PATH = "2.cnn_1.h5"  # 네가 저장한 모델 파일
OUT_PATH = "2_cnn_1_fp32.tflite"  # 출력 파일명

# 1) 모델 로드
model = keras.models.load_model(H5_PATH, compile=False)

# 2) 입력 시그니처 고정
# Embedding을 쓰므로 입력은 (batch, 100) 정수 인덱스가 맞다 (dtype=int32).
spec = tf.TensorSpec([None, 100], tf.int32)
concrete = tf.function(lambda x: model(x)).get_concrete_function(spec)

# 3) TFLite 변환 (빌트인-only, SELECT_TF_OPS 사용 금지)
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete])
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

tflite_model = converter.convert()

# 4) 저장
with open(OUT_PATH, "wb") as f:
    f.write(tflite_model)

print(f"✅ TFLite(FP32, builtins-only) 저장 완료: {OUT_PATH}")
