import tensorflow as tf
from tensorflow import keras

H5_PATH = "2.cnn_4.h5"
OUT_PATH = "2_cnn_4_fp32.tflite"

# 1) 모델 로드
model = keras.models.load_model(H5_PATH, compile=False)

# 2) 입력 시그니처 고정: Embedding → (None, 100) int32 인덱스
spec = tf.TensorSpec([None, 100], tf.int32)
concrete = tf.function(lambda x: model(x)).get_concrete_function(spec)

# 3) 빌트인-only 변환 (SELECT_TF_OPS 금지)
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete])
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

tflite_model = converter.convert()

# 4) 저장
with open(OUT_PATH, "wb") as f:
    f.write(tflite_model)

print(f"✅ TFLite(FP32, builtins-only) 저장 완료: {OUT_PATH}")
