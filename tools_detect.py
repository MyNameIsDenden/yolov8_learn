import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from process import *



# 判断是使用GPU或CPU
providers = [
    ('CUDAExecutionProvider', {
        'device_id': 0,  # 可以选择GPU设备ID，如果你有多个GPU
    }),
    'CPUExecutionProvider',  # 也可以设置CPU作为备选
]

def init_detect_model(model_path):
    # 使用ONNX模型文件创建一个推理会话，并指定执行提供者
    session = ort.InferenceSession(model_path, providers=providers)
    # 获取模型的输入信息
    model_inputs = session.get_inputs()
    # 获取输入的形状，用于后续使用
    input_shape = model_inputs[0].shape
    # 从输入形状中提取输入宽度
    input_width = input_shape[2]
    # 从输入形状中提取输入高度
    input_height = input_shape[3]
    # 返回会话、模型输入信息、输入宽度和输入高度
    return session, model_inputs, input_width, input_height

def detect_object(image, session, model_inputs, input_width, input_height):
    # 如果输入的图像是PIL图像对象，将其转换为NumPy数组
    if isinstance(image, Image.Image):
        result_image = np.array(image)
    else:
        # 否则，直接使用输入的图像（假定已经是NumPy数组）
        result_image = image
    # 预处理图像数据，调整图像大小并可能进行归一化等操作
    img_data, img_height, img_width = preprocess(result_image, input_width, input_height)
    # 使用预处理后的图像数据进行推理
    outputs = session.run(None, {model_inputs[0].name: img_data})
    # 对推理结果进行后处理，例如解码检测框，过滤低置信度的检测等
    output_image = postprocess(result_image, outputs, input_width, input_height, img_width, img_height)
    # 返回处理后的图像
    return output_image