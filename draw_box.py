import cv2
import numpy as np

# 类别
classes = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck',
           8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
           14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
           22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase',
           29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
           35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
           40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple',
           48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut',
           55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet',
           62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
           69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase',
           76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

# 随机颜色
color_palette = np.random.uniform(100, 255, size=(len(classes), 3))

def draw_detections(img, box, score, class_id):
    """
    在输入图像上绘制检测到的对象的边界框和标签。

    参数:
            img: 要在其上绘制检测结果的输入图像。
            box: 检测到的边界框。
            score: 对应的检测得分。
            class_id: 检测到的对象的类别ID。

    返回:
            无
    """

    # 提取边界框的坐标
    x1, y1, w, h = box
    # 根据类别ID检索颜色
    color = color_palette[class_id]
    # 在图像上绘制边界框
    cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)
    # 创建标签文本，包括类名和得分
    label = f'{classes[class_id]}: {score:.2f}'
    # 计算标签文本的尺寸
    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    # 计算标签文本的位置
    label_x = x1
    label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10
    # 绘制填充的矩形作为标签文本的背景
    cv2.rectangle(img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED)
    # 在图像上绘制标签文本
    cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

