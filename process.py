import numpy as np
import cv2
from iou_yolov8 import *
from draw_box import draw_detections

# 置信度
confidence_thres = 0.35
# iou阈值
iou_thres = 0.5

# 应用非最大抑制过滤重叠的边界框
def custom_NMSBoxes(boxes, scores, confidence_threshold, iou_threshold):
    # 如果没有边界框，则直接返回空列表
    if len(boxes) == 0:
        return []
    # 将得分和边界框转换为NumPy数组
    scores = np.array(scores)
    boxes = np.array(boxes)
    # 根据置信度阈值过滤边界框
    mask = scores > confidence_threshold
    filtered_boxes = boxes[mask]
    filtered_scores = scores[mask]
    # 如果过滤后没有边界框，则返回空列表
    if len(filtered_boxes) == 0:
        return []
    # 根据置信度得分对边界框进行排序
    sorted_indices = np.argsort(filtered_scores)[::-1]
    # 初始化一个空列表来存储选择的边界框索引
    indices = []
    # 当还有未处理的边界框时，循环继续
    while len(sorted_indices) > 0:
        # 选择得分最高的边界框索引
        current_index = sorted_indices[0]
        indices.append(current_index)
        # 如果只剩一个边界框，则结束循环
        if len(sorted_indices) == 1:
            break
        # 获取当前边界框和其他边界框
        current_box = filtered_boxes[current_index]
        other_boxes = filtered_boxes[sorted_indices[1:]]
        # 计算当前边界框与其他边界框的IoU
        iou = calculate_iou(current_box, other_boxes)
        # 找到IoU低于阈值的边界框，即与当前边界框不重叠的边界框
        non_overlapping_indices = np.where(iou <= iou_threshold)[0]
        # 更新sorted_indices以仅包含不重叠的边界框
        sorted_indices = sorted_indices[non_overlapping_indices + 1]
    # 返回选择的边界框索引
    return indices



def preprocess(img, input_width, input_height):
    """
    在执行推理之前预处理输入图像。

    返回:
        image_data: 为推理准备好的预处理后的图像数据。
    """

    # 获取输入图像的高度和宽度
    img_height, img_width = img.shape[:2]
    # 将图像颜色空间从BGR转换为RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    # 将图像大小调整为匹配输入形状
    img = cv2.resize(img, (input_width, input_height))
    # 通过除以255.0来归一化图像数据
    image_data = np.array(img) / 255.0
    # 转置图像，使通道维度为第一维
    image_data = np.transpose(image_data, (2, 0, 1))  # 通道首
    # 扩展图像数据的维度以匹配预期的输入形状
    image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
    # 返回预处理后的图像数据
    return image_data, img_height, img_width

def postprocess(input_image, output, input_width, input_height, img_width, img_height):
    """
    对模型输出进行后处理，提取边界框、得分和类别ID。

    参数:
        input_image (numpy.ndarray): 输入图像。
        output (numpy.ndarray): 模型的输出。
        input_width (int): 模型输入宽度。
        input_height (int): 模型输入高度。
        img_width (int): 原始图像宽度。
        img_height (int): 原始图像高度。

    返回:
        numpy.ndarray: 绘制了检测结果的输入图像。
    """

    # 转置和压缩输出以匹配预期的形状
    outputs = np.transpose(np.squeeze(output[0]))
    # 获取输出数组的行数
    rows = outputs.shape[0]
    # 用于存储检测的边界框、得分和类别ID的列表
    boxes = []
    scores = []
    class_ids = []
    # 计算边界框坐标的缩放因子
    x_factor = img_width / input_width
    y_factor = img_height / input_height
    # 遍历输出数组的每一行
    for i in range(rows):
        # 从当前行提取类别得分
        classes_scores = outputs[i][4:]
        # 找到类别得分中的最大得分
        max_score = np.amax(classes_scores)
        # 如果最大得分高于置信度阈值
        if max_score >= confidence_thres:
            # 获取得分最高的类别ID
            class_id = np.argmax(classes_scores)
            # 从当前行提取边界框坐标  x,y是中心点
            x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
            # 计算边界框的缩放坐标
            left = int((x - w / 2) * x_factor)
            top = int((y - h / 2) * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)
            # 将类别ID、得分和框坐标添加到各自的列表中
            class_ids.append(class_id)
            scores.append(max_score)
            boxes.append([left, top, width, height])
    # 应用非最大抑制过滤重叠的边界框
    indices = custom_NMSBoxes(boxes, scores, confidence_thres, iou_thres)
    # 遍历非最大抑制后的选定索引
    for i in indices:
        # 根据索引获取框、得分和类别ID
        box = boxes[i]
        score = scores[i]
        class_id = class_ids[i]
        # 在输入图像上绘制检测结果
        draw_detections(input_image, box, score, class_id)
    # 返回修改后的输入图像
    return input_image
