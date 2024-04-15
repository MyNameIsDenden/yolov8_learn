import cv2
import numpy as np
import time
from tools_detect import *

if __name__ == '__main__':
    # 模型文件的路径
    model_path = "res\yolov8n.onnx"
    # 初始化检测模型，加载模型并获取模型输入节点信息和输入图像的宽度、高度
    session, model_inputs, input_width, input_height = init_detect_model(model_path)
    # 三种模式 1为图片预测，并显示结果图片；2为摄像头检测，并实时显示FPS； 3为视频检测，并保存结果视频
    mode = 1
    if mode == 1:
        # 读取图像文件
        image_data = cv2.imread("test_data/bus.jpg")
        # 使用检测模型对读入的图像进行对象检测
        result_image = detect_object(image_data, session, model_inputs, input_width, input_height)
        # 将检测后的图像保存到文件
        cv2.imwrite("output_image.jpg", result_image)
        # 在窗口中显示检测后的图像
        cv2.imshow('Output', result_image)
        # 等待用户按键，然后关闭显示窗口
        cv2.waitKey(0)
    elif mode == 2:
        # 打开摄像头
        cap = cv2.VideoCapture(0)  # 0表示默认摄像头，如果有多个摄像头可以尝试使用1、2等
        # 检查摄像头是否成功打开
        if not cap.isOpened():
            print("Error: Could not open camera.")
            exit()
        # 初始化帧数计数器和起始时间
        frame_count = 0
        start_time = time.time()
        # 循环读取摄像头视频流
        while True:
            # 读取一帧
            ret, frame = cap.read()
            # 检查帧是否成功读取
            if not ret:
                print("Error: Could not read frame.")
                break
            # 使用检测模型对读入的帧进行对象检测
            output_image = detect_object(frame, session, model_inputs, input_width, input_height)
            # 计算帧速率
            frame_count += 1
            end_time = time.time()
            elapsed_time = end_time - start_time
            fps = frame_count / elapsed_time
            print(f"FPS: {fps:.2f}")
            # 将FPS绘制在图像上
            cv2.putText(output_image, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            # 在窗口中显示当前帧
            cv2.imshow("Video", output_image)
            # 按下 'q' 键退出循环
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # 释放摄像头资源
        cap.release()
        # 关闭窗口
        cv2.destroyAllWindows()
    elif mode == 3:
        # 输入视频路径
        input_video_path = 'D:/videos/saved/Battlefield V/nice.mp4'
        # 输出视频路径
        output_video_path = 'lv_det.mp4'
        # 打开视频文件
        cap = cv2.VideoCapture(input_video_path)
        # 检查视频是否成功打开
        if not cap.isOpened():
            print("Error: Could not open video.")
            exit()
        # 读取视频的基本信息
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = cap.get(cv2.CAP_PROP_FPS)
        # 定义视频编码器和创建VideoWriter对象
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 根据文件名后缀使用合适的编码器
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
        # 初始化帧数计数器和起始时间
        frame_count = 0
        start_time = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Info: End of video file.")
                break
            # 对读入的帧进行对象检测
            output_image = detect_object(frame, session, model_inputs, input_width, input_height)
            # 计算并打印帧速率
            frame_count += 1
            end_time = time.time()
            elapsed_time = end_time - start_time
            if elapsed_time > 0:
                fps = frame_count / elapsed_time
                print(f"FPS: {fps:.2f}")
            # 将处理后的帧写入输出视频
            out.write(output_image)
            #（可选）实时显示处理后的视频帧
            cv2.imshow("Output Video", output_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # 释放资源
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    else:
        print("输入错误，请检查mode的赋值")



