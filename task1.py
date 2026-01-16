import cv2
import numpy as np
import os

def process_video(video_path, task_name):
    """
    处理单个视频的核心函数，并输出带有标记的视频。

    Args:
        video_path (str): 输入视频文件的路径。
        task_name (str): 任务名称（用于打印和输出区分）。
    """
    print(f"Starting to process {task_name} video: {video_path}")

    # 1. 加载输入视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open {task_name} video file: {video_path}")
        return

    # 获取视频的基本信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Input video info - FPS: {fps}, Resolution: {width}x{height}, Total Frames: {total_frames}")

    # 2. 创建视频写入器，输出检测结果
    output_filename = f"output_{task_name}_motion_detected.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # 使用 mp4v 编码器
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"Error: Cannot create output video writer for {output_filename}")
        cap.release()
        return

    # 3. 初始化前一帧（用于帧差法）
    prev_frame = None
    frame_count = 0
    motion_detection_count = 0  # 记录检测到变化的帧数

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"{task_name} video processing complete.")
            break

        frame_count += 1
        if frame_count % 30 == 0: # 每处理30帧打印一次进度
            print(f"Processing frame {frame_count}/{total_frames} for {task_name}")

        # 4. 图像预处理
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 5. 方法一：帧差法检测变化（简单高效）
        significant_change_detected = False
        if prev_frame is not None:
            # 计算当前帧与前一帧的绝对差值
            frame_delta = cv2.absdiff(prev_frame, gray_frame)
            
            # 二值化，阈值可根据需要调整
            thresh = cv2.threshold(frame_delta, 30, 255, cv2.THRESH_BINARY)[1]
            
            # 形态学操作，连接小的区域，去除噪点
            thresh = cv2.dilate(thresh, None, iterations=2)
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 6. 分析变化区域
            for contour in contours:
                # 过滤掉面积过小的轮廓（噪点）
                if cv2.contourArea(contour) < 1000:  # 阈值可调
                    continue

                # 计算轮廓的边界框
                (x, y, w, h) = cv2.boundingRect(contour)
                
                # 在原图上画出边界框
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                significant_change_detected = True

            if significant_change_detected:
                motion_detection_count += 1
                # 可选：在帧上添加文字提示
                cv2.putText(frame, f"Motion Detected!", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 7. 将处理后的帧写入输出视频
        out.write(frame)

        # 更新前一帧
        prev_frame = gray_frame

    # 8. 释放资源
    cap.release()
    out.release()
    print(f"Statistics for {task_name}: {motion_detection_count} frames with significant changes detected.")
    print(f"Output video saved as: {output_filename}\n")


def main():
    """
    主函数，只处理视频1。
    """
    # --- 在程序内直接指定 macOS 风格的视频路径 ---
    # 请将这些路径修改为您的实际视频文件路径
    
    # 示例 1: 使用相对路径 (假设视频和脚本在同一目录)
    TASK1_VIDEO_PATH = "./video1.mp4" 

    # 示例 2: 使用绝对路径 (请替换 YourName 和具体路径)
    # TASK1_VIDEO_PATH = "/Users/YourName/Documents/task1_video.mp4"
    
    print("Starting video change detection program (output as video)...")

    # 检查视频文件是否存在
    if not os.path.exists(TASK1_VIDEO_PATH):
        print(f"Error: Task1 video file does not exist: {TASK1_VIDEO_PATH}")
        print("Please ensure the file path is correct and the file exists.")
        return

    # 处理任务1视频
    process_video(TASK1_VIDEO_PATH, "Task1")

    print("Video processing complete. Output video has been generated.")


if __name__ == "__main__":
    main()