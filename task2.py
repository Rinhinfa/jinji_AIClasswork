import cv2
import numpy as np
from ultralytics import YOLO

# ================== 参数设置 ==================
VIDEO_PATH = "video2.mp4"
OUTPUT_PATH = "output_abnormal.mp4"

HEIGHT_RATIO_THRESHOLD = 1.45
BOTTOM_SHIFT_THRESHOLD = 30
STAND_FRAME_THRESHOLD = 20

MOVE_DISTANCE_THRESHOLD = 120
MOVE_FRAME_THRESHOLD = 25

IOU_THRESHOLD = 0.3
CENTER_DISTANCE_THRESHOLD = 80

# ================== 加载模型 ==================
model = YOLO("yolov8n.pt")

# ================== 工具函数 ==================
def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

# ================== 打开视频 ==================
cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# ⭐⭐⭐ 新增：视频写入器 ⭐⭐⭐
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

# ================== 状态存储 ==================
person_states = {}
person_id_counter = 0

# ================== 主循环 ==================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.4, classes=[0])
    boxes = results[0].boxes.xyxy.cpu().numpy()

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        h = y2 - y1
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        matched_id = None
        for pid, state in person_states.items():
            iou_val = iou(state["box"], box)
            center_dist = np.linalg.norm(
                np.array((cx, cy)) - np.array(state["last_center"])
            )
            if iou_val > IOU_THRESHOLD or center_dist < CENTER_DISTANCE_THRESHOLD:
                matched_id = pid
                break

        if matched_id is None:
            person_id_counter += 1
            person_states[person_id_counter] = {
                "box": box,
                "init_height": h,
                "init_bottom": y2,
                "last_center": (cx, cy),
                "stand_count": 0,
                "move_count": 0,
                "move_distance": 0
            }
            matched_id = person_id_counter

        state = person_states[matched_id]

        # ===== 起身检测 =====
        height_ratio = h / state["init_height"]
        bottom_shift = state["init_bottom"] - y2

        if height_ratio > HEIGHT_RATIO_THRESHOLD and bottom_shift > BOTTOM_SHIFT_THRESHOLD:
            state["stand_count"] += 1
        else:
            state["stand_count"] = 0

        # ===== 移动检测 =====
        dist = np.linalg.norm(
            np.array((cx, cy)) - np.array(state["last_center"])
        )
        state["move_distance"] += dist

        if state["move_distance"] > MOVE_DISTANCE_THRESHOLD:
            state["move_count"] += 1
        else:
            state["move_count"] = 0

        state["last_center"] = (cx, cy)
        state["box"] = box

        # ===== 显示 =====
        label = f"ID {matched_id}"
        color = (0, 255, 0)

        if state["stand_count"] > STAND_FRAME_THRESHOLD:
            label += " Standing"
            color = (0, 0, 255)

        if state["move_count"] > MOVE_FRAME_THRESHOLD:
            label += " Moving"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # ⭐⭐⭐ 写入视频 ⭐⭐⭐
    writer.write(frame)

    # 可选：实时显示
    cv2.imshow("Abnormal Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# ================== 释放资源 ==================
cap.release()
writer.release()
cv2.destroyAllWindows()

print("处理完成，结果已保存为：", OUTPUT_PATH)
