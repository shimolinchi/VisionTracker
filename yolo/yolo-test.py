# import cv2
# from ultralytics import YOLO

# # 加载 YOLO 模型（替换为你的模型路径）
# model = YOLO("yolo11n.pt")  # 或者 "yolov8n.pt"（Ultralytics 官方模型）

# # 打开摄像头（0 表示默认摄像头）
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("无法打开摄像头！")
#     exit()

# # 设置摄像头分辨率（可选）
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# while True:
#     # 读取一帧图像
#     ret, frame = cap.read()
#     if not ret:
#         print("无法读取帧！")
#         break

#     # 使用 YOLO 检测物体
#     results = model(frame, verbose=False)  # verbose=False 关闭冗余输出

#     # 遍历检测结果并绘制标注
#     for result in results:
#         # 获取检测到的物体信息
#         boxes = result.boxes  # 边界框
#         for box in boxes:
#             # 提取坐标和类别
#             x1, y1, x2, y2 = map(int, box.xyxy[0])  # 边界框坐标
#             conf = float(box.conf[0])                # 置信度
#             cls_id = int(box.cls[0])                 # 类别ID
#             cls_name = model.names[cls_id]           # 类别名称

#             # 绘制边界框和标签
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             label = f"{cls_name} {conf:.2f}"
#             cv2.putText(frame, label, (x1, y1 - 10), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     # 显示结果
#     cv2.imshow("YOLO Object Detection", frame)

#     # 按 'q' 退出
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # 释放资源
# cap.release()
# cv2.destroyAllWindows()



import cv2
import os
from ultralytics import YOLO

# 加载模型（假设 yolov8n.pt 也在 yolo 目录下）
model = YOLO("yolo/yolo11l.pt")  # 如果模型在上级目录，改为 "../yolov8n.pt"

# 图片路径（使用 os.path 兼容不同操作系统）
image_path = os.path.join("yolo", "6.png")

# 检查文件是否存在
if not os.path.exists(image_path):
    print(f"错误：文件不存在 {os.path.abspath(image_path)}")
    exit()

# 读取图片
image = cv2.imread(image_path)
if image is None:
    print(f"错误：无法读取图片（可能损坏或权限问题）")
    exit()

# 执行检测
results = model(image)

# 标注结果
annotated_image = results[0].plot()

# 保存结果（保存在 yolo 目录下）
output_path = os.path.join("yolo", "detected_6.png")
cv2.imwrite(output_path, annotated_image)

print(f"检测完成！结果已保存到:\n{os.path.abspath(output_path)}")