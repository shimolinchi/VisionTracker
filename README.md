# Visual Hand Tracker
**双语 README / Bilingual README**

![](demo1.gif)
![Demo](demo2.gif)


本项目基于 OpenCV + MediaPipe，实现灵巧手的视觉遥操作。使用的灵巧手为睿研智控 RY-H1(16)。

### 功能特点

使用 MediaPipe Hands 进行手部关键点检测

自动建立手掌局部坐标系

各手指关节角度实时计算

视频流中实时可视化坐标轴和关节角度

进行对指距离检测，对对指进行优化

将关节角度映射为电机控制信号，通过 PCAN 下发

内置滤波器，降低抖动，提高稳定性

### 安装

python环境 3.9
克隆并安装依赖（依赖在 requirements.txt 中）：
```
cd VisionTracker
pip install -r requirements.txt
```

注意：若使用 L515 摄像头且使用 pyrealsense 的话，需要安装 pyrealsense 2.50.0.3812 版本，需要到官网上下载其whl文件后手动安装（由于该摄像头现已停产，因此更新版本的pyrealsense不可使用）

## 使用方法

### 硬件准备

确保摄像头可用（默认 cv2.VideoCapture(0)，也可选择使用 pyrealsense 获取摄像头数据流）

将 PCAN 设备连接到机械手（确保驱动安装）

### 运行
```
python example.py
```

或者直接调用 VisualTracker 类的 track() 方法。

### 运行效果

窗口中显示手部关键点、局部坐标轴和实时关节角度

角度自动映射为电机目标位置，并通过 PCAN 下发

### 代码结构
```
├── VisualTracker.py     # 手部追踪与电机控制核心类
├── utils/
│   ├── math_tools.py    # 向量运算、角度计算
│   ├── image_tools.py   # 绘制与标注工具
├── requirements.txt     # 依赖文件
├── main.py              # 示例入口
```