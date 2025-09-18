# Visual Hand Tracker for Robotic Hand Control
**双语 README / Bilingual README**

![Demo](demo.gif)

---


### Overview
This project uses **OpenCV + MediaPipe + PCAN** to perform real-time hand joint tracking and map the results and control the ruiyan-H1 hand.

### Features
- Hand landmark detection via **MediaPipe Hands**  
- Automatic construction of palm local coordinate system  
- Joint angle calculation for each finger  
- Real-time visualization of axes and joint angles in video stream  
- Mapping of joint angles to motor control signals through **PCAN bus**  
- Built-in filter to reduce jitter  

### Installation

The project need python 3.9, ruiyan-H1 hand (16dof)
Clone and install dependencies (listed in `requirements.txt`):

```bash
cd visual-hand-tracker
pip install -r requirements.txt
```

### Usage

Hardware setup

Ensure camera is available (default cv2.VideoCapture(0))

Connect PCAN device to robotic hand

### Run
```
python example.py
```

Or directly use the VisualTracker class with the track() method.

Expected results

Window displays hand landmarks, local axes, and real-time joint angles

Angles are mapped to target motor positions and sent via CAN

### Project Structure
```
├── VisualTracker.py     # Core class: hand tracking + motor control
├── utils/
│   ├── math_tools.py    # Vector operations, angle calculations
│   ├── image_tools.py   # Drawing and annotation functions
├── requirements.txt     # Dependencies
├── main.py              # Example entry point
```

本项目基于 OpenCV + MediaPipe + CAN 总线，实现手部关节角度的实时追踪，并将结果映射到 16 路电机，从而驱动仿生机械手。

### 功能特点

使用 MediaPipe Hands 进行手部关键点检测

自动建立手掌局部坐标系

各手指关节角度实时计算

视频流中实时可视化坐标轴和关节角度

将角度映射为电机控制信号，通过 PCAN 总线 下发

内置滤波器，降低抖动，提高稳定性

### 安装

克隆并安装依赖（依赖在 requirements.txt 中）：
```
cd visual-hand-tracker
pip install -r requirements.txt
```
## 使用方法

### 硬件准备

确保摄像头可用（默认 cv2.VideoCapture(0)）

将 PCAN 设备连接到机械手（确保驱动安装）

### 运行
```
python example.py
```

或者直接调用 VisualTracker 类的 track() 方法。

### 运行效果

窗口中显示手部关键点、局部坐标轴和实时关节角度

角度自动映射为电机目标位置，并通过 CAN 下发

### 代码结构
```
├── VisualTracker.py     # 手部追踪与电机控制核心类
├── utils/
│   ├── math_tools.py    # 向量运算、角度计算
│   ├── image_tools.py   # 绘制与标注工具
├── requirements.txt     # 依赖文件
├── main.py              # 示例入口
```