import cv2
import numpy as np

def AnnotateNumber(image, point, number, color=(0, 0, 255), font_scale=0.5, thickness=1):
    """
    在图像上的指定点旁边标注数值
    
    参数:
    image: 输入图像 (numpy数组)
    point: 点的坐标 (x, y)
    number: 要标注的数值 (int/float)
    color: 文本颜色 (BGR格式)
    font_scale: 字体大小
    thickness: 文本粗细
    
    返回:
    标注后的图像
    """
    # 转换为整数坐标
    x, y = int(point[0]), int(point[1])
    
    # 设置文本位置偏移
    offset_x, offset_y = 7, -7  # 向右上方偏移
    
    # 绘制文本
    text = f"{number:.2f}" if isinstance(number, float) else str(number)
    cv2.putText(image, text, 
                (x + offset_x, y + offset_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, color, thickness)
    
    return image
def AnnotateCoordinate(image, point, coordinate, color=(255, 0, 0 ), font_scale=0.5, thickness=1):
    """
    在图像上的指定点旁边标注数值
    
    参数:
    image: 输入图像 (numpy数组)
    point: 点的坐标 (x, y)
    number: 要标注的数值 (int/float)
    color: 文本颜色 (BGR格式)
    font_scale: 字体大小
    thickness: 文本粗细
    
    返回:
    标注后的图像
    """
    # 转换为整数坐标
    x, y = int(point[0]), int(point[1])
    
    # 设置文本位置偏移
    offset_x, offset_y = 7, -7  # 向右上方偏移
    
    # 绘制文本
    text = f"{coordinate[0]:.2f},{coordinate[1]}" 
    cv2.putText(image, text, 
                (x + offset_x, y + offset_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, color, thickness)
    
    return image

def AnnotateText(image, point, text,
                       color=(255, 0, 0), font_scale=0.5, thickness=1, offset=(7, -7)):
    """在图像上标注文字"""
    x, y = int(point[0]), int(point[1])
    ox, oy = offset
    cv2.putText(image, text, (x + ox, y + oy),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    return image

