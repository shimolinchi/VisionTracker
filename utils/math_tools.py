import numpy as np
from scipy.linalg import svd
from mediapipe.framework.formats import landmark_pb2

def RegressionLine(points):
    """
    求空间中多个点的回归直线
    
    参数:
    points: 点集，形状为 (n, 3) 的数组
    
    返回:
    point: 直线上的一点（质心）
    direction: 直线的方向向量
    """
    # 转换为numpy数组
    points = np.array(points)
    
    # 计算质心
    centroid = np.mean(points, axis=0)
    
    # 中心化数据
    centered_points = points - centroid
    
    # 使用奇异值分解(SVD)求主方向
    # 等价于求协方差矩阵的特征向量
    U, S, Vt = svd(centered_points)
    
    # 第一个右奇异向量就是主方向（对应最大奇异值）
    direction = Vt[0]
    
    return np.array(centroid), np.array(direction)



def CalculateTheta(v1, v2):
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)

    dot = np.dot(v1, v2)
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    
    # 避免除零
    if norm_product < 1e-8:
        return 0.0
    
    cos_theta = np.clip(dot / norm_product, -1.0, 1.0)
    return np.arccos(cos_theta)   # 返回弧度 [0, π]

def CalculateDepth(bone_length, point_1, point_2):
    """
    通过手指骨长度和骨头两端点坐标计算深度大小（不包含深度方向）
    """
    x = point_1[0] - point_2[0]
    y = point_1[1] - point_2[1]
    return np.sqrt(bone_length ** 2 - x ** 2 - y ** 2)


class LowPassFilter:
    """简单低通滤波器 (一阶)"""
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        self.prev = {}

    def filter(self, angles):
        filtered = {}
        for finger, joints in angles.items():
            filtered[finger] = {}
            for joint, value in joints.items():
                key = (finger, joint)
                if key not in self.prev:
                    self.prev[key] = value
                # EMA 低通滤波
                self.prev[key] = self.alpha * value + (1 - self.alpha) * self.prev[key]
                filtered[finger][joint] = self.prev[key]
        return filtered


class KalmanFilter1D:
    """1D 卡尔曼滤波器"""
    def __init__(self, process_variance=1e-3, measurement_variance=1e-2):
        self.x = None  # 估计值
        self.P = 1.0   # 估计协方差
        self.Q = process_variance
        self.R = measurement_variance

    def update(self, z):
        if self.x is None:
            self.x = z
        # 预测
        self.P = self.P + self.Q
        # 卡尔曼增益
        K = self.P / (self.P + self.R)
        # 更新
        self.x = self.x + K * (z - self.x)
        self.P = (1 - K) * self.P
        return self.x
    
class KalmanFilterManager:
    """管理多个角度的卡尔曼滤波"""
    def __init__(self):
        self.filters = {}

    def filter(self, angles):
        filtered = {}
        for finger, joints in angles.items():
            filtered[finger] = {}
            for joint, value in joints.items():
                key = (finger, joint)
                if key not in self.filters:
                    self.filters[key] = KalmanFilter1D()
                filtered[finger][joint] = self.filters[key].update(value)
        return filtered
    

    
def angle_between(v1, v2):
    """计算两个向量的夹角，返回角度值（度）"""
    v1, v2 = np.array(v1, float), np.array(v2, float)
    dot = np.dot(v1, v2)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom < 1e-8:
        return 0.0
    cos_theta = np.clip(dot / denom, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def normalize(v):
    """向量归一化"""
    v = np.array(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v


class LowPassFilter_:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        self.prev = None

    def filter(self, current):
        """输入输出均为列表，如 [0]*16"""
        if self.prev is None:
            self.prev = current.copy()
            return current
        
        # 计算滤波后的值
        filtered = [self.alpha * x + (1 - self.alpha) * y 
                   for x, y in zip(current, self.prev)]
        
        # 更新 prev 为当前滤波后的值
        self.prev = filtered.copy()
        return filtered
    
def map_value(x, from_min, from_max, to_min, to_max):
    """
    将数值 x 从 [from_min, from_max] 区间线性映射到 [to_min, to_max] 区间。
    支持输入区间和输出区间的最小值可能大于最大值的情况。
    """
    if from_max == from_min:
        raise ValueError("源区间的最大值和最小值不能相同，否则无法映射。")

    return to_min + (x - from_min) * (to_max - to_min) / (from_max - from_min)