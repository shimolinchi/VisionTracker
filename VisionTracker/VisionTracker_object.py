import cv2
import mediapipe as mp
import numpy as np
# import pyrealsense2 as rs
from ryhand_win_python_pcan.PCAN_RY16lib import *
from utils.math_tools import *
from utils.image_tools import *


class VisualTracker:

    pos_cal = [0] * 16          # 计算出来的值（参考值0~1，但也可能大于1或小于0）
    pos_now = [0] * 16          # 每个电机的当前位置    
    pos_drive = [0] * 16        # 每个电机的实际驱动位置
    current = [110] * 16         # 每个电机的电流限制
    velocity = [10] * 16

    HAND_LANDMARKS = {
    "WRIST": 0,
    "THUMB_CMC": 1, "THUMB_MCP": 2, "THUMB_IP": 3, "THUMB_TIP": 4,
    "INDEX_MCP": 5, "INDEX_PIP": 6, "INDEX_DIP": 7, "INDEX_TIP": 8,
    "MIDDLE_MCP": 9, "MIDDLE_PIP": 10, "MIDDLE_DIP": 11, "MIDDLE_TIP": 12,
    "RING_MCP": 13, "RING_PIP": 14, "RING_DIP": 15, "RING_TIP": 16,
    "PINKY_MCP": 17, "PINKY_PIP": 18, "PINKY_DIP": 19, "PINKY_TIP": 20,
    }
    def __init__(self):
        # # RealSense 摄像头初始化
        # pipeline = rs.pipeline()
        # config = rs.config()
        # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        # pipeline.start(config)

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # Mediapipe 初始化
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                            min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils

        # 低通滤波器，平滑角度数据
        self.angle_filter = LowPassFilter(alpha=0.5)
        self.pos_filter = LowPassFilter_(alpha=0.5)


    def calc_hand_axes(self, pts):
        """
        基于 MCP 关节点拟合手掌坐标系
        X 轴: 通过右手定则确定
        Y 轴: MCP 排列方向
        Z 轴: 垂直手掌，指向掌心外侧
        """
        wrist = pts[self.HAND_LANDMARKS["WRIST"]]

        # 四个 MCP 点用于回归直线
        mcp_points = np.array([
            pts[self.HAND_LANDMARKS["INDEX_MCP"]],
            pts[self.HAND_LANDMARKS["MIDDLE_MCP"]],
            pts[self.HAND_LANDMARKS["RING_MCP"]],
            pts[self.HAND_LANDMARKS["PINKY_MCP"]],
        ])

        # 用 SVD 求出 MCP 基准方向
        centroid = np.mean(mcp_points, axis=0)
        _, _, vv = np.linalg.svd(mcp_points - centroid)
        direction = vv[0]  # 主方向

        # 定义坐标系
        z_dir = np.cross(direction, centroid - wrist)
        y_dir = -direction.copy()
        x_dir = np.cross(y_dir, z_dir)

        return {
            "wrist": wrist,
            "x": normalize(x_dir),
            "y": normalize(y_dir),
            "z": normalize(z_dir),
        }

    def calc_hand_angles(self, pts, axes, zero_offsets=None):
        if zero_offsets is None:
            zero_offsets = {}

        try:
            wrist = pts[self.HAND_LANDMARKS["WRIST"]]
            x_axis, y_axis, z_axis = axes["x"], axes["y"], axes["z"]

            # 每根手指关节点索引
            lm = self.HAND_LANDMARKS
            finger_joints = {
                "thumb":  [lm["THUMB_CMC"],  lm["THUMB_MCP"],  lm["THUMB_IP"],   lm["THUMB_TIP"]],
                "index":  [lm["INDEX_MCP"],  lm["INDEX_PIP"],  lm["INDEX_DIP"],  lm["INDEX_TIP"]],
                "middle": [lm["MIDDLE_MCP"], lm["MIDDLE_PIP"], lm["MIDDLE_DIP"], lm["MIDDLE_TIP"]],
                "ring":   [lm["RING_MCP"],   lm["RING_PIP"],   lm["RING_DIP"],   lm["RING_TIP"]],
                "pinky":  [lm["PINKY_MCP"],  lm["PINKY_PIP"],  lm["PINKY_DIP"],  lm["PINKY_TIP"]],
            }

            angles = {finger: {} for finger in finger_joints.keys()}

            for finger, ids in finger_joints.items():
                if finger == "thumb":
                    cmc, mcp, ip, tip = ids

                    # CMC 弯曲（不使用，计算不准，采用MCP关节的侧摆）
                    cmc_flex = angle_between(wrist - pts[cmc], pts[mcp] - pts[cmc])
                    angles[finger]["CMC_flex"] = cmc_flex

                    # MCP 弯曲  不用角度计算方式了，算的不准，用坐标分量方式
                    mcp_flex = np.dot(pts[ip] - pts[mcp], x_axis) * 100
                    angles[finger]["MCP_flex"] = mcp_flex

                    # MCP 侧摆  用和y轴夹角方式
                    mcp_abduction = angle_between(y_axis, pts[ip] - pts[cmc]) - mcp_flex * 1.6
                    mcp_abduction -= zero_offsets.get(finger, 0.0)
                    angles[finger]["MCP_abduction"] = mcp_abduction

                    # IP 弯曲
                    ip_flex = angle_between(pts[mcp] - pts[ip], pts[tip] - pts[ip])
                    angles[finger]["IP"] = ip_flex

                else: 
                    mcp, pip, dip, tip = ids

                    # MCP 弯曲
                    mcp_flex = angle_between(wrist - pts[mcp], pts[pip] - pts[mcp])
                    angles[finger]["MCP_flex"] = mcp_flex

                    # MCP 侧摆
                    finger_vec = normalize(pts[pip] - pts[mcp])
                    proj_vec = normalize(finger_vec - np.dot(finger_vec, z_axis) * z_axis)
                    mcp_abduction = angle_between(proj_vec, y_axis)
                    if np.dot(proj_vec, x_axis) < 0:
                        mcp_abduction *= -1
                    mcp_abduction -= zero_offsets.get(finger, 0.0)
                    angles[finger]["MCP_abduction"] = mcp_abduction

                    # PIP 弯曲
                    pip_flex = angle_between(pts[mcp] - pts[pip], pts[dip] - pts[pip])
                    angles[finger]["PIP"] = pip_flex

                    # DIP 弯曲
                    dip_flex = angle_between(pts[pip] - pts[dip], pts[tip] - pts[dip])
                    angles[finger]["DIP"] = dip_flex
        except Exception as e:
            print(e)
        return angles, wrist

    def draw_joint_angles(self, image, landmarks, angles, width, height):
        """在图像上绘制关节角度"""
        lm = self.HAND_LANDMARKS
        angle_to_point = {
            "thumb":  {"CMC_flex": lm["THUMB_CMC"],"MCP_flex": lm["THUMB_MCP"], "MCP_abduction": lm["THUMB_MCP"], "IP": lm["THUMB_IP"], "DIP": lm["THUMB_TIP"]},
            "index":  {"MCP_flex": lm["INDEX_MCP"], "MCP_abduction": lm["INDEX_MCP"], "PIP": lm["INDEX_PIP"], "DIP": lm["INDEX_DIP"]},
            "middle": {"MCP_flex": lm["MIDDLE_MCP"], "MCP_abduction": lm["MIDDLE_MCP"], "PIP": lm["MIDDLE_PIP"], "DIP": lm["MIDDLE_DIP"]},
            "ring":   {"MCP_flex": lm["RING_MCP"], "MCP_abduction": lm["RING_MCP"], "PIP": lm["RING_PIP"], "DIP": lm["RING_DIP"]},
            "pinky":  {"MCP_flex": lm["PINKY_MCP"], "MCP_abduction": lm["PINKY_MCP"], "PIP": lm["PINKY_PIP"], "DIP": lm["PINKY_DIP"]},
        }

        # 每个关节的文字偏移 & 颜色
        offset_rules = {"CMC_flex": (7, -18),"MCP_flex": (7, -12), "MCP_abduction": (7, 12), "PIP": (7, -7), "DIP": (7, -7),"IP": (7, -7)}
        color_rules = {"CMC_flex": (255, 128, 0), "MCP_flex": (0,128,255), "MCP_abduction": (0,128,255),
                    "PIP": (0,255,0), "DIP": (255,0,0), "IP": (0,255,0)}

        for finger, joint_angles in angles.items():
            for joint, value in joint_angles.items():
                if joint not in angle_to_point[finger]:
                    continue  
                idx = angle_to_point[finger][joint]
                u, v = int(landmarks[idx][0] * width), int(landmarks[idx][1] * height)
                AnnotateText(image, (u, v), f"{value:.1f}",
                                color=color_rules[joint], offset=offset_rules[joint])
        return image

    def draw_axis(self, image, wrist, x_axis, y_axis, z_axis, length_ratio=0.12):
        """在图像上绘制手掌局部坐标系"""
        h, w = image.shape[:2]
        origin_px = (int(w * wrist[0]), int(h * wrist[1]))
        length_px = int(min(w, h) * length_ratio)

        # 计算坐标轴终点
        x_end = (int(origin_px[0] + x_axis[0] * length_px), int(origin_px[1] + x_axis[1] * length_px))
        y_end = (int(origin_px[0] + y_axis[0] * length_px), int(origin_px[1] + y_axis[1] * length_px))
        z_end = (int(origin_px[0] + z_axis[0] * length_px), int(origin_px[1] + z_axis[1] * length_px))

        # 绘制坐标轴
        cv2.arrowedLine(image, origin_px, x_end, (0, 0, 255), 2, tipLength=0.2)   # X 红
        cv2.arrowedLine(image, origin_px, y_end, (0, 255, 0), 2, tipLength=0.2)  # Y 绿
        cv2.arrowedLine(image, origin_px, z_end, (255, 0, 0), 2, tipLength=0.2)  # Z 蓝

        # 标注文字
        cv2.putText(image, "X", (x_end[0] + 4, x_end[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(image, "Y", (y_end[0] + 4, y_end[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(image, "Z", (z_end[0] + 4, z_end[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        return image


    def send_predefined_action(self, vel_array, pos_array, current_array):
        """发送一组电机目标动作到 CAN 总线"""
        for i in range(16):
            motor_id = i + 1
            speed = vel_array[i]
            position = int(pos_array[i])
            current = current_array[i]

            # CAN 数据帧
            data = [
                0xAA,
                position & 0xFF, (position >> 8) & 0xFF,
                speed & 0xFF, (speed >> 8) & 0xFF,
                current & 0xFF, (current >> 8) & 0xFF
            ]

            send_can_message(motor_id, data)


    def get_motor_positions(self):
        """
        连续读取 16 个电机的反馈，并提取位置
        返回: pos_now[16] 列表 (每个元素是对应电机的 P 值, 范围 0~4095)
        """

        for _ in range(16):
            can_id, buf = receive_can_message()
            if buf is None or len(buf) < 3:
                continue  # 没读到数据，或者数据太短，跳过
            state = parse_finger_feedback(buf)
            # === 确定电机编号 ===
            motor_index = (can_id - 257) % 16  # ID=257~272 对应 16 个电机
            pos = state["P"]
            self.pos_now[motor_index] = pos


    def set_motor_velocities(self):
        for i in range(16):
            error = abs(self.pos_now[i] - self.pos_drive[i])
            self.velocity[i] = int(error * 1.2 + 50)
            self.velocity[i] = max(min(self.velocity[i], 1900), 0)

    def track(self):
        try:
            while True:
                # 获取相机帧
                # frames = pipeline.wait_for_frames()
                # color_frame = frames.get_color_frame()
                # if not color_frame:
                #     continue
                # # 转为 OpenCV 格式并镜像
                # image = np.asanyarray(color_frame.get_data())
                # image = cv2.flip(image, 1)
                # h, w = image.shape[:2]

                ret, image = self.cap.read()
                if not ret:
                    continue
                # 镜像图像
                image = cv2.flip(image, 1)
                h, w = image.shape[:2]

                # Mediapipe 关键点检测
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb)

                if results.multi_hand_landmarks:
                    
                    # 取第一个手的关键点
                    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.multi_hand_landmarks[0].landmark])

                    # 手坐标系
                    axes = self.calc_hand_axes(landmarks)

                    # 关节角度
                    angles, wrist = self.calc_hand_angles(landmarks, axes)
                    # angles = angle_filter.filter(angles)
                    j = []
                    j.append(map_value(angles["thumb"]["MCP_flex"],         3,  10,  0, 1))   # 0
                    j.append(map_value(angles["thumb"]["MCP_abduction"],   24,  72,  0, 1))   # 1
                    j.append(map_value(angles["thumb"]["IP"],             165, 145,  0, 1))   # 2

                    j.append(map_value(angles["index"]["MCP_flex"],       167, 140,  0, 1))   # 3
                    j.append(map_value(angles["index"]["MCP_abduction"],  45 , 75 , -1, 1))   # 4
                    j.append(map_value(angles["index"]["PIP"],            175, 135,  0, 1))   # 5
                    j.append(map_value(angles["index"]["DIP"],            175,  100,  0, 1))  # 6

                    j.append(map_value(angles["middle"]["MCP_flex"],      175, 140,  0, 1))   # 7
                    j.append(map_value(angles["middle"]["MCP_abduction"],  80,  56, -1, 1))   # 8
                    j.append(map_value(angles["middle"]["PIP"],           175, 120,  0, 1))   # 9
                    j.append(map_value(angles["middle"]["DIP"],           176,  100, 0, 1))   # 10

                    j.append(map_value(angles["ring"]["MCP_flex"],        173, 140,  0, 1))   # 11
                    j.append(map_value(angles["ring"]["MCP_abduction"],    86,  65, -1, 1))   # 12
                    j.append(map_value(angles["ring"]["PIP"],             174, 132,  0, 1))   # 13
                    j.append(map_value(angles["ring"]["DIP"],             175, 100,  0, 1))   # 14

                    j.append(map_value(angles["pinky"]["MCP_flex"],       167, 130,  0, 1))   # 15
                    j.append(map_value(angles["pinky"]["MCP_abduction"],  105,  67, -1, 1))   # 16
                    j.append(map_value(angles["pinky"]["PIP"],            172, 125,  0, 1))   # 17
                    j.append(map_value(angles["pinky"]["DIP"],            173, 100,  0, 1))   # 18

                    if angles["index"]["MCP_abduction"] < 0: j[4] = 0  #处理当握拳且角度朝下的时候，会出现MCP侧摆值为负的情况，此时握拳因此全部为最大值
                    if angles["middle"]["MCP_abduction"] < 0:j[8] = 0  #处理当握拳且角度朝下的时候，会出现MCP侧摆值为负的情况，此时握拳因此全部为最大值
                    if angles["ring"]["MCP_abduction"] < 0:  j[12] = 0 #处理当握拳且角度朝下的时候，会出现MCP侧摆值为负的情况，此时握拳因此全部为最大值
                    if angles["pinky"]["MCP_abduction"] < 0: j[16] = 0 #处理当握拳且角度朝下的时候，会出现MCP侧摆值为负的情况，此时握拳因此全部为最大值

                    self.pos_cal[2] = 0.8 * j[2] + 0.2 * j[0]    # 大拇指IP关节，等于IP 与 MCP_flex加权平均
                    # print(self.pos_cal[2])
                    self.pos_cal[5] = 0.8 * j[5] + 0.2 * j[6]    # 食指PIP关节，等于PIP 与 DIP加权平均
                    self.pos_cal[8] = 0.8 * j[9] + 0.2 * j[10]   # 中指PIP关节，等于PIP 与 DIP加权平均
                    self.pos_cal[11] = 0.8 * j[13] + 0.2 * j[14] # 无名指PIP关节，等于PIP 与 DIP加权平均
                    self.pos_cal[14] = 0.8 * j[17] + 0.2 * j[18] # 小拇指PIP关节，等于PIP 与 DIP加权平均

                    # 大拇指
                    j[0] -= j[1] * 0.36
                    self.pos_cal[1]  =  j[0] + j[1] * 0.2  # 大拇指 MCP 关节 大拇指内侧电机
                    self.pos_cal[0]  =  j[0]   # 大拇指 MCP 关节 大拇指外侧电机
                    self.pos_cal[15]  = j[1]   # 手腕关节 

                    # 食指
                    j[3] -= ((abs(j[4]) + 0.1) * 0.4 - 0.25)  # 使用MCP侧摆角度来纠正MCP弯曲的角度，
                    j[4] -= 0.2                               # 侧摆角度偏移
                    if j[3] > 0.6:j[4] = 0.0                  # 当食指MCP弯曲大于0.6时，侧摆角度归零（弯曲到极限姿势，防止手指姿势过于怪异）
                    # print(f'{pos_cal[5]:.1f}\t{pos_cal[9]:.1f}\t{pos_cal[12]:.1f}\t{pos_cal[15]:.1f}')
                    self.pos_cal[3]  = 0.7 * j[3] + 0.3 * j[4] if j[4] > 0 else j[3]  # 食指 MCP 关节 远大拇指电机
                    self.pos_cal[4]  = 0.6 * j[3] - 0.4 * j[4] if j[4] < 0 else j[3]  # 食指 MCP 关节 近大拇指电机


                    # 中指
                    j[7] -= (abs(j[8]) * 0.25)      # 使用MCP侧摆角度来纠正MCP弯曲的角度，
                    if j[7] > 0.4: j[8] = 0.0       # 当食指MCP弯曲大于0.4时，侧摆角度归零（弯曲到极限姿势，防止手指姿势过于怪异）
                    self.pos_cal[6]  = 0.65 * j[7] - 0.35 * j[8] if j[8] < 0 else j[7]  # 中指 MCP 关节 远大拇指电机
                    self.pos_cal[7]  = 0.6 * j[7] + 0.4 * j[8] if j[8] > 0 else j[7]  # 中指 MCP 关节 近大拇指电机


                    # 无名指
                    j[11] -= (abs(j[8]) * 0.25)
                    if j[11] > 0.4: j[12] = 0.0
                    self.pos_cal[9]  = 0.75 * j[11] - 0.25 * j[12] if j[12] < 0 else j[11]  # 无名指 MCP 关节 远大拇指电机
                    self.pos_cal[10] = 0.75 * j[11] + 0.25 * j[12] if j[12] > 0 else j[11]  # 无名指 MCP 关节 近大拇指电机
                    

                    # 小拇指
                    if j[15] > 0.6:j[16] = 0.0
                    j[15] -= (abs(j[16]) * 0.65 - 0.22)
                    self.pos_cal[12] = 0.5 * j[15] - 0.5 * j[16] if j[16] < 0 else j[15]  # 小拇指 MCP 关节 远大拇指电机
                    self.pos_cal[13] = 0.5 * j[15] + 0.5 * j[16] if j[16] > 0 else j[15]  # 小拇指 MCP 关节 近大拇指电机
                    if j[15] >= 1:self.pos_cal[12] = self.pos_cal[13] = 1

                    self.pos_cal = [max(0.0, min(1.0, x)) for x in self.pos_cal]  # 限幅到 [0, 1]
                    

                    self.pos_drive = self.pos_filter.filter(self.pos_cal)
                    self.pos_drive = list(map(lambda x: int(x * 4095), self.pos_drive))

                    self.send_predefined_action(self.velocity, self.pos_drive, self.current)
                    self.get_motor_positions()  # 读取电机反馈
                    print(self.pos_drive[2], self.pos_now[2])
                    self.set_motor_velocities()
                    # print(f'{pos_now[0]}\t{pos_drive[0]}\t{pos_now[1]}\t{pos_drive[1]}')
                    # print(pos_now)


                    # 绘制手部骨架和角度
                    self.mp_draw.draw_landmarks(image, results.multi_hand_landmarks[0], self.mp_hands.HAND_CONNECTIONS)
                    self.draw_joint_angles(image, landmarks, angles, w, h)
                    self.draw_axis(image, wrist, axes["x"], axes["y"], axes["z"], 0.12)

                    # cv2.waitKey(1)  # 最小延迟


                # 显示结果
                cv2.imshow("image", image)
                key = cv2.waitKey(1)
                if key == 27:  # ESC 退出
                    break

        finally:
            cv2.destroyAllWindows()
            # pipeline.stop()

if __name__ == "__main__":
    try:
        mytracker = VisualTracker()
        mytracker.track()
    except Exception as e:
        print(e)
