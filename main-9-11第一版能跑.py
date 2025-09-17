import cv2
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs
from ryhand_win_python_pcan.PCAN_RY16lib import *
from utils.math_tools import *
from utils.image_tools import *

# Mediapipe 手部关键点索引
HAND_LANDMARKS = {
    "WRIST": 0,
    "THUMB_CMC": 1, "THUMB_MCP": 2, "THUMB_IP": 3, "THUMB_TIP": 4,
    "INDEX_MCP": 5, "INDEX_PIP": 6, "INDEX_DIP": 7, "INDEX_TIP": 8,
    "MIDDLE_MCP": 9, "MIDDLE_PIP": 10, "MIDDLE_DIP": 11, "MIDDLE_TIP": 12,
    "RING_MCP": 13, "RING_PIP": 14, "RING_DIP": 15, "RING_TIP": 16,
    "PINKY_MCP": 17, "PINKY_PIP": 18, "PINKY_DIP": 19, "PINKY_TIP": 20,
}

# 电机控制参数
velocity = [1300] * 16        # 每个电机的速度
velocity[15] = 800
velocity[0] = 1100
velocity[1] = 1100
pos = [0] * 16           # 每个电机的目标位置
pos_now = [0] * 16
pos_drive = [0] * 16
current = [30] * 16     # 每个电机的电流限制

# MCP 侧摆角度调零偏置（单位：度）
zero_offsets = {
    "thumb": 28,
    "index": 60,
    "middle": 70,
    "ring": 75,
    "pinky": 84
}

def calc_hand_axes(pts):
    """
    基于 MCP 关节点拟合手掌坐标系
    X 轴: 通过右手定则确定
    Y 轴: MCP 排列方向
    Z 轴: 垂直手掌，指向掌心外侧
    """
    wrist = pts[HAND_LANDMARKS["WRIST"]]

    # 四个 MCP 点用于回归直线
    mcp_points = np.array([
        pts[HAND_LANDMARKS["INDEX_MCP"]],
        pts[HAND_LANDMARKS["MIDDLE_MCP"]],
        pts[HAND_LANDMARKS["RING_MCP"]],
        pts[HAND_LANDMARKS["PINKY_MCP"]],
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

def calc_hand_angles(pts, axes, zero_offsets=None):
    if zero_offsets is None:
        zero_offsets = {}

    try:
        wrist = pts[HAND_LANDMARKS["WRIST"]]
        x_axis, y_axis, z_axis = axes["x"], axes["y"], axes["z"]

        # 每根手指关节点索引
        lm = HAND_LANDMARKS
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

def draw_joint_angles(image, landmarks, angles, width, height):
    """在图像上绘制关节角度"""
    lm = HAND_LANDMARKS
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

def draw_axis(image, wrist, x_axis, y_axis, z_axis, length_ratio=0.12):
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


def send_predefined_action(vel_array, pos_array, current_array):
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
        cv2.waitKey(1)  # 最小延迟


def main():
    # # RealSense 摄像头初始化
    # pipeline = rs.pipeline()
    # config = rs.config()
    # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # pipeline.start(config)


    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Mediapipe 初始化
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                           min_detection_confidence=0.7, min_tracking_confidence=0.7)
    mp_draw = mp.solutions.drawing_utils

    # 低通滤波器，平滑角度数据
    angle_filter = LowPassFilter(alpha=0.35)


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

            ret, image = cap.read()
            if not ret:
                continue
            # 镜像图像
            image = cv2.flip(image, 1)
            h, w = image.shape[:2]

            # Mediapipe 关键点检测
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                
                # 电机映射
                try:
                    # 取第一个手的关键点
                    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.multi_hand_landmarks[0].landmark])

                    # 手坐标系
                    axes = calc_hand_axes(landmarks)

                    # 关节角度
                    angles, wrist = calc_hand_angles(landmarks, axes, zero_offsets)
                    angles = angle_filter.filter(angles)

                    dead_zone = 1.2
                    for finger, joints in angles.items():
                        if "MCP_abduction" in joints and abs(joints["MCP_abduction"]) < dead_zone and finger != "thumb":
                            joints["MCP_abduction"] = 0.0
                        if "MCP_flex" in joints and abs(joints["MCP_flex"]) < 130 and finger != "thumb":
                            joints["MCP_abduction"] = 0.0
                    # 大拇指
                    
                    pos[2]  = np.clip(int(100 *((165-angles["thumb"]["IP"]) + angles["thumb"]["MCP_flex"] * 2)), 0, 4095)
                    if angles["index"]["MCP_abduction"] > 0:
                        pos[0]  = np.clip(int(300*(angles["thumb"]["MCP_flex"])), 0, 4095)
                        pos[1]  = np.clip(int(300*(angles["thumb"]["MCP_flex"])), 0, 4095)
                    else:
                        pos[0]  = np.clip(int(300*(angles["thumb"]["MCP_flex"])), 0, 4095)
                        pos[1]  = np.clip(int(300*(angles["thumb"]["MCP_flex"])), 0, 4095)
                    pos[15] = np.clip(int(91*(angles["thumb"]["MCP_abduction"] + 10)), 0, 4095)


                    # 食指
                    angles["index"]["MCP_flex"] += abs(angles["index"]["MCP_abduction"]) / 2.4
                    if angles["index"]["MCP_abduction"] > 0:
                        pos[3]  = np.clip(int(150*(169-angles["index"]["MCP_flex"]) + 150 * angles["index"]["MCP_abduction"] ** 2 / 36), 0, 4095)
                        pos[4]  = np.clip(int(150*(169-angles["index"]["MCP_flex"])), 0, 4095)
                    else:
                        pos[3]  = np.clip(int(150*(169-angles["index"]["MCP_flex"])), 0, 4095)
                        pos[4]  = np.clip(int(150*(169-angles["index"]["MCP_flex"]) + 150 * angles["index"]["MCP_abduction"] ** 2 / 22), 0, 4095)

                    pos[5]  = np.clip(int(37.1 *((177-angles["index"]["PIP"])   + (177 - angles["index"]["DIP"]))), 0, 4095)
                    
                    # 中指
                    if angles["middle"]["MCP_abduction"] > 0:angles["middle"]["MCP_flex"] += angles["middle"]["MCP_abduction"] / 1.2     # 实测侧摆比较大时，弯曲角度会有些变化，因此在此补偿
                    pos[6]  = np.clip(int(150 * (173 - angles["middle"]["MCP_flex"]) + 62.5 * angles["middle"]["MCP_abduction"]), 0, 4095) # 分别代表弯曲权重、弯曲识别最大角、侧摆权重
                    pos[7]  = np.clip(int(150 * (173 - angles["middle"]["MCP_flex"]) - 62.5 * angles["middle"]["MCP_abduction"]), 0, 4095) # 分别代表弯曲权重、弯曲识别最大角、侧摆权重
                    pos[8]  = np.clip(int(37.2 *((177 - angles["middle"]["PIP"])  + (177 - angles["middle"]["DIP"]))), 0, 4095)            # 分别代表缩放比例、PIP识别最大角、DIP识别最大角（计算时将DIP与PIP取平均值了）

                    # 无名指
                    if angles["ring"]["MCP_abduction"] > 0:angles["ring"]["MCP_flex"] += angles["ring"]["MCP_abduction"] / 1.8
                    pos[9]  = np.clip(int(150 * (170 - angles["ring"]["MCP_flex"]) + 62.5 * angles["ring"]["MCP_abduction"]), 0, 4095)
                    pos[10] = np.clip(int(150 * (170 - angles["ring"]["MCP_flex"]) - 62.5 * angles["ring"]["MCP_abduction"]), 0, 4095)
                    pos[11] = np.clip(int(37.1 *((177 - angles["ring"]["PIP"])    + (177 - angles["ring"]["DIP"]))), 0, 4095)

                    # 小拇指
                    angles["pinky"]["MCP_flex"] += abs(angles["pinky"]["MCP_abduction"]) / 2.2
                    if angles["pinky"]["MCP_abduction"] > 0:
                        pos[12] = np.clip(int(150*(165 - angles["pinky"]["MCP_flex"]) + 150*angles["pinky"]["MCP_abduction"] ** 2 / 45), 0, 4095)
                        pos[13] = np.clip(int(150*(165 - angles["pinky"]["MCP_flex"])), 0, 4095)
                    else:
                        pos[12] = np.clip(int(150*(165 - angles["pinky"]["MCP_flex"])), 0, 4095)
                        pos[13] = np.clip(int(150*(165 - angles["pinky"]["MCP_flex"]) + 150*angles["pinky"]["MCP_abduction"] ** 2 / 25), 0, 4095)

                    pos[14] = np.clip(int(36.5 *((175 - angles["pinky"]["PIP"])   + (177 - angles["pinky"]["DIP"]))), 0, 4095)

                    # 减少抖动
                    for i in range(16):
                        if abs(pos_now[i] - pos[i]) >= 80:
                            pos_drive[i] = pos[i]
                            pos_now[i] = pos[i]

                except Exception as e:
                    print("角度映射错误:", e)

                # print(f'{pos[3]}\t{pos[4]}\t{pos[6]}\t{pos[7]}\t{pos[9]}\t{pos[10]}\t{pos[12]}\t{pos[13]}')
                send_predefined_action(velocity, pos_drive, current)

                # 绘制手部骨架和角度
                mp_draw.draw_landmarks(image, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
                draw_joint_angles(image, landmarks, angles, w, h)
                draw_axis(image, wrist, axes["x"], axes["y"], axes["z"], 0.12)

            # 显示结果
            cv2.imshow("image", image)
            key = cv2.waitKey(1)
            if key == 27:  # ESC 退出
                break

    finally:
        cv2.destroyAllWindows()
        # pipeline.stop()

if __name__ == "__main__":
    main()

