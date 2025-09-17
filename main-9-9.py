import pyrealsense2 as rs
import numpy as np
import mediapipe as mp
import cv2
from utils.image_tools import *
from utils.math_tools import *
from ryhand_win_python_pcan.PCAN_RY16lib import *

# ---------------- 摄像头初始化 ---------------

# 创建 RealSense 流管道
pipeline = rs.pipeline()

# 创建配置对象
config = rs.config()

# 尝试设置合适的分辨率和帧率
config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)  # 深度流
# config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)  # 彩色流
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # 彩色流


# 启动流管道
pipeline.start(config)




# --------------- MediaPipe 初始化 --------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


vel = [500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,500]
pos = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
current = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,100]

# 异步主循环
async def FingerCtrl():
    while True:
        await send_predefined_action(vel, pos, current)
        await asyncio.sleep(0.001)
        

async def VisionLoop():
    """主逻辑任务：相机/mediapipe处理"""
    # 创建 Hand 对象
    with mp_hands.Hands(
        model_complexity = 0,
        min_detection_confidence = 0.5,
        min_tracking_confidence = 0.5,
        max_num_hands = 1) as hands:
        
        try:
            while True:
                # 等待新的图像帧
                frames = pipeline.wait_for_frames()

                # 获取深度和颜色帧
                # depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                # 将帧转换为 NumPy 数组
                # depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                color_h, color_w, _ = color_image.shape

                # 转换图像为 RGB 格式 (MediaPipe 需要 RGB 图像)
                image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

                # 使用 MediaPipe Hands 处理图像
                results = hands.process(image_rgb)

                # 绘制手部标注
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            color_image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style()
                        )


                        # for idx, landmark in enumerate(hand_landmarks.landmark):
                        #     h, w, c = color_image.shape
                        #     x = landmark.x 
                        #     y = landmark.y 
                        #     z = landmark.z  
                        #     if idx == 8:
                        #         print(f'关节 {mp_hands.HandLandmark(idx)}: ({x}, \t{y}, \t{z})')


                        kps_pixel = {
                            name: np.array([int(lm.x * color_w), int(lm.y * color_h), lm.z])  # 转为 NumPy 数组
                            for name, lm in zip(
                                mp.solutions.hands.HandLandmark.__members__.keys(),
                                hand_landmarks.landmark
                            )
                        }
                        try:
                            # 建立每个节点的坐标点


                            # 计算手掌三条边的比例长度，最大比例长度, 用作近似不变量 （用于估算手掌深度）
                            edge_proportion = [0.93, 0.82, 0.57]
                            palm_edge1 = np.linalg.norm([(kps_pixel["WRIST"][0] - kps_pixel["INDEX_FINGER_MCP"][0]), (kps_pixel["WRIST"][1] - kps_pixel["INDEX_FINGER_MCP"][1])]) / edge_proportion[0]
                            palm_edge2 = np.linalg.norm([(kps_pixel["WRIST"][0] - kps_pixel["PINKY_MCP"][0]), (kps_pixel["WRIST"][1] - kps_pixel["PINKY_MCP"][1])]) / edge_proportion[1]
                            palm_edge3 = np.linalg.norm([(kps_pixel["INDEX_FINGER_MCP"][0] - kps_pixel["PINKY_MCP"][0]), (kps_pixel["INDEX_FINGER_MCP"][1] - kps_pixel["PINKY_MCP"][1])]) / edge_proportion[2]
                            max_edge = max(palm_edge1, palm_edge2, palm_edge3)
                            # print(f' {palm_edge1}, \t{palm_edge2}, \t{palm_edge3}')
                            # print(max_edge)
                            
                            distance_proportion = 0.233

                            bones_length = {
                                "palm_edge1":3.9,
                                "palm_edge2":3.45,
                                "palm_edge3":2.4,
                                
                                "index_proximal":1.3,
                                "index_middle":0.9,
                                "index_distal":0.8,

                                "middle_proximal":1.7,
                                "middle_middle":1.0,
                                "middle_distal":0.9,
                                
                                "ring_proximal":1.4,
                                "ring_middle":0.9,
                                "ring_distal":0.8,

                                "pinky_proximal":1.1,
                                "pinky_middle":0.8,
                                "pinky_distal":0.7,
                            }
                            
                            bones_length = {
                                key: value * distance_proportion * max_edge 
                                for key, value in bones_length.items()
                            }
                            palm_edge1_model = np.linalg.norm([(kps_pixel["WRIST"][0] - kps_pixel["INDEX_FINGER_MCP"][0]), (kps_pixel["WRIST"][1] - kps_pixel["INDEX_FINGER_MCP"][1])])
                            palm_edge2_model = np.linalg.norm([(kps_pixel["WRIST"][0] - kps_pixel["PINKY_MCP"][0]), (kps_pixel["WRIST"][1] - kps_pixel["PINKY_MCP"][1])])
                            palm_edge3_model = np.linalg.norm([(kps_pixel["INDEX_FINGER_MCP"][0] - kps_pixel["PINKY_MCP"][0]), (kps_pixel["INDEX_FINGER_MCP"][1] - kps_pixel["PINKY_MCP"][1])])
                            

                            # print(f'{kps_pixel["INDEX_FINGER_MCP"][2]}\t{kps_pixel["PINKY_MCP"][2]}')
                            depth = CalculateDepth(bones_length["palm_edge1"], kps_pixel["INDEX_FINGER_MCP"], kps_pixel["WRIST"])
                            depth = 0 if np.isnan(depth) else depth
                            
                            print(f'{depth}')
                            # print(f'{bones_length["palm_edge1"]}\t{palm_edge1_model}\t{bones_length["palm_edge2"]}\t{palm_edge2_model}\t{bones_length["palm_edge3"]}\t{palm_edge3_model}')













                            

                            # 获取坐标系的方向 (手腕作为原点，手指尖端作为坐标轴的方向)
                            # 建立手掌坐标系
                            
                            origin_point = kps_pixel["WRIST"]

                            mcp_points = np.array([
                                [x, y, z] for x, y, z in [
                                    kps_pixel["INDEX_FINGER_MCP"],
                                    kps_pixel["MIDDLE_FINGER_MCP"],
                                    kps_pixel["RING_FINGER_MCP"],
                                    kps_pixel["PINKY_MCP"]
                                ]
                            ])  
                            centroid, direction = RegressionLine(mcp_points)   # 建立手掌上端基准线，用于定位手掌坐标轴。
                            
                            z_dir = np.cross(direction, centroid - np.array([origin_point[0], origin_point[1], origin_point[2]]))  # z轴为垂直手掌，指向为掌心向外
                            y_dir = -direction.copy()  # y轴平行该基准线
                            x_dir = np.cross(y_dir, z_dir) # x轴通过右手定则确定

                            z_axis = z_dir / np.linalg.norm(z_dir) if np.linalg.norm(z_dir) > 1e-5 else z_dir
                            x_axis = x_dir / np.linalg.norm(x_dir) if np.linalg.norm(x_dir) > 1e-5 else x_dir
                            y_axis = y_dir / np.linalg.norm(y_dir) if np.linalg.norm(y_dir) > 1e-5 else y_dir
                            


                            
                            new_index_mcp = PointTransform(kps_pixel["INDEX_FINGER_MCP"], origin_point, x_axis, y_axis)
                            new_index_pip = PointTransform(kps_pixel["INDEX_FINGER_PIP"], origin_point, x_axis, y_axis)
                            new_index_dip = PointTransform(kps_pixel["INDEX_FINGER_DIP"], origin_point, x_axis, y_axis)
                            new_index_tip = PointTransform(kps_pixel["INDEX_FINGER_TIP"], origin_point, x_axis, y_axis)
                            
                            index_dip_bending = CalculateTheta(new_index_tip - new_index_dip, new_index_dip - new_index_pip) * 180 / np.pi
                            index_pip_bending = CalculateTheta(new_index_dip - new_index_pip, new_index_pip - new_index_mcp) * 180 / np.pi
                            # print((index_pip_bending + index_dip_bending) / 2)




                            # 在节点旁边进行数据标注
                            AnnotateNumber(color_image, (int(kps_pixel["INDEX_FINGER_PIP"][0] * color_w), int(kps_pixel["INDEX_FINGER_PIP"][1] * color_h)), index_pip_bending )
                            AnnotateNumber(color_image, (int(kps_pixel["INDEX_FINGER_DIP"][0] * color_w), int(kps_pixel["INDEX_FINGER_DIP"][1] * color_h)), index_dip_bending )
                            


                            

                            # pos[5] = index_pip_bending

                            # send_predefined_action(vel, pos, current)








                            # # 绘制手掌上边界基准线

                            # axis_length = 0.2  # 坐标轴箭头长度

                            # # 计算直线起点和终点（在质心两侧）
                            # start_point = centroid - direction * axis_length / 2
                            # end_point = centroid + direction * axis_length / 2
                            
                            # start_2d = project_3d_to_2d(start_point, color_w, color_h)
                            # end_2d = project_3d_to_2d(end_point, color_w, color_h)
                            # cv2.line(color_image, start_2d, end_2d, (0, 255, 255), 2)  # 黄色直线
                            
                            # 绘制坐标轴

                            # cv2.arrowedLine(color_image, 
                            #             (int(color_w * origin_point[0]), int(color_h * origin_point[1])), 
                            #             (int(color_w * (origin_point[0] + x_axis[0] * axis_length)), 
                            #                 int(color_h * (origin_point[1] + x_axis[1] * axis_length))), 
                            #             (0, 0, 255), 2)  # 红色 X轴

                            # cv2.arrowedLine(color_image, 
                            #             (int(color_w * origin_point[0]), int(color_h * origin_point[1])), 
                            #             (int(color_w * (origin_point[0] + y_axis[0] * axis_length)), 
                            #                 int(color_h * (origin_point[1] + y_axis[1] * axis_length))), 
                            #             (0, 255, 0), 2)  # 绿色 Y轴

                            # cv2.arrowedLine(color_image, 
                            #             (int(color_w * origin_point[0]), int(color_h * origin_point[1])), 
                            #             (int(color_w * (origin_point[0] + z_axis[0] * axis_length)), 
                            #                 int(color_h * (origin_point[1] + z_axis[1] * axis_length))), 
                            #             (255, 0, 0), 2)  # 蓝色 Z轴
                            
                        except Exception as e:
                            print(f"坐标绘制错误: {e}")
                            continue

                            

                # 显示带有标注的图像
                # cv2.imshow('Hand Detection', cv2.flip(color_image, 1))
                # cv2.imshow('hand', depth_image)
                cv2.imshow('Hand Detection', color_image)

                # 检查用户是否按下了 'q' 键来退出
                key = cv2.waitKey(1)
                if key == ord('q') or key == 27:  # 按 'q' 或 'Esc' 键退出
                    break
        except Exception as e:
            print(e)
        finally:
            # 停止流管道
            pipeline.stop()
            cv2.destroyAllWindows()

async def main():
    """程序入口：并发运行多个任务"""
    # 创建并发任务
    task_finger = asyncio.create_task(FingerCtrl())
    task_vision = asyncio.create_task(VisionLoop())
    
    # 并发运行，直到所有任务结束（这里不会结束）
    await asyncio.gather(task_finger, task_vision)

if __name__ == "__main__":
    asyncio.run(main())





