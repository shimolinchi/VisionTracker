from VisionTracker.VisionTracker_object import VisualTracker

# 这两个矩阵为对指位置。四行中每一行分别是从食指到小指分别与大拇指对指的电机位置
# 第一个矩阵为手指（食指到小拇指）较平时的对指姿态，第二个为手指（食指到小拇指）近节弯曲幅度较小时的对指姿态

pointing_positions = [
    [ 1900, 1900, 2950, 1200, 1200, 2418, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2000 ],
    [ 2179, 2179, 2264, 0, 0, 0, 1482, 1482, 2200, 0, 0, 0, 0, 0, 0, 3145 ],
    [ 2303, 2369, 2400, 0, 0, 0, 0, 0, 0, 1161, 1161, 2000, 0, 0, 0, 3910 ],
    [ 1220, 2160, 2400, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1210, 1210, 1660, 4095 ]
]

second_pointing_positions = [
    [ 3600, 3600, 999, 1, 1, 2000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1500 ],
    [ 3800, 3800, 850 ,0 ,0, 0, 1, 1, 2500, 0, 0, 0, 0, 0, 0, 3145 ],
    [ 3700, 3700, 850, 0, 0, 0, 0, 0, 0, 1, 1, 2200, 0, 0, 0, 3910 ],
    [ 3600, 3900, 850, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2100, 4095 ]
]

# 对指优化强度
pointing_strength = 0.08

# 对指优化阈值(距离单位为米)
pointing_threadhold = 0.1

if __name__ == "__main__":
    try:
        mytracker = VisualTracker(pointing_positions, second_pointing_positions, pointing_strength, pointing_threadhold)
        mytracker.track()
    except Exception as e:
        print(e)
