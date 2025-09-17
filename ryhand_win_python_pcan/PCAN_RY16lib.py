# ... existing imports ...
from time import sleep
from .PCANBasic import *  # PCAN-Basic library import
import asyncio
import time
# PCAN初始化
objPCANBasic = PCANBasic()
PcanHandle = PCAN_USBBUS1
result = objPCANBasic.Initialize(PcanHandle, PCAN_BAUD_1M)
if result != PCAN_ERROR_OK:
    print(f"Failed to initialize the PCAN device: {result}")
    exit(1)

# 定义速度二维数组 ActionTab1
ActionTab1 = [

    [500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,500], 
    [500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,500], 
    [500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,500], 
    [500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,500], 
    [500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,500], 
    [500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,500], 
    [500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,500], 
    [500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,500], 
    [500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,500], 
    [500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,500], 
    [500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,500], 
    [500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,500], 
    [500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,500], 
    [500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,500], 
    [500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,500], 
    [500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,500], 
]

# # 定义二维数组 ActionTab26
ActionTab2 = [
    [4095, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ], 
    [0, 4095, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ], 
    [0, 0, 4095, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ], 
    [0, 0, 0, 4095, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ], 
    [0, 0, 0, 0, 4095, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ], 
    [0, 0, 0, 0, 0, 4095, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ], 
    [0, 0, 0, 0, 0, 0, 4095, 0, 0, 0, 0, 0, 0, 0, 0, 0 ], 
    [0, 0, 0, 0, 0, 0, 0, 4095, 0, 0, 0, 0, 0, 0, 0, 0 ], 
    [0, 0, 0, 0, 0, 0, 0, 0, 4095, 0, 0, 0, 0, 0, 0, 0 ], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 4095, 0, 0, 0, 0, 0, 0 ], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4095, 0, 0, 0, 0, 0 ], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4095, 0, 0, 0, 0 ], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4095, 0, 0, 0 ], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4095, 0, 0 ], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4095, 0 ], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4095 ], 
    
          
    
    ]


ActionTab3 = [
    [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,100], 
    [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,100], 
    [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,100], 
    [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,100], 
    [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,100], 
    [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,100], 
    [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,100], 
    [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,100], 
    [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,100], 
    [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,100], 
    [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,100], 
    [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,100], 
    [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,100], 
    [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,100], 
    [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,100], 
    [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,100], 
    ]


# 发送CAN消息的函数
def send_can_message(motor_id, data):
    pcan_msg = TPCANMsg()
    pcan_msg.ID = motor_id
    pcan_msg.MSGTYPE = PCAN_MESSAGE_STANDARD
    pcan_msg.LEN = len(data)
    for i in range(len(data)):
        pcan_msg.DATA[i] = data[i]

    result = objPCANBasic.Write(PcanHandle, pcan_msg)
    if result != PCAN_ERROR_OK:
        print(f"Failed to send message: {result}")

def receive_can_message():
    """
    非阻塞接收一条 CAN 消息
    有数据就返回 (msg.ID, data)，没数据就返回 (None, None)
    """
    try:
        result, msg, timestamp = objPCANBasic.Read(PcanHandle)
    except Exception as e:
        print("PCAN Read exception:", e)
        return None, None

    if result == PCAN_ERROR_QRCVEMPTY:
        # 没有新消息
        return None, None
    elif result != PCAN_ERROR_OK:
        print(f"Read error: {hex(result)}")
        return None, None
    else:
        data = bytes(msg.DATA[:msg.LEN])
        return msg.ID, data
    

def parse_finger_feedback(buf: bytes):
    """解析一个电机的 8 字节反馈"""
    if len(buf) != 8:
        return None

    raw = int.from_bytes(buf, byteorder="little")  # 小端拼成一个整数

    cmd     = (raw >> 0)  & 0xFF
    status  = (raw >> 8)  & 0xFF
    P       = (raw >> 16) & 0xFFF
    V       = (raw >> 28) & 0xFFF
    I       = (raw >> 40) & 0xFFF
    F       = (raw >> 52) & 0xFFF

    # 速度、电流是有符号的 12bit，需要手动符号扩展
    if V > 2047:  
        V -= 4096
    if I > 2047:
        I -= 4096

    return {
        "cmd": cmd,
        "status": status,
        "P": P,
        "V": V,
        "I": I,
        "F": F,
    }


# 批注:
# 编程时，和前面命令字一直，可以用如下结构体进行解释：

# typedef struct
# {
#     uint64_t :8;            // 前面的CMD
#     uint64_t status:8;   // 故障状态 ，0 表示无故障，异常详情：
# 0 电机正常
# 1 电机过温告警
# 2 电机过温保护
# 3 电机低压保护
# 4 电机过压保护
# 5 电机过流保护
# 6 电机力矩保护
# 7 电机熔丝位错保护
# 8 电机堵转保护
# ...

#     uint64_t P:12;        // 当前位置，0-4095 对应 0到满行程
#     uint64_t V:12;        // 当前速度，-2048~2047 单位 0.001行程/s
#     uint64_t I:12;         // 当前电流，-2048~2047 单位 0.001A
#     uint64_t F:12;        // 当前位置，0-4095 对应手指压力传感器Adc原始值
# }MFingerInfo_t;
# 张礼富:
# 和前面命令字一起


# 封装发送预定义动作的函数
def send_predefined_action1(vel_array, pos_array, current_array):
    """
    发送预设动作
    输入三个长度为16的数组，分别为电机目标速度、位置、电流
    """
    for i in range(16):
        motor_id = i + 1  # 电机ID从0到15
        speed = vel_array[i]  # 获取电机速度
        position = pos_array[i]  # 获取电机位置
        current = current_array[i]
        # 构造CAN消息数据：0xA1是命令，后跟位置（低字节，高字节）和速度（低字节，高字节）
        data = [
            0xAA,  # 命令
            position & 0xFF, (position >> 8) & 0xFF,  # 位置（低字节，高字节）
            speed & 0xFF, (speed >> 8) & 0xFF,  # 速度（低字节，高字节）
            current & 0xFF,(current >>8) & 0xFF

        ]
        send_can_message(motor_id, data)
        # print(f"发送到电机 {motor_id}: 速度={speed}, 位置={position}, 数据={data}")
        sleep(0.001)

# 封装发送预定义动作的函数
async def send_predefined_action(vel_array, pos_array, current_array):
    """
    发送预设动作
    输入三个长度为16的数组，分别为电机目标速度、位置、电流
    """


    for i in range(16):
        motor_id = i + 1  # 电机ID从1到15
        speed = vel_array[i]  # 获取电机速度
        position = pos_array[i]  # 获取电机位置
        dianliu = current_array[i]
        # 构造CAN消息数据：0xA1是命令，后跟位置（低字节，高字节）和速度（低字节，高字节）
        data = [
            0xAA,  # 命令
            position & 0xFF, (position >> 8) & 0xFF,  # 位置（低字节，高字节）
            speed & 0xFF, (speed >> 8) & 0xFF,  # 速度（低字节，高字节）
            dianliu & 0xFF,(dianliu >>8) & 0xFF

        ]
        send_can_message(motor_id, data)
        # print(f"发送到电机 {motor_id}: 速度={speed}, 位置={position}, 数据={data}")
        await asyncio.sleep(0.001)


 # 封装发送预定义动作的函数
async def send_clean():
    """
    发送预定义的动作数据到15个电机
    :param action_index: 动作编号 (0-100)
    """


    for i in range(16):
        motor_id = i + 1  # 电机ID从1到15
        data = [
            0xA5,  # 命令

        ]
        send_can_message(motor_id, data)
        await asyncio.sleep(0.001)



async def main():
    while True:
        await send_predefined_action(ActionTab1[1], ActionTab2[1],ActionTab3[1])
            



# 运行异步主循环
if __name__ == '__main__':
    asyncio.run(main())