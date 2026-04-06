import modbus_tk.exceptions as modbus_exceptions
import modbus_tk.modbus_rtu as modbus_rtu
import serial
import modbus_tk.defines as cst
import time
import keyboard
import socket
import threading
import itertools
import random
import struct
import json
import base64
import cv2
#import pygame
from playsound import playsound
from periphery import GPIO
# import pigpio
import os
from PIL import Image
import numpy as np
#import datetime
from bird import BIRDVIEW
import queue
from getMFCC import micMFCC
from datetime import datetime

#pip install jieba -i https://mirrors.aliyun.com/pypi/simple/
#pip install sktime==0.16.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
#pip3 install opencv-python
# 设置串口名称和波特率
serial_port = '/dev/ttyS0'
#serial_port = 'COM1'
out1_pin=128
out1_gpio=GPIO(out1_pin,"out")
out2_pin=117
out2_gpio=GPIO(out2_pin,"out")
out3_pin=116
out3_gpio=GPIO(out3_pin,"out")
out4_pin=111  # 31脚 
out4_gpio=GPIO(out4_pin,"out")
in1_pin=112
in1_gpio=GPIO(in1_pin,"in")

mainhost = '10700axhr0763.vicp.fun'  # 服务器的IP地址或域名#10700axhr0763.vicp.fun:19278
mainport = 19278        # 服务器的端口号
#mainhost = 'localhost'  # 服务器的IP地址或域名
#mainport = 12345        # 服务器的端口号
localhost='192.168.20.100'
#localhost='127.0.0.1'
rtsp_url = r'rtsp://admin:admin123@192.168.20.50:554/cam/realmonitor?channel=1&subtype=0'
localport=5005
baud_rate = 9600
write485 = 0
writeid=20
writeaddress=0
writecontent=0
main_step=0
poll_step=100
folder_path = "/home/cat/camana/pro/music"
extension = '.mp3'
#mp3count=0
soundin=0
mp3index=0
mainpoll_time=600
ptz_waitVangle=60
noisedB=0
chirpIn=0
serverFindTarget=0
localFindTarget=0
boardPow=100
boardTmp=0
boardHum=0
mainDormancy=0
fieldDormancy=0
cameraoff=0
camerastatus=0
ptz_workVangle=50
ptz_VangleUplimit=80
ptz_VangleDownlimit=5
ptz_HangleLeftlimit=120
ptz_HangleRightlimit=240
dBtrigger=100
workPowerVolume=50
fieldworkPowerVolume=[25,25,25,25,25,25]
fieldOnline=[1,1,1,1,1,1]
fieldOnlineCount=[0,0,0,0,0,0]
fieldInUse=[1,1,1,1,1,1]
deviceID=1
info101id=0
#oldinfo101id=0
info102id=0
#oldinfo102id=0
info103id=0
#oldinfo103id=0
info105id=0
#oldinfo105id=0
info110id=0
#oldinfo110id=0
info301id=0
#oldinfo301id=0
powerVolume=[100,100,100,100,100,100]
switchStatus1=[0,0,0,0,0,0]
switchStatus2=[0,0,0,0,0,0]
switchStatus3=[0,0,0,0,0,0]
switchSet1=[0,0,0,0,0,0]
switchSet2=[0,0,0,0,0,0]
switchSet3=[0,0,0,0,0,0]
switchInUseNo=0
switchSetOn=0
ptz_Hset=0
ptz_Vset=0
ptz_Hread=0
ptz_Vread=0
laserOn=0
soundOn=0
controlType=1 # 1, 自动控制  0 手动控制  2 服务器控制 
send201count=0
send201num=0
send206count=0
send206num=3
send208count=0
send208num=4
inTrigger=0
detectOn=0
olddetectOn=0

localBirdsnum=0 #本地AI识别到鸟，本地最多反馈两个
bird1_x=0  #第一只鸟坐标
bird1_y=0
bird2_x=0  #第二只鸟坐标
bird2_y=0
camera = cv2.VideoCapture(rtsp_url)
# camera = cv2.VideoCapture(rtsp_url,cv2.CAP_FFMPEG)
camera.set(cv2.CAP_PROP_BUFFERSIZE, 10)
# camera.set(cv2.CAP_PROP_HW_ACCELERATION,cv2.VIDEO_ACCELERATION_ANY)
# bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=20,detectShadows=True)
start_time = cv2.getTickCount()
resolution_printed = False
ten_seconds_passed = False
video_width = 1280
video_height = 720
TH_temp=20
TH_hum=20
qdata = queue.Queue()
def get_cpu_temp_linux():
    try:
        with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
            temp_str = f.readline().strip()
            temp = int(temp_str) / 1000.0
            return int(temp)
    except FileNotFoundError:
        return "CPU temperature file not found."
    except Exception as e:
        return f"Error reading CPU temperature: {e}"
 
#print(get_cpu_temp_linux())
def playmusic(file):
    #pygame.init()
    #pygame.mixer.init()
    #pygame.mixer.music.load(file)  # 替换为你的音频文件路径
    #pygame.mixer.music.play()
    #while pygame.mixer.music.get_busy():  # 等待音乐播放完毕
     #   pygame.time.Clock().tick(10)
    playsound(file)
def find_files_by_extension(directory, extension):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                yield os.path.join(root, file)    
def playmp3():
    ind=0
    global mp3index
    global folder_path
    global extension
    for filepath in find_files_by_extension(folder_path, extension):
        if ind==mp3index:
            playmusic(filepath)
        ind=ind+1
    mp3index=mp3index+1
    if(mp3index>=ind):
        mp3index=0

def has_ten_seconds_passed():
    # 检查是否已经过了10秒
    elapsed_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
    return elapsed_time > 10
def detect_birds_ai(q):
    birdview= BIRDVIEW(q)
    birdview.getbird()
    
def detect_birds():
    global video_height
    global video_width
    global localBirdsnum
    global bird1_x
    global bird1_y
    global resolution_printed
    global ten_seconds_passed
    global detectOn
    global olddetectOn
    global start_time
    global localFindTarget
    global bg_subtractor
    global controlType
    global camera
    # 初始化变量来存储背景帧
    background_frame = None
    # if not camera.isOpened():
    #     print('无法打开视频流')
    #     return
    try:
        while not camera.isOpened():
            print('无法打开视频流')
            time.sleep(5)
        while True:
            if(controlType==3):
                time.sleep(3)
                continue
            try:
                ret, frame = camera.read()
            except Exception as e:
                print(f"Error occurred: {e}")
                continue
            # ret, frame = camera.grab()
            if detectOn==0 or controlType==0:
                time.sleep(1)
                continue
            if not ret:
                print('无法从视频流中读取帧，退出...')
                time.sleep(1)
                camera.release()
                cv2.destroyAllWindows()
                detectOn=0
                olddetectOn=0
                
                time.sleep(1)
                camera = cv2.VideoCapture(rtsp_url)
                # camera = cv2.VideoCapture(rtsp_url,cv2.CAP_FFMPEG)
                camera.set(cv2.CAP_PROP_BUFFERSIZE, 10)
                # camera.set(cv2.CAP_PROP_HW_ACCELERATION,cv2.VIDEO_ACCELERATION_ANY)
                # camera.set(cv2.CAP_PROP_HW_DEVICE,0)
                while not camera.isOpened():
                    print('无法打开视频流')
                    time.sleep(3)
                continue
            elif (detectOn==1 and olddetectOn==0):
                print('重新获取背景，...')
                time.sleep(1)
                camera.release()
                cv2.destroyAllWindows()
                olddetectOn=2
              
                time.sleep(1)
                camera = cv2.VideoCapture(rtsp_url)
                # camera = cv2.VideoCapture(rtsp_url,cv2.CAP_FFMPEG)
                camera.set(cv2.CAP_PROP_BUFFERSIZE, 10)
                # camera.set(cv2.CAP_PROP_HW_ACCELERATION,cv2.VIDEO_ACCELERATION_ANY)
                # camera.set(cv2.CAP_PROP_HW_DEVICE,0)
                while not camera.isOpened():
                    print('无法打开视频流')
                    time.sleep(3)
                continue

            now = datetime.now()
            #timestamp = now.strftime("%Y%m%d_%H%M%S")
            if not resolution_printed:
                height, width = frame.shape[:2]
                print(f'视频流分辨率: {width}x{height}')
                video_width = width
                video_height = height
                resolution_printed = True
            if(detectOn==1) and (olddetectOn<6):
                # if olddetectOn==2 or olddetectOn==6  or olddetectOn==10 or olddetectOn==14 :
                # bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=1, varThreshold=20,detectShadows=False)
                # bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
                # bg_subtractor.empty()
                # bg_subtractor.setVarInit(30)  # 设置初始方差，影响模型对新样本的适应性
                # bg_subtractor.setVarMin(15)   # 设置最小方差，防止模型过于僵硬
                # bg_subtractor.setVarMax(500)  # 设置最大方差，防止模型过于灵活导致不稳定

                
            #    # 使用背景减除器更新背景模型并获取前景掩码
            #     fgMask = bg_subtractor.apply(frame)
                
            #     # 反转掩码以得到背景（白色区域为背景）
            #     background = cv2.bitwise_not(fgMask)
            #     # 将掩码应用到原帧以得到背景图像（可选步骤）
            #     background_image = cv2.bitwise_and(frame, frame, mask=background)
            #     # 转换为图像
            #     background_frame = background_image.copy()
                
                if background_frame is not None and olddetectOn==5:
                     
                    
                    # filename = f'F:/DynoTron/XZH/EPylonDriveBird/Program/temp/back/background_image_{timestamp}.png'
                    # filename = f'/home/cat/camana/background_image_{timestamp}.png'
                    # cv2.imwrite(filename, background_frame)
                    start_time = cv2.getTickCount()
                    ten_seconds_passed = False
                    print("restart background cap")
                # img_array = np.array(bg_subtractor)  # 将数据转换为NumPy数组
                # img = Image.fromarray(img_array)  # 创建图像对象
                # cv2.imwrite('back.jpg', bg_subtractor)
                # img.save('temp/back.jpg', 'JPEG')
                olddetectOn +=1
                # start_time = cv2.getTickCount()
                # ten_seconds_passed = False
                # print("restart background cap")
                time.sleep(0.2)
                continue    
                
            # 背景减除
            fg_mask = bg_subtractor.apply(frame)

            # 寻找轮廓
            contours,_= cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 筛选最大轮廓
            max_area = 0
            max_contour = None
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > max_area:
                    max_area = area
                    max_contour = contour

            # 检查是否已经过了10秒
            if not ten_seconds_passed:
                ten_seconds_passed = has_ten_seconds_passed()

            # 如果存在最大轮廓，且面积大于阈值，并且已经过了10秒，则绘制矩形框
            if max_contour is not None and max_area >5000 and max_area <400000 and ten_seconds_passed:
                (x, y, w, h) = cv2.boundingRect(max_contour)
                if w>1000 or x<3 or y<3 :
                    pass
                else:
                    
                    # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    # 输出矩形框的坐标和尺寸
                    print(f'矩形框坐标和尺寸: x={x}, y={y}, width={w}, height={h}')
                    # send_message(x,y,w,h)
                    localBirdsnum=1
                    localFindTarget=1
                    bird1_x=x
                    bird1_y=y
                    start_time = cv2.getTickCount()
                    ten_seconds_passed = False
                    # filename = f'/home/cat/camana/snap_image_{timestamp}.png'
                    # cv2.imwrite(filename, frame)
                    #cv2.imshow('Bird Detection', frame)
                    #time.sleep(1)
            # 显示结果
            # time.sleep(1)
            #cv2.imshow('Bird Detection', frame)

        # 按'q'键退出
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
    finally:
        # 释放VideoCapture对象
        camera.release()

        # 关闭所有OpenCV窗口
        cv2.destroyAllWindows()
def resetOut(no):
    if no==1:
        out1_gpio.write(False)
    elif no==2:
        out2_gpio.write(False)
    elif no==3:
        out3_gpio.write(False)
    elif no==4:
        out4_gpio.write(False)
    else:
        pass
def setOut(no):
    if no==1:
        out1_gpio.write(True)
    elif no==2:
        out2_gpio.write(True)
    elif no==3:
        out3_gpio.write(True)
    elif no==4:
        out4_gpio.write(True)
    else:
        pass
def readOut(no):
    if no==1:
        response=out1_gpio.read()
    elif no==2:
        response=out2_gpio.read()
    elif no==3:
        response=out3_gpio.read()
    elif no==4:
        response=out4_gpio.read()
    else:
        pass
    return response
def readIn(no):
    if no==1:
        response=in1_gpio.read()
    # elif no==2:
    #     response=in2_gpio.read()
    # elif no==3:
    #     response=in3_gpio.read()
    else:
        pass
    return response
def doLaserOn():
    setOut(1)
def doLaserOff():
    resetOut(1)
def doFanOn():
    setOut(2)
def doFanOff():
    resetOut(2)
def doThermOn():
    setOut(3)
def doThermOff():
    resetOut(3)
def readChirp():
    return readIn(1)
#tcp server
def tcp_handle_client(client_socket,addr):#分布式和小主机通讯
    global controlType
    global deviceID
    global info101id
    #global oldinfo101id
    global info102id
    #global oldinfo102id
    global info103id
    #global oldinfo103id
    global info105id
    #global oldinfo105id
    global info110id
    #global oldinfo110id
    global info301id
    #global oldinfo301id
    global powerVolume
    global switchStatus1
    global switchStatus2
    global switchStatus3
    global switchSet1
    global switchSet2
    global switchSet3
    global workPowerVolume
    global fieldworkPowerVolume
    global mainpoll_time
    global ptz_workVangle
    global ptz_HangleLeftlimit
    global ptz_HangleRightlimit
    global ptz_VangleUplimit
    global ptz_VangleDownlimit
    global serverFindTarget
    global ptz_Hset
    global ptz_Vset
    global laserOn
    global send201count
    global send201num
    global send206count
    global send206num
    global send208count
    global send208num
    global boardPow
    global localBirdsnum
    global fieldOnline
    
    try:
        ad=addr[0].split('.')
        id=str(int(ad[3])-181)
        mesno206=0
        mesno208=0
        did=int(id)-1
        client_socket.settimeout(1)
        while True:
            now=datetime.now()
            rnow=now.strftime("%H")
            if mesno206>send206num:
                
                str206="fieldcontrol,"
                str206+=str(send206count)
                str206+=","
                
                str206+=id
                str206+=",206,"
                str206+=str(switchSet1[did])
                str206+=","
                str206+=str(switchSet2[did])
                str206+=","
                str206+=str(switchSet3[did])
                str206+=",fieldcontrolend#"
               # print(str206)
                try:
                    client_socket.sendall(str206.encode('utf-8'))
                except Exception as e:
                    pass
                mesno206=0
                send206count+=1
            else:
                mesno206+=1
            if mesno208>send208num:
                
                #心跳包 发到分布式 208
                str208="fieldinfo2,"
                str208+=str(send208count)
                str208+=","
                str208+=id
                str208+=",208,"
                if(controlType==3) or ((fieldDormancy==1) and(int(rnow)>18 or int(rnow)<7)):
                    str208+="1"
                else:
                    str208+="0"
                str208+=",fieldinfo2end#"
                # print(str208)
                try:
                    client_socket.sendall(str208.encode('utf-8'))
                except Exception as e:
                    pass
                mesno208=0
                send208count+=1
            else:
                mesno208+=1
          
            try:
                message = client_socket.recv(1024)
                if not message:
                    break
                # client_socket.sendall(message)  # Echo the received data back to the client
                tempdata=message.decode()
                # print(tempdata)
                if '#' not in tempdata:
                    continue
                data = tempdata.split('#')[0]
                if "fieldcontrol" not in data:
                    continue
                words=data.split(',')
                if len(words)<8:
                    continue
                if words[0]=="fieldcontrol" and int(words[1])!=info301id and int(words[3])==301:
                    powerVolume[int(words[2])-1]=int(words[4])
                    switchStatus1[int(words[2])-1]=int(words[5])
                    switchStatus2[int(words[2])-1]=int(words[6])
                    switchStatus3[int(words[2])-1]=int(words[7])
                    #oldinfo301id=info301id
                    info301id=int(words[1])
                    fieldOnlineCount[int(words[2])-1]=0
                    fieldOnline[int(words[2])-1]=1
                    # print ("fieldstatus")
                    # print (words[2])
                    # print(str(switchStatus1[2]))
                    print (data)
                else:
                    pass
                    # if fieldOnlineCount[int(words[2])-1]>50 :#判断离线
                    #     fieldOnline[int(words[2])-1]=0
                    # else :
                    #     fieldOnlineCount[int(words[2])-1]+=1
            except socket.timeout:
                #print("No data received within the timeout period.")
                pass
            except ConnectionResetError:
                pass
            except Exception as e:
                pass
            time.sleep(1)    
    finally:
        client_socket.close()
        print("Connection closed.")
def tcp_server(host=localhost, port=localport):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen(8)
        #conn, addr = s.accept()
        #with conn:
         #   with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        #s.bind((host, port))
        #s.listen()
        print(f"Server started on {host}:{port}")
        try:

            while True:
                client_sock, addr = s.accept()
                print(f"Connected by {addr}")
                client_thread = threading.Thread(target=tcp_handle_client, args=(client_sock,addr))
                client_thread.start()
                # client_thread.join()
        finally:
            s.close()
            print("Server closed.")

# def connect_to_server(host, port):
#     try:
#         # 创建socket对象
#         s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#         # 连接到服务器
#         s.connect((host, port))
#         print(f"Connected to {host}:{port}")
#         while True:
#             # 发送和接收数据
#             message = mainmessage
#             s.sendall(message.encode())
#             data = s.recv(1024)
#             print(f"Received: {data.decode()}")
#     except (ConnectionResetError, ConnectionRefusedError) as e:
#         print(f"Connection error: {e}. Attempting to reconnect...")
#     finally:
#         # 关闭socket连接
#         s.close()
#         # print("Connection closed.")
def connect_to_server(host, port, timeout=5, retries=5, delay=10):#和服务器通讯
    """
    尝试连接到服务器。
    
    :param host: 服务器地址
    :param port: 服务器端口
    :param timeout: 连接超时时间（秒）
    :param retries: 最大重试次数
    :param delay: 每次重试之间的延迟（秒）
    :return: 成功时返回socket对象，失败时返回None
    """
    global controlType
    global deviceID
    global info101id
    #global oldinfo101id
    global info102id
    #global oldinfo102id
    global info103id
    #global oldinfo103id
    global info105id
    #global oldinfo105id
    global info110id
    #global oldinfo110id
    global info301id
    #global oldinfo301id
    global powerVolume
    global switchStatus1
    global switchStatus2
    global switchStatus3
    global switchSet1
    global switchSet2
    global switchSet3
    global workPowerVolume
    global fieldworkPowerVolume
    global mainpoll_time
    global ptz_workVangle
    global ptz_HangleLeftlimit
    global ptz_HangleRightlimit
    global ptz_VangleUplimit
    global ptz_VangleDownlimit
    global serverFindTarget
    global ptz_Hset
    global ptz_Vset
    global laserOn
    global send201count
    global send201num
    global send206count
    global send206num
    global send208count
    global send208num
    global boardPow
    global localBirdsnum
    global fieldOnline
    global ptz_Hread
    global ptz_Vread
    attempts = 0
    sendcount=0
    while attempts < retries:
        # s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # s.settimeout(timeout)
        # s.connect((host, port))
        # print(f"Connected to {host}:{port}")
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # s.setblocking(False)  # 设置为非阻塞模式
            s.settimeout(timeout)
            s.connect((host, port))
            print(f"Connected to {host}:{port}")
            s.settimeout(3)
            while True:
                # 发送和接收数据
                #接收
                
                #发送
                if sendcount>=3:
                    sendcount=0
                    str201="fieldinfo,"
                    str201+=str(send201count)
                    str201+=","
                    str201+=str(deviceID)
                    str201+=",201,"
                    str201+=str(boardPow)
                    if main_step<100 or poll_step<100 :
                        str201+=",1,"
                    else:
                        str201+=",0,"
                    str201+=str(localBirdsnum)
                    str201+=","
                    str201+=str(bird1_x)
                    str201+="_"
                    str201+=str(bird1_y)
                    str201+=","
                    str201+=str(bird2_x)
                    str201+="_"
                    str201+=str(bird2_y)
                    str201+=","
                    str201+=str(laserOn)
                    str201+=","
                    str201+=str(ptz_Hset)
                    str201+=","
                    str201+=str(ptz_Vset)
                    str201+=","
                    str201+=str(fieldOnline[0])
                    str201+=","
                    str201+=str(powerVolume[0])
                    str201+=","
                    str201+=str(switchStatus1[0])
                    str201+=","
                    str201+=str(switchStatus2[0])
                    str201+=","
                    str201+=str(switchStatus3[0])
                    str201+=","
                    str201+=str(fieldOnline[1])
                    str201+=","
                    str201+=str(powerVolume[1])
                    str201+=","
                    str201+=str(switchStatus1[1])
                    str201+=","
                    str201+=str(switchStatus2[1])
                    str201+=","
                    str201+=str(switchStatus3[1])
                    str201+=","
                    str201+=str(fieldOnline[2])
                    str201+=","
                    str201+=str(powerVolume[2])
                    str201+=","
                    str201+=str(switchStatus1[2])
                    str201+=","
                    str201+=str(switchStatus2[2])
                    str201+=","
                    str201+=str(switchStatus3[2])
                    str201+=","
                    str201+=str(fieldOnline[3])
                    str201+=","
                    str201+=str(powerVolume[3])
                    str201+=","
                    str201+=str(switchStatus1[3])
                    str201+=","
                    str201+=str(switchStatus2[3])
                    str201+=","
                    str201+=str(switchStatus3[3])
                    str201+=","
                    str201+=str(fieldOnline[4])
                    str201+=","
                    str201+=str(powerVolume[4])
                    str201+=","
                    str201+=str(switchStatus1[4])
                    str201+=","
                    str201+=str(switchStatus2[4])
                    str201+=","
                    str201+=str(switchStatus3[4])
                    str201+=","
                    str201+=str(fieldOnline[5])
                    str201+=","
                    str201+=str(powerVolume[5])
                    str201+=","
                    str201+=str(switchStatus1[5])
                    str201+=","
                    str201+=str(switchStatus2[5])
                    str201+=","
                    str201+=str(switchStatus3[5])
                    str201+=",fieldinfoend"
                    # message = "\"cmdType\":201,\"id\":"+str(deviceID)+","+"\"content\":\""+str201+"\"#";
                    """序列化为紧凑JSON字节流"""
                    payload = {
                        "cmdType": 201,  # 字段简称减少数据量
                        "id": deviceID,  # 控制浮点数精度
                        "content": str201,
                        
                    }
                    # s.sendall(message.encode())
                    message=json.dumps(payload, separators=(',', ':')).encode('utf-8')+"#".encode('utf-8')
                    try:
                        s.sendall(message)
                    except Exception as e:
                        pass
                    
                    # print ("201 sent")
                    if localBirdsnum>0:
                        localBirdsnum=0
                    # num = random.randint(1, 100)
                    # # 将整数转换为字节（使用64位整数格式）
                    # packed_num = struct.pack('q', num)
                    # # 发送字节
                    # s.sendall(packed_num)
                else:
                    sendcount=sendcount+1
                try:

                    message = s.recv(150*1024)
                   
                    if not message:
                        break
                    tempdata=message.decode()
                    # print(tempdata)
                    if '#' not in tempdata:
                        continue
                    data = tempdata.split('#')[0]
                    # print (data)
                    if "cmdType\":2" in data:
                        data=data[data.index("content")+10:len(data)-2]
                        # print(data)
                        audio_data = base64.b64decode(data.encode('utf-8'))
                        # audio_data = base64.b64decode(data)
                        # audio_filename = "f:/play.mp3"
                        audio_filename = "/home/cat/camana/play.mp3"
                        with open(audio_filename, "wb") as audio_file:
                            audio_file.write(audio_data)
                    elif "controlinfo" in data:
                        
                        data=data[data.index("controlinfo"):]
                        # print(data)
                        words = data.split(',')
                        # print(words[3])
                        if words[0]=="controlinfo" and int(words[2])==deviceID and int(words[1])!=info101id and int(words[3])==101:
                            controlType=int(words[4])
                            if(write485 !=10):
                                write485=10
                            # print(str(controlType))
                            #oldinfo101id=info101id
                            info101id=int(words[1])
                        elif words[0]=="controlinfo" and int(words[2])==deviceID and int(words[1])!=info102id and int(words[3])==102:
                            ptz_Hread=int(words[4])
                            ptz_Vread=int(words[5])
                            # print(str(ptz_Hread))
                            # print(str(controlType))
                            laserOn=int(words[6])
                            switchSet1[0]=int(words[7])
                            switchSet2[0]=int(words[8])
                            switchSet3[0]=int(words[9])
                            switchSet1[1]=int(words[10])
                            switchSet2[1]=int(words[11])
                            switchSet3[1]=int(words[12])
                            switchSet1[2]=int(words[13])
                            switchSet2[2]=int(words[14])
                            switchSet3[2]=int(words[15])
                            switchSet1[3]=int(words[16])
                            switchSet2[3]=int(words[17])
                            switchSet3[3]=int(words[18])
                            switchSet1[4]=int(words[19])
                            switchSet2[4]=int(words[20])
                            switchSet3[4]=int(words[21])
                            switchSet1[5]=int(words[22])
                            switchSet2[5]=int(words[23])
                            switchSet3[5]=int(words[24])
                            #oldinfo102id=info102id
                            info102id=int(words[1])
                        elif words[0]=="controlinfo" and int(words[2])==deviceID and int(words[1])!=info103id and int(words[3])==103:
                            mainpoll_time=int(words[4])
                            ptz_workVangle=int(words[5])
                            ptz_HangleLeftlimit=int(words[6])
                            ptz_HangleRightlimit=int(words[7])
                            ptz_VangleUplimit=int(words[8])
                            ptz_VangleDownlimit=int(words[9])
                            workPowerVolume=int(words[10])
                            fieldworkPowerVolume[0]=int(words[11])
                            fieldworkPowerVolume[1]=int(words[12])
                            fieldworkPowerVolume[2]=int(words[13])
                            fieldworkPowerVolume[3]=int(words[14])
                            fieldworkPowerVolume[4]=int(words[15])
                            fieldworkPowerVolume[5]=int(words[16])
                            data = {
                                "mainpoll_time": str(words[4]),
                                "ptz_workVangle": str(words[5]),
                                "ptz_VangleUplimit": str(words[8]),
                                "ptz_VangleDownlimit": str(words[9]),
                                "ptz_HangleLeftlimit": str(words[6]),
                                "ptz_HangleRightlimit": str(words[7]),
                                "workPowerVolume": str(words[10]),
                                "fieldworkPowerVolume0": str(words[11]),
                                "fieldworkPowerVolume1": str(words[12]),
                                "fieldworkPowerVolume2": str(words[13]),
                                "fieldworkPowerVolume3": str(words[14]),
                                "fieldworkPowerVolume4": str(words[15]),
                                "fieldworkPowerVolume5": str(words[16]),
                                "mainDormancy":str(mainDormancy),
                                "fieldDormancy":str(fieldDormancy),
                                "cameraoff":str(cameraoff),
                                "dBtrigger":str(dBtrigger)
                            }
                             
                            with open('lubancat.json', 'w', encoding='utf-8') as file:
                                json.dump(data, file, ensure_ascii=False, indent=4)
                            #oldinfo103id=info103id
                            info103id=int(words[1])
                        elif words[0]=="controlinfo" and int(words[2])==deviceID and int(words[1])!=info105id and int(words[3])==105:
                            serverFindTarget=int(words[4])
                            #oldinfo105id=info105id
                            info105id=int(words[1])
                        elif words[0]=="controlinfo" and int(words[2])==deviceID and int(words[1])!=info110id and int(words[3])==110:
                            # controlType=int(words[4])
                            #oldinfo110id=info110id
                            info110id=int(words[1])
                        else:
                            pass
                    
                except socket.timeout:
                    #print("No data received within timeout.")
                    pass
                # except BlockingIOError:  # Python 2中捕获此异常以处理非阻塞模式下的情形
                #     print("No data available at the moment.")
                # print("2222222222222222222")
                time.sleep(1)
            # return s
        except socket.error as e:
            print(f"Connection failed: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
            attempts += 1
    print("Failed to connect after multiple attempts.")
    return None
def tcp_client2main(host=mainhost, port=mainport):#小主机联系服务器 
    # with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    #     sock.connect((host, port))
    #     sock.sendall(message.encode())
    #     response = sock.recv(1024)
    #     print("Received:", response.decode())
    # while True:
    #     try:
    #         connect_to_server(host, port)
    #         # 如果连接成功，则不进行重连等待，直接开始新的连接
    #         break
    #     except Exception as e:
    #         print(f"Reconnect failed: {e}. Retrying in 5 seconds...")
    #         time.sleep(5)  # 等待5秒后重试
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        # sock.connect((host, port))
        connect_to_server(host, port) 
    while True:
        if sock is None or sock.fileno() == -1:  # 检查socket是否有效或已关闭
            # print("8888888888888")
            sock = connect_to_server(host, port)  # 重连
            
            time.sleep(5)
            # if sock is None:  # 如果重连失败，则退出循环或进行其他处理
            #     break
        
#         try:
#             # 发送数据和接收响应的代码...
#             # sock.sendall(b"Hello, server!")  # 示例发送数据
#             message = "Hello, server!"
# #           s.sendall(message.encode())
#             response = sock.recv(1024)  # 示例接收数据
#             print(f"Received from server: {response.decode()}")
#         except socket.error as e:  # 如果发生错误，尝试重新建立连接（可选）
#             print(f"Error occurred: {e}. Attempting to reconnect...")
#             sock.close()  # 关闭当前socket（如果有）并等待重连逻辑处理新的连接。
#             sock = None  # 重置socket变量以便重连逻辑触发。
# 创建串口连接

conn = serial.Serial(serial_port, baud_rate, 8, 'N', 1, 0)
master = modbus_rtu.RtuMaster(conn)
master.set_timeout(1)  # 设置超时时间
master.set_verbose(True)
# 发送PELCO-D命令
# 例如，将摄像头向左移动10步（具体命令取决于摄像头的支持）
# 格式通常为：FF <address> <command> <data> <checksum>
# 其中<address>通常为0x01，<command>和<data>根据需要设置，<checksum>是地址、命令和数据部分的和的补码
def send_pelco_d_command(address, command, data):
    # print(str(command))
    # message = bytes([0xFF, hex(address),0x00,hex(command),hex(45),hex(56)])
    message = bytes([0xFF, address])
    message+=bytes([command>>8 & 0xFF])
    message+=bytes([command & 0xFF])
    message+=bytes([data>>8 & 0xFF])
    # print(str(data))
    message+=bytes([data & 0xFF])
    checksum = (sum(message[1:])) & 0xFF  # 计算校验和（补码）
    # print(str(command & 0xFF))
    message += bytes([checksum])  # 添加校验和到消息末尾
    conn.write(message)  # 发送消息
    # print("999999999999999999999999999999")
    #print(message)
    time.sleep(0.3)  # 等待一小段时间让摄像头响应命令  
def scan_rs485_devices():
  #  devices = []
    global write485
    global writeid
    global writeaddress
    global writecontent
    global noisedB
    global chirpIn
    global boardPow
    global boardTmp
    global boardHum
    #global h_angle
    #global v_angle
    global TH_hum
    global TH_temp
    global laserOn
    global ptz_Hset
    global ptz_Vset
    readcount=0
    dBcount=0
    autospeed=32 #0x20
    manualspeed=16 #0x10
    # 创建串口连接
    #try:
     #   conn = serial.Serial(serial_port, baud_rate, 8, 'N', 1, 0)
    #except Exception as e:
        #pass
     #   print(f"Error start serial: {str(e)}")
    #master = modbus_rtu.RtuMaster(conn)
    #master.set_timeout(1)  # 设置超时时间
    #master.set_verbose(True)
    # powerVolume=[100,100,100,100,100,100]
    # switchStatus1=[0,0,0,0,0,0]
    # switchStatus2=[0,0,0,0,0,0]
    # switchStatus3=[0,0,0,0,0,0]
    while True:
        '''
        chirpIn=readChirp()
        # print(chirpIn)
        if (chirpIn==1) :
            print("receive bird chirp")
        # else:
        #     print("987654321123456789")
        '''
        if(TH_temp>35) or (get_cpu_temp_linux()>55):
            setOut(2)
            switchSet2[0]=1
            switchSet2[1]=1
            switchSet2[2]=1
            switchSet2[3]=1
            switchSet2[4]=1
            switchSet2[5]=1
        elif (TH_temp<30):
            resetOut(2)
            switchSet2[0]=0
            switchSet2[1]=0
            switchSet2[2]=0
            switchSet2[3]=0
            switchSet2[4]=0
            switchSet2[5]=0
        if(laserOn==1):
            doLaserOn()
        else :
            doLaserOff()
        onlinecount=0
        while onlinecount<6:
            if fieldOnlineCount[onlinecount]>10 :#判断离线
                fieldOnline[onlinecount]=0
            else :
                fieldOnlineCount[onlinecount]+=1
            onlinecount=onlinecount+1
        
        if write485==0:
            #噪声
            try:
                response = master.execute(11, 3, 0, 1)
                tempdB=response[0]/10
                if dBcount<5:
                    if noisedB<tempdB:
                        noisedB=tempdB
                    dBcount=dBcount+1
                else :
                    dBcount=0
                    noisedB=tempdB
                        
                #print('噪音分贝数----->', noisedB,'dB')
            except Exception as e:
                #print(f"Error scanning device at 11_noise module: {str(e)}")
                pass
            if readcount>=10:   
                readcount=0
                #温湿度
                try:
                    response = master.execute(12, 3, 0x0, 2)
                    if response[0]<32767:
                        #print('温度值--------->', response[0]/10,'℃')
                        TH_temp=response[0]/10
                    else :
                        # print('温度值--------->', 0-(response[0]-32768)/10,'℃')
                        TH_temp=0-(response[0]-32768)/10
                        #print('湿度值--------->', TH_temp,'%')
                    TH_hum=response[1]/10
                except Exception as e:
                    pass
                    # print(f"Error scanning device at 12_TH module: {str(e)}")
                
                # 读取指定设备寄存器的指定位置的值 master.execute(设备地址, 读的功能码3, 寄存器地址, 读取长度)
                #电量
                try:
                    response = master.execute(13, 4, 0x0, 2)
                    #print('电量百分比----->', response[0],'%')
                    boardPow=response[0]
                    #print(str(boardPow))
                    #print('电压值--------->', response[1]/100,'V')
                except Exception as e:
                    #print(f"Error scanning device at 13_battery meter module: {str(e)}")
                    pass
            else:
                readcount=readcount+1
            time.sleep(1.0)
            # print(get_cpu_temp_linux())
        elif (write485>=1) and (write485 < 10):
            #云台地址20 水平 4B 75  垂直 4D 77
            try:
                # write_RS485_Devices(writeid,writeaddress,writecontent)
                #print('2222222222222222222222222222222222')                   
                send_pelco_d_command(writeid, writeaddress, writecontent)
                
            except Exception as e:
                #print(f"Error writing device at 20_PTZ module: {str(e)+str(writecontent)}")
                pass
            write485+=1
            #print(str(writecontent))
            if write485>=3:
                write485 = 0
            time.sleep(0.5)
        elif (write485>=10) and (write485 < 20):
            #云台地址20 水平 4B 75  垂直 4D 77
            try:
                # write_RS485_Devices(writeid,writeaddress,writecontent)
                if(controlType==0):
                    send_pelco_d_command(writeid, 2, manualspeed*256) #right
                    time.sleep(0.2)
                    send_pelco_d_command(writeid, 4, manualspeed*256)
                    time.sleep(0.2)
                    send_pelco_d_command(writeid, 8, manualspeed)#up
                    time.sleep(0.2)
                    send_pelco_d_command(writeid, 16, manualspeed)
                    time.sleep(0.2)
                else :
                    send_pelco_d_command(writeid, 2, autospeed*256) #right
                    time.sleep(0.2)
                    send_pelco_d_command(writeid, 4, autospeed*256)
                    time.sleep(0.2)
                    send_pelco_d_command(writeid, 8, autospeed)#up
                    time.sleep(0.2)
                    send_pelco_d_command(writeid, 16, autospeed)
                    time.sleep(0.2)
                write485+=4
                if(write485>20):
                    write485=0
                #print('2222222222222222222222222222222222')
            except Exception as e:
                #print(f"Error writing device at 20_PTZ module: {str(e)+str(writecontent)}")
                pass
            time.sleep(0.1) 
 
    conn.close()
    master.close()  # 关闭串口连接
 

def write_RS485_Devices(id,address,message):
    # 给指定设备的寄存器写值  master.execute(设备地址, 写的功能码6, 寄存器地址, output_value=写入的值)
    response = master.execute(id, 6, address, output_value=message)
    print('写入返回结果----->', response)  # 返回一个数组，第一个为寄存器地址，第二个为里面的值
def HangleCal(ang):
    global ptz_Hset
    if ang>ptz_HangleRightlimit:
        ang=ptz_HangleRightlimit
    elif ang<=ptz_HangleLeftlimit:
        ang=ptz_HangleLeftlimit
    ptz_Hset=ang
    return ang*100
def VangleCal(ang):
    global ptz_Vset
    if ang>ptz_VangleUplimit:
        ang=ptz_VangleUplimit
    elif ang<=ptz_VangleDownlimit:
        ang=ptz_VangleDownlimit
    ptz_Vset=ang
    return ang*100
    #if ang>=0:
     #   return 36000-ang*100
    #else :
     #   return (0-ang)*100

def main_server(qdata):
    global write485
    global writeid
    global writeaddress
    global writecontent
    global main_step
    global poll_step
    global noisedB
    global chirpIn
    global boardTmp
    global boardHum
    global serverFindTarget
    global localFindTarget
    global inTrigger
    global controlType
    global switchSet1
    global switchSet2
    global switchSet3
    global switchInUseNo
    global switchSetOn
    global laserOn
    global detectOn
    global ptz_Vread
    global ptz_Hread
    global olddetectOn
    global start_time
    #global mp3count
    global camerastatus
    global soundin
    mp3count=0
    timecount=0
    timenum=0
    # Trigger206=0
    pollfreezecount=100
    pollfreezenum=100
    fieldcount=0
    nightcount=0
    nightin=0
    while True:
        now=datetime.now()
        rnow=now.strftime("%H")
        #print(str(rnow))      
        try:
            msg = qdata.get(block=False)  # 阻塞直到队列中有项目,非阻塞
            if msg is not None:
                #print("Received:", msg)
                print(f'矩形框坐标和尺寸: x={msg[0]}, y={msg[1]}, x2={msg[2]}, y2={msg[3]}')
                    # send_message(x,y,w,h)
                localBirdsnum=1
                localFindTarget=1
                bird1_x=msg[0]
                bird1_y=msg[1]
        except Exception as e:
            #print( f"queue error: {e}")
            pass
        i=0
        while i<6:
            # if powerVolume[i]>fieldworkPowerVolume[i] and fieldOnline[i]>0:
            if fieldOnline[i]>0:
                fieldInUse[i]=1
            else:
                fieldInUse[i]=0
            i+=1
        
        if controlType==1 or controlType==2:
            # chirpIn=readChirp()
            # if  (((noisedB>dBtrigger) or ((chirpIn==1) and (noisedB>dBtrigger)) or (serverFindTarget>0) or(localFindTarget>0) ) and (mp3count==0)):
            '''
            if  (((noisedB>dBtrigger) or (chirpIn==1) ) and (mp3count==0)):
                playmp3()
                mp3count+=1
                if chirpIn==1:
                    chirpIn=0
                if noisedB>dBtrigger:
                    noisedB=0
                # if localFindTarget>0:
                #     localFindTarget=0
                # if serverFindTarget>0:
                #     serverFindTarget=0
            if mp3count>0 :
                mp3count+=1
                if mp3count>20:
                    mp3count=0
            '''
            #print("123456789")
            if int(rnow)>18 or int(rnow)<7:
                if nightcount>2400:#40分钟
                    nightcount=0
                if(mainDormancy==1) and (cameraoff==0):#摄像头识别驱动激光休眠，只采用定时轮询 1有效
                    
                    if nightcount<5:
                        setOut(4)
                        camerastatus=1
                    elif nightcount>2100 and nightcount<2110:#打开摄像头，云台
                        resetOut(4)
                        camerastatus=0
                    elif nightcount>2150 and nightcount<2155:#触发激光扫描
                        nightin=1
                    else:
                        pass
                    
                else:
                    if nightcount==0:
                        resetOut(4)
                        camerastatus=0
                nightcount=nightcount+1
                
                
                    
                if(cameraoff==1) and (camerastatus==0):#cameraoff 关闭摄像头和云台，1有效
                    setOut(4)
                    camerastatus=1
                
            else:
                #if(camerastatus==1):
                resetOut(4)
                camerastatus=0
                nightin=0
                nightcount=0
            if  (((noisedB>dBtrigger) or (nightin==1) or (serverFindTarget>0) or(localFindTarget>0) ) and (fieldcount==0)):
                switchSet1[switchInUseNo]=1
                switchSet1[switchInUseNo+3]=1
                switchSetOn=3 #206指令
                fieldcount=1

            if fieldcount>0 :
                fieldcount+=1
                if fieldcount>240:
                    fieldcount=0
                    switchInUseNo+=1
                elif fieldcount>=15:
                    switchSet1[switchInUseNo]=0
                    switchSet1[switchInUseNo+3]=0
                    switchSetOn=3
                    
                    if(switchInUseNo>2):
                        switchInUseNo=0

            #if ((inTrigger==0) and ((noisedB>dBtrigger) or (chirpIn==1) or (serverFindTarget>0) or (localFindTarget>0))) and poll_step==100 and pollfreezecount>60:
            #if ( (localFindTarget>0)) and poll_step==100 and pollfreezecount>pollfreezenum and boardPow>40:
            if ( (localFindTarget>0) or nightin==1) and main_step>0 and main_step<100 and pollfreezecount>pollfreezenum and boardPow>40:
            #if ( (localFindTarget>0)) and poll_step==100 and pollfreezecount>pollfreezenum and boardPow>60:
                inTrigger=1
                #print("1111111111111111111111111")
                main_step=100
                poll_step=0
                pollfreezecount=0
                detectOn=0
                olddetectOn=0
                soundin=1
            elif ( (localFindTarget>0)) :
                soundin=1
            if localFindTarget>0:
                localFindTarget=0
            if serverFindTarget>0:
                serverFindTarget=0
            if nightin==1:
                nightin=0
            if( poll_step<70):
                laserOn=1
                #激光器打开
            else:
                laserOn=0
            if  ((chirpIn==1 ) or noisedB>dBtrigger or soundin==1)and (mp3count==0)and boardPow>20:
                playmp3()
                mp3count=1
                soundin=0
            
            if mp3count>300:
                mp3count=0
            elif mp3count>0:
                mp3count+=1
                
                
            time.sleep(0.8)
                #激光器关闭
            # if(poll_step<90) and Trigger206==0:
            #     switchSet1[switchInUseNo]=1
            #     switchSet1[switchInUseNo+3]=1
            #     switchSetOn=3 #206指令
            #     Trigger206=1
            # elif (poll_step==100) and Trigger206==1:
            #     switchSet1[switchInUseNo]=0
            #     switchSet1[switchInUseNo+3]=0
            #     switchSetOn=3
            #     switchInUseNo+=1
            #     Trigger206=0
            #     if(switchInUseNo==3):
            #         switchInUseNo=0
            #print("123456789")
            if main_step>=100 and poll_step>=100:
                main_step=0
                poll_step=100
                
            if main_step==0:
                    #垂直角度 到-5
                    writeaddress=77
                    writecontent=VangleCal(10)
                    write485=1
                    main_step=5
                    timecount=0
                    timenum=5
                    inTrigger=0
                    print("m0")
            elif main_step==10:
                    #垂直角度 到-5
                    
                    writeaddress=77
                    writecontent=VangleCal(ptz_waitVangle)
                    write485=1
                    main_step=15
                    timecount=0
                    timenum=5
                    inTrigger=0
                    print("m10")
            elif main_step==20:
                #垂直角度 到-15
                    
                    writeaddress=77
                    #角度正
                    writecontent=VangleCal(ptz_workVangle)
                    write485=1
                    main_step=25
                    timecount=0
                    timenum=10
                    print("m20")
                    
            elif main_step==30:
                    #水平角度150
                    
                    writeaddress=75
                    writecontent=HangleCal(ptz_HangleLeftlimit)
                    write485=1
                    main_step=35
                    timecount=0
                    timenum=40
                    print("m30")
                    
            elif main_step==40:
                    #水平角度180
                    
                    writeaddress=75
                    writecontent=HangleCal(180)
                    write485=1
                    #time.sleep(6)
                    main_step=45
                    timecount=0
                    timenum=40
                    print("m40")
                    
            elif main_step==50:
                    #水平角度210
                    
                    writeaddress=75
                    writecontent=HangleCal(ptz_HangleRightlimit)
                    write485=1
                    #time.sleep(6)
                    main_step=55
                    timecount=0
                    timenum=40
                    print("m50")
                    
            elif main_step==60:
                    #水平角度180
                    
                    writeaddress=75
                    writecontent=HangleCal(180)
                    write485=1
                    #time.sleep(6)
                    main_step=65
                    timecount=0
                    timenum=40
                    print("m60")
                    
            elif main_step==70:
                    #垂直角度 到-5
                    
                    writeaddress=77
                    writecontent=VangleCal(10)
                    write485=1
                    main_step=75
                    timecount=0
                    print("m70")
                    timenum=mainpoll_time #最后休眠等待600秒
            elif main_step>=100:
                    pass
            else:
                    if int(rnow)>19 or int(rnow)<7:
                        pass
                    else:
                        
                        timecount=timecount+1
                        if timecount>=timenum:
                            timecount=0
                            timenum=0
                            main_step=main_step+5
                            if main_step>=80:
                                main_step=0
                                poll_step=100
                        #elif timecount>=8 and detectOn==0:
                        #    detectOn=1
                         #   olddetectOn=0

                         #   start_time = cv2.getTickCount()
                        #elif (timecount<8 and timecount>0) or timecount>42:
                         #  detectOn=0
                        #   olddetectOn=0

            if poll_step==0:
                #垂直角度 到
                    
                    detectOn=0
                    olddetectOn=0
                    writeaddress=77
                    #角度正
                    writecontent=VangleCal(ptz_workVangle-3)
                    # writecontent=VangleCal(3)
                    write485=1
                    poll_step=15
                    timecount=0
                    timenum=5
                    print("p0")
            elif poll_step==20:
                    #水平角度60
                    
                    writeaddress=75
                    writecontent=HangleCal(ptz_HangleLeftlimit)
                    write485=1
                    poll_step=25
                    timecount=0
                    timenum=5
                    print("p20")
                    #time.sleep(6)
            elif poll_step==30:
                    #水平角度300
                    
                    writeaddress=75
                    writecontent=HangleCal(ptz_HangleRightlimit)
                    write485=1
                    #time.sleep(6)
                    poll_step=35
                    timecount=0
                    timenum=5
                    print("p30")
            elif poll_step==40:
                #垂直角度 到
                    
                    writeaddress=77
                    #角度正
                    writecontent=VangleCal(ptz_workVangle+3)
                    #writecontent=VangleCal(1)
                    write485=1
                    poll_step=45
                    timecount=0
                    timenum=5
                    print("p40")
            elif poll_step==50:
                    #水平角度
                    
                    writeaddress=75
                    writecontent=HangleCal(ptz_HangleLeftlimit)
                    write485=1
                    poll_step=55
                    timecount=0
                    timenum=5
                    print("p50")
                    #time.sleep(6)
            elif poll_step==60:
                    #水平角度180
                    
                    writeaddress=75
                    writecontent=HangleCal(180)
                    write485=1
                    #time.sleep(6)
                    poll_step=65
                    timecount=0
                    timenum=3
                    print("p60")
            elif poll_step==70:
                    #垂直角度 到-5
                    
                    writeaddress=77
                    #角度正
                    writecontent=VangleCal(10)
                    write485=1
                    poll_step=75
                    timecount=0
                    print("p70")
                    timenum=mainpoll_time #最后休眠等待600秒
            elif poll_step>=100:
                    pass
            else :
                    timecount=timecount+1
                    if timecount>=timenum:
                        timecount=0
                        timenum=0
                        poll_step=poll_step+5
                        if poll_step>=80:
                            poll_step=100
                            main_step=0
                            inTrigger=0
                            pollfreezecount=0
            time.sleep(1)
            if (pollfreezecount<60000):
                pollfreezecount=pollfreezecount+1
            #ptz_Hset=180
            #ptz_Vset=7
        elif controlType==0:
            # cycle_iterator = itertools.cycle(switchSet1)
            # for _ in range(6):
            #     sta=next(cycle_iterator)
            #     if(sta==0):
            
            if ptz_Vread==1:
                # print(str(ptz_Vset))
                writeaddress=77
                writecontent=VangleCal(ptz_Vset-1)
                write485=1
                ptz_Vread=0
            elif ptz_Vread==2:
                # print(str(ptz_Vset))
                writeaddress=77
                writecontent=VangleCal(ptz_Vset+1)
                write485=1
                ptz_Vread=0
            elif ptz_Hread==3:
                # print(str(ptz_Hset))
                writeaddress=75
                writecontent=HangleCal(ptz_Hset-5)
                write485=1
                ptz_Hread=0
            elif ptz_Hread==4:
                # print(str(ptz_Hset))
                writeaddress=75
                writecontent=HangleCal(ptz_Hset+5)
                write485=1
                ptz_Hread=0
            # pass
            main_step=0
            poll_step=100     
            time.sleep(2) 
        elif (controlType==3):
                time.sleep(3)
                # continue
        else :
            pass
def mfccdetect():
    global chirpIn
    #global noisedB
  #  global soundin
  #  mp3count=0
  #  acton=0
    #mfcc= micMFCC()
    
    while True:
        #print("acton")
        chirpIn=micMFCC.predictMfcc()
        #print(str(chirpIn))
     #   if acton==0:
            #print("sound process.......")
        #    chirpIn=micMFCC.predictMfcc()
            #print(str(chirpIn))
        #    if  ((chirpIn==1 ) or noisedB>dBtrigger or soundin==1)and (mp3count==0)and boardPow>20:
        #        acton=1
       #         print("sound acton")
        time.sleep(2)
      #  else :
       #     if mp3count==0:
       #         playmp3()
      #      mp3count+=1
            #if chirpIn==1:
             #   chirpIn=0
                #if noisedB>dBtrigger:
                #    noisedB=0
        #    if mp3count>30:
       #         mp3count=0
       #         acton=0
      #          soundin=0
                
     #       time.sleep(1000)

def run_tcp_server():
    tcp_server()  # Start TCP server on default settings or customize parameters here
def run_tcp_client2main():
    tcp_client2main()
def run_modbus_server():
    scan_rs485_devices()
def run_main_server(q):
    main_server(q)
def run_video_server(q):
    detect_birds_ai(q)
def run_mfcc_server():
    mfccdetect()
tcp_thread = threading.Thread(target=run_tcp_server)
modbus_thread = threading.Thread(target=run_modbus_server)
main_thread = threading.Thread(target=run_main_server,args=(qdata,))
tcp_client2main_thread=threading.Thread(target=run_tcp_client2main)
video_thread=threading.Thread(target=run_video_server,args=(qdata,))
mfcc_thread= threading.Thread(target=run_mfcc_server)
resetOut(1)
resetOut(2)
resetOut(4)

#write485 = 10
time.sleep(10)
with open('lubancat.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
    #print(data)
    #data = json.loads(json_str)
    #print(data)
    mainpoll_time=int(data['mainpoll_time'])
    ptz_workVangle=int(data['ptz_workVangle'])
    ptz_VangleUplimit=int(data['ptz_VangleUplimit'])
    ptz_VangleDownlimit=int(data['ptz_VangleDownlimit'])
    ptz_HangleLeftlimit=int(data['ptz_HangleLeftlimit'])
    ptz_HangleRightlimit=int(data['ptz_HangleRightlimit'])
    workPowerVolume=int(data['workPowerVolume'])
    fieldworkPowerVolume[0]=int(data['fieldworkPowerVolume0'])
    fieldworkPowerVolume[1]=int(data['fieldworkPowerVolume1'])
    fieldworkPowerVolume[2]=int(data['fieldworkPowerVolume2'])
    fieldworkPowerVolume[3]=int(data['fieldworkPowerVolume3'])
    fieldworkPowerVolume[4]=int(data['fieldworkPowerVolume4'])
    fieldworkPowerVolume[5]=int(data['fieldworkPowerVolume5'])
    mainDormancy=int(data['mainDormancy'])
    fieldDormancy=int(data['fieldDormancy'])
    cameraoff=int(data['cameraoff'])
    dBtrigger=int(data['dBtrigger'])
#time.sleep(1)
resetOut(3)
writeaddress=75
writecontent=HangleCal(210)
write485=1
time.sleep(8)
writeaddress=75
writecontent=HangleCal(180)
write485=1
time.sleep(8)
writeaddress=77
writecontent=VangleCal(10)
write485=1
time.sleep(8)
#writeaddress=77
#writecontent=VangleCal(60)
#write485=1
#time.sleep(5)

#playmusic('/home/cat/camana/pro/music/alarms.mp3')
playmp3()
time.sleep(3)
write485 = 10
#time.sleep(15)
tcp_thread.start()
modbus_thread.start()
main_thread.start()
tcp_client2main_thread.start()
video_thread.start()
mfcc_thread.start()
tcp_thread.join()  # Wait for TCP server to finish if needed (optional)
modbus_thread.join()  # Wait for Modbus server to finish if needed (optional)
main_thread.join()
tcp_client2main_thread.join()
video_thread.join()
mfcc_thread.join()
