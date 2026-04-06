import argparse
import time
import cv2
import imutils
#from FlowPuser import StreamPusher
#from Yolov5Compents import YOLOv5
import urllib
import sys
import numpy as np
import platform
from rknnlite.api import RKNNLite
from queue import Queue
import threading
from datetime import datetime

class BIRDVIEW:
    def __init__(self, qdata):
        
        # decice tree for rk356x/rk3588
        self.DEVICE_COMPATIBLE_NODE = '/proc/device-tree/compatible'
        #qdata.clear()
        
        self.RK3566_RK3568_RKNN_MODEL = 'yolov5s_for_rk3566_rk3568.rknn'
        self.RK3588_RKNN_MODEL = 'yolov5s_for_rk3588.rknn'
        self.RK3562_RKNN_MODEL = 'yolov5s_for_rk3562.rknn'
        #self.IMG_PATH = './bus.jpg'

        self.OBJ_THRESH = 0.25
        self.NMS_THRESH = 0.45
        self.IMG_SIZE = 640
        self.imgpath =  r'rtsp://admin:admin123@192.168.20.50:554/cam/realmonitor?channel=1&subtype=0'
        #self.rtmp_server = 'rtmp://你的IP地址:1935/video'

        self.CLASSES = ("person", "bicycle", "car", "motorbike ", "aeroplane ", "bus ", "train", "truck ", "boat", "traffic light",
                   "fire hydrant", "stop sign ", "parking meter", "bench", "bird", "cat", "dog ", "horse ", "sheep", "cow", "elephant",
                   "bear", "zebra ", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
                   "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife ",
                   "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza ", "donut", "cake", "chair", "sofa",
                   "pottedplant", "bed", "diningtable", "toilet ", "tvmonitor", "laptop ", "mouse   ", "remote ", "keyboard ", "cell phone", "microwave ",
                   "oven ", "toaster", "sink", "refrigerator ", "book", "clock", "vase", "scissors ", "teddy bear ", "hair drier", "toothbrush ")
        self.cap = cv2.VideoCapture(self.imgpath)
        self.frame = None
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.update)
        self.thread.start()
        self.qdata = qdata
        #self.camerapower = 1
    # decice tree for rk356x/rk3588
    #DEVICE_COMPATIBLE_NODE = '/proc/device-tree/compatible'

    def get_host(self):
        # get platform and device type
        system = platform.system()
        machine = platform.machine()
        os_machine = system + '-' + machine
        if os_machine == 'Linux-aarch64':
            try:
                with open(self.DEVICE_COMPATIBLE_NODE) as f:
                    device_compatible_str = f.read()
                    if 'rk3588' in device_compatible_str:
                        host = 'RK3588'
                    elif 'rk3562' in device_compatible_str:
                        host = 'RK3562'
                    else:
                        host = 'RK3566_RK3568'
            except IOError:
                print('Read device node {} failed.'.format(DEVICE_COMPATIBLE_NODE))
                exit(-1)
        else:
            host = os_machine
        return host
        
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))


    def xywh2xyxy(self,x):
        # Convert [x, y, w, h] to [x1, y1, x2, y2]
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y


    def process(self,input, mask, anchors):

        anchors = [anchors[i] for i in mask]
        grid_h, grid_w = map(int, input.shape[0:2])

        box_confidence = self.sigmoid(input[..., 4])
        box_confidence = np.expand_dims(box_confidence, axis=-1)

        box_class_probs = self.sigmoid(input[..., 5:])

        box_xy = self.sigmoid(input[..., :2])*2 - 0.5

        col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
        row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)
        col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
        row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
        grid = np.concatenate((col, row), axis=-1)
        box_xy += grid
        box_xy *= int(self.IMG_SIZE/grid_h)

        box_wh = pow(self.sigmoid(input[..., 2:4])*2, 2)
        box_wh = box_wh * anchors

        box = np.concatenate((box_xy, box_wh), axis=-1)

        return box, box_confidence, box_class_probs


    def filter_boxes(self,boxes, box_confidences, box_class_probs):
        """Filter boxes with box threshold. It's a bit different with origin yolov5 post process!

        # Arguments
            boxes: ndarray, boxes of objects.
            box_confidences: ndarray, confidences of objects.
            box_class_probs: ndarray, class_probs of objects.

        # Returns
            boxes: ndarray, filtered boxes.
            classes: ndarray, classes for boxes.
            scores: ndarray, scores for boxes.
        """
        boxes = boxes.reshape(-1, 4)
        box_confidences = box_confidences.reshape(-1)
        box_class_probs = box_class_probs.reshape(-1, box_class_probs.shape[-1])

        _box_pos = np.where(box_confidences >= self.OBJ_THRESH)
        boxes = boxes[_box_pos]
        box_confidences = box_confidences[_box_pos]
        box_class_probs = box_class_probs[_box_pos]

        class_max_score = np.max(box_class_probs, axis=-1)
        classes = np.argmax(box_class_probs, axis=-1)
        _class_pos = np.where(class_max_score >= self.OBJ_THRESH)

        boxes = boxes[_class_pos]
        classes = classes[_class_pos]
        scores = (class_max_score* box_confidences)[_class_pos]

        return boxes, classes, scores


    def nms_boxes(self,boxes, scores):
        """Suppress non-maximal boxes.

        # Arguments
            boxes: ndarray, boxes of objects.
            scores: ndarray, scores of objects.

        # Returns
            keep: ndarray, index of effective boxes.
        """
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]

        areas = w * h
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x[i], x[order[1:]])
            yy1 = np.maximum(y[i], y[order[1:]])
            xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
            yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

            w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
            h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
            inter = w1 * h1

            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= self.NMS_THRESH)[0]
            order = order[inds + 1]
        keep = np.array(keep)
        return keep


    def yolov5_post_process(self,input_data):
        masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
                   [59, 119], [116, 90], [156, 198], [373, 326]]

        boxes, classes, scores = [], [], []
        for input, mask in zip(input_data, masks):
            b, c, s = self.process(input, mask, anchors)
            b, c, s = self.filter_boxes(b, c, s)
            boxes.append(b)
            classes.append(c)
            scores.append(s)

        boxes = np.concatenate(boxes)
        boxes = self.xywh2xyxy(boxes)
        classes = np.concatenate(classes)
        scores = np.concatenate(scores)

        nboxes, nclasses, nscores = [], [], []
        for c in set(classes):
            inds = np.where(classes == c)
            b = boxes[inds]
            c = classes[inds]
            s = scores[inds]

            keep = self.nms_boxes(b, s)

            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])

        if not nclasses and not nscores:
            return None, None, None

        boxes = np.concatenate(nboxes)
        classes = np.concatenate(nclasses)
        scores = np.concatenate(nscores)

        return boxes, classes, scores


    def draw(self,image, boxes, scores, classes):
        """Draw the boxes on the image.

        # Argument:
            image: original image.
            boxes: ndarray, boxes of objects.
            classes: ndarray, classes of objects.
            scores: ndarray, scores of objects.
            all_classes: all classes name.
        """
        for box, score, cl in zip(boxes, scores, classes):
            top, left, right, bottom = box
            print('class: {}, score: {}'.format(self.CLASSES[cl], score))
            print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top, left, right, bottom))
            top = int(top)
            left = int(left)
            right = int(right)
            bottom = int(bottom)

            cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
            cv2.putText(image, '{0} {1:.2f}'.format(self.CLASSES[cl], score),
                        (top, left - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 2)


    def letterbox(im, new_shape=(640, 640), color=(0, 0, 0)):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, ratio, (dw, dh)
    def update(self):
        self.cap.set(cv2.CAP_PROP_FPS,10)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
        while self.running:
            if not self.cap.isOpened():
                print(f"Error: Could not open {self.imgpath}")
                time.sleep(2)
                self.cap = cv2.VideoCapture(self.imgpath)
                continue
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            else:
                print('无法从视频流中读取帧，退出...')
                time.sleep(1)
                self.cap.release()
                #cv2.destroyAllWindows()
                time.sleep(3)
                

                    
    def read(self):
        with self.lock:
            return self.frame

        
    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()
    
    def getbird(self):
        #parser = argparse.ArgumentParser()
        #parser.add_argument('--imgpath', type=str, default='0', help="image path or camera index")
        #parser.add_argument('--modelpath', type=str, default='../models/yolov5s.onnx', help="onnx filepath")
        #parser.add_argument('--confThreshold', default=0.3, type=float, help='class confidence')
        #parser.add_argument('--nmsThreshold', default=0.5, type=float, help='nms iou thresh')
        #args = parser.parse_args()

        host_name = self.get_host()
        if host_name == 'RK3566_RK3568':
            rknn_model = self.RK3566_RK3568_RKNN_MODEL
        elif host_name == 'RK3562':
            rknn_model = self.RK3562_RKNN_MODEL
        elif host_name == 'RK3588':
            rknn_model = self.RK3588_RKNN_MODEL
        else:
            print("This demo cannot run on the current platform: {}".format(host_name))
            exit(-1)
            
        # Create RKNN object
        rknn_lite = RKNNLite()

          # load RKNN model
        print('--> Load RKNN model')
        ret = rknn_lite.load_rknn(rknn_model)
        if ret != 0:
            print('Load RKNN model failed')
            exit(ret)
        print('done')

        # Init runtime environment
        print('--> Init runtime environment')
        # run on RK356x/RK3588 with Debian OS, do not need specify target.
        if host_name == 'RK3588':
            ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
        else:
            ret = rknn_lite.init_runtime()
        if ret != 0:
            print('Init runtime environment failed!')
            exit(ret)
        print('done')

        #yolov5_detector = YOLOv5(args.modelpath, conf_thres=args.confThreshold, iou_thres=args.nmsThreshold)

        # imgpath = args.imgpath
        #imgpath = '../../test.mp4'
        #imgpath =  r'rtsp://admin:admin123@192.168.1.50:554/cam/realmonitor?channel=1&subtype=0'
        #cap = None
        #pusher = None

        try:
            # 处理摄像头 or 视频文件
            #if imgpath.isnumeric() and len(imgpath) == 1:
            #    cap = cv2.VideoCapture(int(imgpath))  # 读取摄像头
            #else:
            #   cap = cv2.VideoCapture(imgpath)  # 读取视频文件

            #if not cap.isOpened():
            #    print(f"Error: Could not open {self.imgpath}")
            #    exit()

            #pusher = StreamPusher(rtmp_server)

            # 获取摄像头 FPS
            #fps = cap.get(cv2.CAP_PROP_FPS)
            #print(f"摄像头 FPS: {fps}")
            #cap.set(cv2.CAP_PROP_FPS,10)
            #cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
            #fail_count = 0
            #MAX_FAILS = 30  # 最多允许 30 帧读取失败

            #prev_time = time.time()
            while True:
                #cap = cv2.VideoCapture(imgpath)  # 读取视频文件
                #success, srcimg = cap.read()
                now=datetime.now()
                rnow=now.strftime("%H")
                self.frame = self.read()
                if self.frame is None:
                    if int(rnow)>19 or int(rnow)<7:
                        time.sleep(30)
                    else:
                        time.sleep(2)
                    continue
                #if not success:
                    #fail_count += 1
                    #if fail_count > MAX_FAILS:
                    #    print("连续30帧无法读取，退出")
                    #    break
                #    continue  # 跳过当前帧

                #img = imutils.resize(srcimg, width=640)
                
                # Set inputs
                #img = cv2.imread(IMG_PATH)
                #img, ratio, (dw, dh) = letterbox(img, new_shape=(IMG_SIZE, IMG_SIZE))
                img = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                img= img[0:720,200:1080]
                img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))

                # Inference
                print('--> Running model')
                outputs = rknn_lite.inference(inputs=[img])
                #np.save('./onnx_yolov5_0.npy', outputs[0])
                #np.save('./onnx_yolov5_1.npy', outputs[1])
                #np.save('./onnx_yolov5_2.npy', outputs[2])
                print('done')

                # post process
                input0_data = outputs[0]
                input1_data = outputs[1]
                input2_data = outputs[2]

                input0_data = input0_data.reshape([3, -1]+list(input0_data.shape[-2:]))
                input1_data = input1_data.reshape([3, -1]+list(input1_data.shape[-2:]))
                input2_data = input2_data.reshape([3, -1]+list(input2_data.shape[-2:]))

                input_data = list()
                input_data.append(np.transpose(input0_data, (2, 3, 0, 1)))
                input_data.append(np.transpose(input1_data, (2, 3, 0, 1)))
                input_data.append(np.transpose(input2_data, (2, 3, 0, 1)))

                boxes, classes, scores = self.yolov5_post_process(input_data)
                
                # 目标检测
                #boxes, scores, class_ids = yolov5_detector.detect(srcimg)
                #if classes[0]!=14:
                #    continue
                # 计算 FPS
                #cur_time = time.time()
                #fps = 1 / (cur_time - prev_time)
                #prev_time = cur_time
                #print(f"当前 FPS: {fps:.2f}")

                # 画出检测结果
                #dstimg = yolov5_detector.draw_detections(srcimg, boxes, scores, class_ids)
                
                img_1 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                #if boxes is not None:
                
                #print (classes)
                cind=0
                if classes is not None:
                    if 14 in classes:
                        for item in classes:
                            if item ==14:
                                #self.draw(img_1, boxes, scores, classes)
                                self.qdata.put(boxes[cind],block=False)
                            cind=cind+1
                        #print("1234567898888888888888888")

                # show output
                #cv2.imwrite("out.jpg", img_1)
                #cv2.waitKey(1)
                #cv2.imshow("post process result", img_1)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()

                # # 推送到 RTMP 服务器
                # try:
                #     pusher.streamPush(dstimg)
                # except Exception as e:
                #     print(f"推流失败: {e}")

                #if cv2.waitKey(1) & 0xFF == ord('q'):
                 #   break
                #cv2.waitKey(2)
                time.sleep(2)

        except Exception as e:
            print(f"程序异常: {e}")
        finally:
            if self.cap:
                #self.cap.release()
                self.stop()
            #if pusher:
            #    pusher.close()
            cv2.destroyAllWindows()
            rknn_lite.release()

if __name__ == '__main__':
    q=Queue(maxsize=10)
    birdview= BIRDVIEW(q)
    birdview.getbird()
