'''
多线程显示目标检测的检测框，
在最大算力情况下保证画面不卡顿

这里以训练的gaussian yolo v3为例
'''

import os
import numpy
import cv2
import threading
import time
from queue import Queue

from ctypes import *
import math
import random

import matplotlib.cm as mpcm
import numpy as np
from PIL import Image,ImageDraw,ImageFont

my_label = {"A":"barrett食管","B":"反流性食管炎","C":"结肠息肉","D":"结肠早癌","E":"结肠进展期癌","F":"早期胃癌",
    "G":"胃溃疡","H":"进展期胃癌","I":"慢性萎缩性胃炎","J":"食管早癌","K":"食管静脉曲张","L":"气泡","M":"反光"}

label_name = ["A","B","C","D","E","F","G","H","I","J","K","L","M"]

class2id = {'barrett食管':"A","barrett 食管":"A","反流性食管炎":"B","结肠息肉":"C","结直肠息肉":"C","结直肠腺瘤性息肉":"C","结直肠非腺瘤性息肉":"C",
    "结肠早癌":"D","早期结直肠癌":"D","早期结直肠癌_0-IIa型":"D","早期结直肠癌_0-I型":"D","早期结直肠癌_0-IIa+c型":"D","结肠进展期癌":"E",
    "胃早癌": "F", "早期胃癌":"F","早期胃癌_0-IIa+c型":"F","早期胃癌_0-IIa型":"F","早期胃癌_0-IIb型":"F","早期胃癌_0-IIc型":"F","早期胃癌_0-IIc+a型":"F","早期胃癌_0-I型":"F",
    "胃溃疡": "G","胃良性溃疡":"G","良性胃溃疡":"G", "胃恶性溃疡":"G","恶性胃溃疡":"G","进展期胃癌":"H",
    "慢性萎缩性胃炎":"I","食管早癌":"J","食管静脉曲张":"K","气泡":"L","反光":"M","强光":"M"
    }


#------调用darknet----------
def change_cv2_draw(image,strs,local,sizes,colour):
    cv2img = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    pilimg = Image.fromarray(cv2img)
    draw = ImageDraw.Draw(pilimg)
    font = ImageFont.truetype("./font_lib/Microsoft-Yahei-UI-Light.ttc",sizes,encoding='utf-8')
    draw.text(local,strs,colour,font=font)
    image = cv2.cvtColor(np.array(pilimg),cv2.COLOR_RGB2BGR)

    return image

def colors_subselect(colors, num_classes=13):
    dt = len(colors) // num_classes
    sub_colors = []
    for i in range(num_classes):
        color = colors[i*dt]
        if isinstance(color[0], float):
            sub_colors.append([int(c * 255) for c in color])
        else:
            sub_colors.append([c for c in color])
    return sub_colors
colors = colors_subselect(mpcm.plasma.colors, num_classes=13)
colors_tableau = [(255, 152, 150),(148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int),
                ("uc", POINTER(c_float))]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]



#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
#lib = CDLL("libdarknet.so", RTLD_GLOBAL)
hasGPU = True
if os.name == "nt":
    cwd = os.path.dirname(__file__)
    os.environ['PATH'] = cwd + ';' + os.environ['PATH']
    winGPUdll = os.path.join(cwd, "yolo_cpp_dll.dll")
    winNoGPUdll = os.path.join(cwd, "yolo_cpp_dll_nogpu.dll")
    envKeys = list()
    for k, v in os.environ.items():
        envKeys.append(k)
    try:
        try:
            tmp = os.environ["FORCE_CPU"].lower()
            if tmp in ["1", "true", "yes", "on"]:
                raise ValueError("ForceCPU")
            else:
                print("Flag value '"+tmp+"' not forcing CPU mode")
        except KeyError:
            # We never set the flag
            if 'CUDA_VISIBLE_DEVICES' in envKeys:
                if int(os.environ['CUDA_VISIBLE_DEVICES']) < 0:
                    raise ValueError("ForceCPU")
            try:
                global DARKNET_FORCE_CPU
                if DARKNET_FORCE_CPU:
                    raise ValueError("ForceCPU")
            except NameError:
                pass
            # print(os.environ.keys())
            # print("FORCE_CPU flag undefined, proceeding with GPU")
        if not os.path.exists(winGPUdll):
            raise ValueError("NoDLL")
        lib = CDLL(winGPUdll, RTLD_GLOBAL)
    except (KeyError, ValueError):
        hasGPU = False
        if os.path.exists(winNoGPUdll):
            lib = CDLL(winNoGPUdll, RTLD_GLOBAL)
            print("Notice: CPU-only mode")
        else:
            # Try the other way, in case no_gpu was
            # compile but not renamed
            lib = CDLL(winGPUdll, RTLD_GLOBAL)
            print("Environment variables indicated a CPU run, but we didn't find `"+winNoGPUdll+"`. Trying a GPU run anyway.")
else:
    lib = CDLL("./libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

copy_image_from_bytes = lib.copy_image_from_bytes
copy_image_from_bytes.argtypes = [IMAGE,c_char_p]

def network_width(net):
    return lib.network_width(net)

def network_height(net):
    return lib.network_height(net)

predict = lib.network_predict_ptr
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

if hasGPU:
    set_gpu = lib.cuda_set_device
    set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int), c_int]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict_ptr
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

load_net_custom = lib.load_network_custom
load_net_custom.argtypes = [c_char_p, c_char_p, c_int, c_int]
load_net_custom.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

predict_image_letterbox = lib.network_predict_image_letterbox
predict_image_letterbox.argtypes = [c_void_p, IMAGE]
predict_image_letterbox.restype = POINTER(c_float)

def array_to_image(arr):
    import numpy as np
    # need to return old values to avoid python freeing memory
    arr = arr.transpose(2,0,1)
    c = arr.shape[0]
    h = arr.shape[1]
    w = arr.shape[2]
    arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
    data = arr.ctypes.data_as(POINTER(c_float))
    im = IMAGE(w,h,c,data)
    return im, arr

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        if altNames is None:
            nameTag = meta.names[i]
        else:
            nameTag = altNames[i]
        res.append((nameTag, out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45, debug= False):
    """
    Performs the meat of the detection
    """
    #pylint: disable= C0321
    im = load_image(image, 0, 0)
    if debug: print("Loaded image")
    ret = detect_image(net, meta, im, thresh, hier_thresh, nms, debug)
    free_image(im)
    if debug: print("freed image")
    return ret

def detect_image(net, meta, im, thresh=.5, hier_thresh=.5, nms=.45, debug= False):
    num = c_int(0)
    if debug: print("Assigned num")
    pnum = pointer(num)
    if debug: print("Assigned pnum")
    predict_image(net, im)
    letter_box = 0
    #predict_image_letterbox(net, im)
    #letter_box = 1
    if debug: print("did prediction")
    #dets = get_network_boxes(net, custom_image_bgr.shape[1], custom_image_bgr.shape[0], thresh, hier_thresh, None, 0, pnum, letter_box) # OpenCV
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum, letter_box)
    if debug: print("Got dets")
    num = pnum[0]
    if debug: print("got zeroth index of pnum")
    if nms:
        do_nms_sort(dets, num, meta.classes, nms)
    if debug: print("did sort")
    res = []
    if debug: print("about to range")
    for j in range(num):
        if debug: print("Ranging on "+str(j)+" of "+str(num))
        if debug: print("Classes: "+str(meta), meta.classes, meta.names)
        for i in range(meta.classes):
            if debug: print("Class-ranging on "+str(i)+" of "+str(meta.classes)+"= "+str(dets[j].prob[i]))
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                if altNames is None:
                    nameTag = meta.names[i]
                else:
                    nameTag = altNames[i]
                if debug:
                    print("Got bbox", b)
                    print(nameTag)
                    print(dets[j].prob[i])
                    print((b.x, b.y, b.w, b.h))
                res.append((nameTag, dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    if debug: print("did range")
    res = sorted(res, key=lambda x: -x[1])
    if debug: print("did sort")
    free_detections(dets, num)
    if debug: print("freed detections")
    return res


netMain = None
metaMain = None
altNames = None

def performDetect(frame_id=0, imagePath="temp.jpg", thresh=0.25, configPath="./data/Gaussian_yolov3_myData.cfg", weightPath="./data/Gaussian_yolov3_myData.weights", metaPath="./data/myData.data", showImage=True, makeImageOnly=False, initOnly=False):

    global metaMain, netMain, altNames #pylint: disable=W0603
    assert 0 < thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `"+os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `"+os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `"+os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = load_net_custom(configPath.encode("ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = load_meta(metaPath.encode("ascii"))
    if altNames is None:
        # In Python 3, the metafile default access craps out on Windows (but not Linux)
        # Read the names file and create a list to feed to detect
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents, re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    if initOnly:
        print("Initialized detector")
        return None
    if not os.path.exists(imagePath):
        raise ValueError("Invalid image path `"+os.path.abspath(imagePath)+"`")
    # Do the detection
    #detections = detect(netMain, metaMain, imagePath, thresh)  # if is used cv2.imread(image)
    detections = detect(netMain, metaMain, imagePath.encode("ascii"), thresh)

    if showImage:
        try:
            scale = 0.4
            text_thickness = 1
            line_type = 8
            thickness=2

            # image = cv2.imread(imagePath)
            print("*** "+str(len(detections))+" Results, color coded by confidence ***")
            detections = {
                "detections": detections,
                "frame_id":frame_id
            }
        except Exception as e:
            print("Unable to show image: "+str(e))
    return detections
#--------------------------------------------------------

#-------------加载视频源，多线程显示和识别------------------

# cap=cv2.VideoCapture("http://name:key@ip:port") #ip视频
# cap = cv2.VideoCapture(0)   #采集卡
cap = cv2.VideoCapture("test.mp4")  #本地视频
# cap.set(5,30)
print(cap.get(5))

detection_queue = Queue(1)
result_queue = Queue()
threadLock1 = threading.Lock()

class MyThread(threading.Thread):
    def __init__(self,th_name):
        threading.Thread.__init__(self)
        self.th_name = th_name

    def run(self):
        while True:
            threadLock1.acquire()  # 加锁
            if not detection_queue.empty():
                frame_info = detection_queue.get(1)

                frame_id = list(frame_info.keys())[0]
                frame_path = frame_info[frame_id]
                detect_res = performDetect(imagePath=frame_path)
                
                result_queue.put(detect_res)
                os.remove(frame_path)
                threadLock1.release()  # 释放线程锁

            else:
                threadLock1.release()
                print("队列为空")

    def get_result(self):
        return result_queue

# ------------开启检测线程-----------------------
t1 = MyThread("detection线程")
t1.start()

#------------开启主线程---------------------------
i = 0
delay_i = 0
detect_result = None
while True:

    ret,frame = cap.read()
    i += 1

    if i == 1:
        print("写入队列")
        save_path = "./temp_"+str(i)+".jpg"
        cv2.imwrite(save_path,frame)
        detection_queue.put({i:save_path})
    if detection_queue.empty() and not result_queue.empty():
        print("写入队列")
        save_path = "./temp_"+str(i)+".jpg"
        cv2.imwrite(save_path,frame)
        detection_queue.put({i:save_path})
    if not result_queue.empty():
        print("result队列")
        delay_i = 0
        detect_result = result_queue.get(1)

    delay_i += 1 
    if not detect_result is None and delay_i <= 60: 

        detections = detect_result["detections"]
        frame_id_ = detect_result['frame_id']

        for detection in detections:
            label = my_label[detection[0]]
            confidence = detection[1]
            pstring = label+": "+str(np.rint(100 * confidence))+"%"
            bounds = detection[2]
            shape = frame.shape

            yExtent = int(bounds[3])
            xEntent = int(bounds[2])
            # Coordinates are around the center
            xCoord = int(bounds[0] - bounds[2]/2)
            yCoord = int(bounds[1] - bounds[3]/2)

            color = colors_tableau[label_name.index(detection[0])]
            p1 = (xCoord, yCoord)
            p2 = (xCoord + xEntent,yCoord + yExtent)
            if (p2[0] - p1[0] < 1) or (p2[1] - p1[1] < 1):
                continue

            cv2.rectangle(frame, p1, p2, color, thickness)

            text_size, baseline = cv2.getTextSize(pstring, cv2.FONT_HERSHEY_SIMPLEX, scale, text_thickness)
            cv2.rectangle(frame, (p1[0], p1[1] - thickness*10 - baseline), (p1[0] + 2*(text_size[0]-20), p1[1]), color, -1)
            frame = change_cv2_draw(frame,pstring,(p1[0],p1[1]-7*baseline),20,(255,255,255))
            cv2.putText(frame, "Gaussion YOLO V3 | DataXujing | Frame:{}".format(frame_id_), (40,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, line_type)
        
    frame = cv2.resize(frame,(int(1920/2),int(1080/2)))
    cv2.imshow('Gaussian_YOLO_V3', frame)

    k = cv2.waitKey(1)& 0xFF
    if k == 27:         # wait for ESC key to exit
        cap.release()
        cv2.destroyAllWindows()

cap.release()
cv.destroyAllWindows()

# 即主线程任务结束之后，进入阻塞状态，一直等待其他的子线程执行结束之后，主线程在终止
t1.join()
# 设置子线程为守护线程时，主线程一旦执行结束，则全部线程全部被终止执行，
# 可能出现的情况就是，子线程的任务还没有完全执行结束，就被迫停止
# t1.setDaemon(True)