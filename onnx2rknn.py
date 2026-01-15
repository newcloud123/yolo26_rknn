import os
import urllib
import traceback
import time
import sys
import numpy as np
import cv2
from rknn.api import RKNN
from math import exp

ONNX_MODEL = './lsy.onnx'
RKNN_MODEL = './yolo26s.float.rknn'
DATASET = './datasets.txt'

QUANTIZE_ON = False

CLASSES = ['ballon', 'drone']

meshgrid = []

class_num = len(CLASSES)
headNum = 3
strides = [8, 16, 32]
mapSize = [[80, 80], [40, 40], [20, 20]]
nmsThresh = 0.4
objectThresh = 0.5

input_imgH = 640
input_imgW = 640


class DetectBox:
    def __init__(self, classId, score, xmin, ymin, xmax, ymax):
        self.classId = classId
        self.score = score
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

def GenerateMeshgrid():
    for index in range(headNum):
        for i in range(mapSize[index][0]):
            for j in range(mapSize[index][1]):
                meshgrid.append(j + 0.5)
                meshgrid.append(i + 0.5)


def IOU(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2):
    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)

    innerWidth = xmax - xmin
    innerHeight = ymax - ymin

    innerWidth = innerWidth if innerWidth > 0 else 0
    innerHeight = innerHeight if innerHeight > 0 else 0

    innerArea = innerWidth * innerHeight

    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    total = area1 + area2 - innerArea

    return innerArea / total


def handleResult(detectResult):
    predBoxs = []

    sort_detectboxs = sorted(detectResult, key=lambda x: x.score, reverse=True)
    for i in range(len(sort_detectboxs)):
        xmin1 = sort_detectboxs[i].xmin
        ymin1 = sort_detectboxs[i].ymin
        xmax1 = sort_detectboxs[i].xmax
        ymax1 = sort_detectboxs[i].ymax
        classId = sort_detectboxs[i].classId
        predBoxs.append(sort_detectboxs[i])
    return predBoxs


def sigmoid(x):
    return 1 / (1 + exp(-x))


def postprocess(out, img_h, img_w):
    print('postprocess ... ')
    left, top, right, bottom = padding

    detectResult = []
    output = []
    for i in range(len(out)):
        print(out[i].shape)
        output.append(out[i].reshape((-1)))

   

    gridIndex = -2
    cls_index = 0
    cls_max = 0

    for index in range(headNum):
        reg = output[index * 2 + 0]
        cls = output[index * 2 + 1]

        for h in range(mapSize[index][0]):
            for w in range(mapSize[index][1]):
                gridIndex += 2

                if 1 == class_num:
                    cls_max = sigmoid(cls[0 * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w])
                    cls_index = 0
                else:
                    for cl in range(class_num):
                        cls_val = cls[cl * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w]
                        if 0 == cl:
                            cls_max = cls_val
                            cls_index = cl
                        else:
                            if cls_val > cls_max:
                                cls_max = cls_val
                                cls_index = cl
                    cls_max = sigmoid(cls_max)

                if cls_max > objectThresh:
       
                    x1 = (meshgrid[gridIndex + 0] - reg[0*mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w]) * strides[index]
                    y1 = (meshgrid[gridIndex + 1] - reg[1*mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w]) * strides[index]
                    x2 = (meshgrid[gridIndex + 0] + reg[2*mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w]) * strides[index]
                    y2 = (meshgrid[gridIndex + 1] + reg[3*mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w]) * strides[index]



                    print(index)
                    print('x1,y1,x2,y2:', x1, y1, x2, y2)
                    

                    xmin = (x1 - left) / ratio[0]
                    ymin = (y1 - top) / ratio[1]
                    xmax = (x2 - left) / ratio[0]
                    ymax = (y2 - top) / ratio[1]

                    xmin = xmin if xmin > 0 else 0
                    ymin = ymin if ymin > 0 else 0
                    xmax = xmax if xmax < img_w else img_w
                    ymax = ymax if ymax < img_h else img_h

                    box = DetectBox(cls_index, cls_max, xmin, ymin, xmax, ymax)
                    detectResult.append(box)
    # NMS
    print('detectResult:', len(detectResult))
    predBox = handleResult(detectResult)
    

    return predBox


def export_rknn_inference(img):
    # Create RKNN object
    rknn = RKNN(verbose=False)

    # pre-process config
    print('--> Config model')
    rknn.config(mean_values=[[0,0,0]], std_values=[[255,255,255]], quantized_algorithm='normal', quantized_method='channel', target_platform='rk3588')
    print('done')

    # Load ONNX model
    print('--> Loading model')
    # ret = rknn.load_onnx(model=ONNX_MODEL, outputs=[ 'reg1','cls1', 'reg2', 'cls2', 'reg3', 'cls3'])
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=QUANTIZE_ON, dataset=DATASET, rknn_batch_size=1)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export RKNN model
    print('--> Export rknn model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    # ret = rknn.init_runtime(target='rk3566')
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    # Inference
    print('--> Running model')
    outputs = rknn.inference(inputs=[img])
    rknn.release()
    print('done')

    return outputs
def letterbox(
    img: np.ndarray,
    new_shape: tuple = (640, 640),
    color: tuple = (114, 114, 114),
    auto: bool = True,
    scaleFill: bool = False,
    scaleup: bool = True
) -> tuple:
    """
    YOLO 图像预处理的 Letterbox 函数（保持长宽比缩放并填充图像）
    :param img: 输入原始图像（OpenCV 读取的 np.ndarray 格式，形状为 (h, w, c)）
    :param new_shape: YOLO 模型要求的目标输入尺寸，默认 (640, 640)
    :param color: 填充边框的颜色，默认黑色 (0, 0, 0)
    :param auto: 自动调整填充，保持缩放后的图像为目标尺寸的整数倍（适配 YOLO 步长要求）
    :param scaleFill: 是否直接拉伸图像至目标尺寸（不保持长宽比，不推荐）
    :param scaleup: 是否允许放大图像（若为 False，仅缩小图像，避免放大引入噪声）
    :return: 处理后的图像、图像缩放比例、上下左右填充尺寸
    """
    # 1. 获取原始图像尺寸和目标尺寸
    shape = img.shape[:2]  # 原始图像 (高, 宽)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    
    # 2. 计算图像缩放比例（保持长宽比）
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # 若禁止放大，限制最大缩放比例为 1（仅缩小）
        r = min(r, 1.0)
    
    # 3. 计算缩放后的图像尺寸
    ratio = r, r  # 宽、高的缩放比例（保持一致，避免畸变）
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # 缩放后的 (宽, 高)
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # 计算需要填充的宽度和高度
    
    # 4. 处理填充尺寸（适配 YOLO 步长，默认 32 倍）
    if auto:  # 自动调整填充，使填充后的尺寸为 32 的整数倍（YOLO 网络步长要求）
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)
    elif scaleFill:  # 直接拉伸图像至目标尺寸（不保持长宽比，不推荐用于检测）
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]
    
    # 5. 分割填充尺寸（上下左右对称填充，避免图像偏移）
    dw /= 2  # 左右填充各占一半
    dh /= 2
    
    # 6. 对原始图像进行缩放
    if shape[::-1] != new_unpad:  # 仅当尺寸不一致时进行缩放
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    # 7. 进行图像填充（cv2.copyMakeBorder 实现边缘填充）
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=color
    )
    print("ratio[0] = {} | ratio[1] = {} | left ={} | top={} | right={} |bottom={}".format(ratio[0],ratio[1],left, top, right, bottom))
    # 8. 返回处理结果（处理后的图像、缩放比例、填充信息）
    return img, ratio, (left, top, right, bottom)

if __name__ == '__main__':
    print('This is main ...')
    GenerateMeshgrid()

    img_path = 'lsy151.jpg'
    orig_img = cv2.imread(img_path)
    img_h, img_w = orig_img.shape[:2]
    
    
    origimg = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    origimg,ratio, padding= letterbox(
        origimg,
        new_shape=(640, 640),
        color=(114, 114, 114),
        auto=False
    )
    #left, top, right, bottom = padding
    print("oriimg shape = ",origimg.shape)
    
    img = np.expand_dims(origimg, 0)

    outputs = export_rknn_inference(img)

    out = []
    for i in range(len(outputs)):
        out.append(outputs[i])

    predbox = postprocess(out, img_h, img_w,padding,ratio)


    print(len(predbox))

    for i in range(len(predbox)):
        xmin = int(predbox[i].xmin)
        ymin = int(predbox[i].ymin)
        xmax = int(predbox[i].xmax)
        ymax = int(predbox[i].ymax)
        classId = predbox[i].classId
        score = predbox[i].score

        cv2.rectangle(orig_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        ptext = (xmin, ymin)
        title = CLASSES[classId] + ":%.2f" % (score)
        cv2.putText(orig_img, title, ptext, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imwrite('./test_rknn_result.jpg', orig_img)
