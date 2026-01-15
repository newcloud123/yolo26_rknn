from ultralytics import YOLO
 
# 假设你已经训练好的模型保存在 yolov8m-cls.pt 文件中
model = YOLO("/data/luoshiyong/code/yolo26/runs/detect/train3/weights/best.pt")
 # /data/luoshiyong/code/ultralytics-main/runs/detect/realballoonall
# 导出为 ONNX 格式
model.export(format='onnx',dynamic=False,imgsz=640,opset = 12)

