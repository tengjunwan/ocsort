from ultralytics import YOLO

model = YOLO("pretrained/yolov8n_visdrone_2.pt")
print("number of classes:",  model.model.model[-1].nc)
model.export(format="onnx", opset=11, imgsz=(640, 640), dynamic=False, simplify=False)
