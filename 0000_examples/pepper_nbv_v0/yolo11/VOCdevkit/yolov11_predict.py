from ultralytics import YOLO
# 加载训练好的模型，改为自己的路径
model = YOLO(r'F:\WRS\wrs-virtual\0000_examples\pepper\yolo11\runs\detect\train3\weights\best.pt')  #修改为训练好的路径
source = r'F:\WRS\wrs-virtual\0000_examples\pepper\yolo11\VOCdevkit\test\images' #修改为自己的图片路径及文件名
# 运行推理，并附加参数
model.predict(source, save=True)