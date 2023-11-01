from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch

# Use the model
results = model.train(data="config.yaml",epochs=100)  # t  rain the model

# import yaml
#
# file_name = 'config.yaml'
# with open(file_name,"r") as s:
#     names= yaml.safe_load(s)["names"]
# print(names)