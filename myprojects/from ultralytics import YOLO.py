from ultralytics import YOLO

# Load a model
model = YOLO(r"C:\Users\Fabian\Documents\GitHub\ultralytics\yolo11x-pose.pt")  # load an official model

# Run inference on a video
results = model(r"C:\Users\Fabian\Documents\GitHub\ultralytics\python_pose\Fab IMG_5989.MOV",show = True, save = True)  # predict on video