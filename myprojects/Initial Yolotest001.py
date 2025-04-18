from ultralytics import YOLO

import torch
import torchvision
print(torch.cuda.is_available())  # Should return True
print(torchvision.__version__)  # Should match the CUDA version

# Load a pretrained YOLO11n model
if __name__ == '__main__':
    # model = YOLO("yolov8n-seg.pt")  
    model = YOLO("yolo11n.pt")    
    model.train(data="config.yaml", epochs=1, imgsz=640)



# Train the model on the COCO8 dataset for 100 epochs
train_results = model.train(
    data="coco8.yaml",  # Path to dataset configuration file
    epochs=100,  # Number of training epochs
    imgsz=640,  # Image size for training
    device="0",  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
)

# Evaluate the model's performance on the validation set
metrics = model.val()

# Perform object detection on an image
results = model("path/to/image.jpg")  # Predict on an image
results[0].show()  # Display results

# Export the model to ONNX format for deployment
path = model.export(format="onnx")  # Returns the path to the exported model