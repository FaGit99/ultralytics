from ultralytics import YOLO

import torch
import torchvision
print(torch.cuda.is_available())  # Should return True
print(torchvision.__version__)  # Should match the CUDA version
learning = 0

# only for directory name finding
from types import SimpleNamespace
from ultralytics.cfg import get_save_dir
args = SimpleNamespace(project="my_project", task="detect", mode="train", exist_ok=True, name=None)
save_dir = get_save_dir(args)
print("Savedir for yolo", save_dir)

# only for directory name finding

# Load a pretrained YOLO11n model
if __name__ == '__main__' and learning == 1:
    print(" in the loop")
    print(" in the loop")
    print(" in the loop")
    print(" in the loop")
    print(" in the loop")
    # model = YOLO("yolov8n-seg.pt")  
    model = YOLO("yolo11n.pt")    
    # model.train(data="config.yaml", epochs=1, imgsz=640)
    model.train(data="coco8.yaml", epochs=1, imgsz=640)



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
    results = model(r"C:\Users\Fabian\Documents\GitHub\ultralytics\img01.jpg")  # Predict on an image
    results[0].show()  # Display results

# Export the model to ONNX format for deployment
    path = model.export(format="onnx")  # Returns the path to the exported model
else:
    model = YOLO(r"C:\Users\Fabian\Documents\GitHub\ultralytics\runs\detect\train162\weights\best.pt")    
    impfilename= r"C:\Users\Fabian\Documents\GitHub\ultralytics\img07"   
    results = model(impfilename + ".jpg")  # Predict on an image
    results[0].show()  # Display results
    results[0].save(filename=impfilename + "_i.jpg")
