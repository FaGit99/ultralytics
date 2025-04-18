import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Load the YOLO model
model = YOLO("yolo11x-pose.pt")

# Open the video file or webcam
cap = cv2.VideoCapture(0)  # Use 0 for webcam

# Function to check for 'q' key press using matplotlib
def check_for_exit():
    plt.pause(0.001)
    if plt.get_fignums():
        fig = plt.gcf()
        if fig.canvas.manager.key_press_handler_id:
            if fig.canvas.manager.key_press_handler_id == ord('q'):
                return True
    return False

# Create a figure and axis for displaying the frames
fig, ax = plt.subplots()

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    if success:
        # Run YOLO inference on the frame
        results = model(frame)
        
        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        
        # Convert BGR to RGB for matplotlib
        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        
        # Display the annotated frame using matplotlib
        ax.imshow(annotated_frame_rgb)
        ax.axis('off')
        plt.draw()
        plt.pause(0.001)
        
        # Check for 'q' key press to exit
        if check_for_exit():
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
plt.close()