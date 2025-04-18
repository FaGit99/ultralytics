import torch
import cv2
import math
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Function to calculate angle between two vectors
def calculate_angle(v1, v2):
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    magnitudes = math.sqrt(v1[0]**2 + v1[1]**2) * math.sqrt(v2[0]**2 + v2[1]**2)
    return math.degrees(math.acos(dot_product / magnitudes))

# Load the YOLO model
model = YOLO('yolo11x-pose.pt')

# Run inference with stream=True
results = model.predict(
    task="pose",
    source="YoutubeTest.mp4",  # webcam source = 0
    show_labels=False,
    show_conf=False,
    stream=True
)

# Keypoint descriptions
keypoint_descriptions = [
    "Nase", "Linkes Auge", "Rechtes Auge", "Linkes Ohr", "Rechtes Ohr",
    "Linke Schulter", "Rechte Schulter", "Linker Ellenbogen", "Rechter Ellenbogen",
    "Linkes Handgelenk", "Rechtes Handgelenk", "Linke Hüfte", "Rechte Hüfte",
    "Linkes Knie", "Rechtes Knie", "Linker Knöchel", "Rechter Knöchel"
]

# Interesting angles table
notso_interesting_angles = [
    {"ref": 1, "designation": "Left Nose angle", "keypoints": [1, 2, 4]},
    {"ref": 2, "designation": "Right Nose angle", "keypoints": [1, 3, 5]},
    {"ref": 3, "designation": "Left shoulder angle", "keypoints": [2, 4, 6]},
    {"ref": 4, "designation": "Right shoulder angle", "keypoints": [3, 5, 7]}
]

# Interesting angles table
interesting_angles = [
    {"ref": 1, "designation": "Left leg hip angle", "keypoints": [6, 12, 14]},
    {"ref": 2, "designation": "Right leg hip  angle", "keypoints": [7, 13, 15]},
    {"ref": 3, "designation": "Left leg knee angle", "keypoints": [13, 15, 17]},
    {"ref": 4, "designation": "Right leg knee  angle", "keypoints": [12, 14, 16]},
    {"ref": 5, "designation": "Left ellbow angle", "keypoints": [6, 8, 10]}, # reihenfolege 10/6 getauscht
    {"ref": 6, "designation": "Right elbow angle", "keypoints": [7, 9, 11]},
    {"ref": 7, "designation": "Left head anglee", "keypoints": [4, 6, 12]}
 
]


# Array to store angles for 200 frames
angle_evolution = []

# Process each frame
frame_count = 0
for result in results:
    if frame_count >= 600:
        break
    
    keypoints = result.keypoints.data  # Get pose keypoints
    
    # Store keypoint positions with descriptions
    frame_keypoints = []
    for i, pose in enumerate(keypoints):
        pose_keypoints = []
        for j, keypoint in enumerate(pose):
            description = keypoint_descriptions[j]
            coordinates = keypoint.cpu().numpy()
            pose_keypoints.append({"number": j+1, "name": description, "coordinates": coordinates})
        frame_keypoints.append(pose_keypoints)
    
    img = result.plot()  # Get the annotated frame
    
    # Calculate angles between defined keypoints
    frame_angles = []
    for pose in keypoints:
        for angle_def in interesting_angles:
            kp_indices = angle_def["keypoints"]
            p1, p2, p3 = pose[kp_indices[0]-1], pose[kp_indices[1]-1], pose[kp_indices[2]-1]
            
            # Create vectors
            v1 = [p2[0] - p1[0], p2[1] - p1[1]]
            v2 = [p3[0] - p2[0], p3[1] - p2[1]]
            
            # Calculate angle
            angle = calculate_angle(v1, v2)
            
            # Store angle information
            frame_angles.append({"ref": angle_def["ref"], "designation": angle_def["designation"], "angle": angle})
            
            # Display angle and designation on image
            cv2.putText(img, 
                       f"{angle_def['designation']} {angle:.1f}°", 
                       (int(p2[0]), int(p2[1])), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, 
                       (255,255,255), 
                       1,
                       cv2.LINE_AA)
    
    # Save the frame
    cv2.imwrite(f'pose_angles_{frame_count}.jpg', img)
    
    # Store angles for the current frame
    angle_evolution.append({"frame_number": frame_count + 1, "angles": frame_angles})
    
# Print keypoint positions and interesting angles for debugging
#   for frame_data in angle_evolution:
#       print(f"Frame {frame_data['frame_number']}:")
#       for angle in frame_data["angles"]:
#           print(f"  {angle['ref']:2d}. {angle['designation']:<20}: {angle['angle']:.1f}°")

    frame_count += 1



# Visualize the angle evolution
def plot_angle_evolution(angle_evolution, interesting_angles):
    plt.figure(figsize=(12, 8))
    
    for angle_def in interesting_angles:
        ref = angle_def["ref"]
        designation = angle_def["designation"]
        
        # Extract angles for the current designation
        angles = [frame["angles"][ref - 1]["angle"] for frame in angle_evolution]
        frames = [frame["frame_number"] for frame in angle_evolution]
        
        plt.plot(frames, angles, label=designation)
    
    plt.xlabel('Frame Number')
    plt.ylabel('Angle (degrees)')
    plt.title('Evolution of Interesting Angles Over 200 Frames')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_angle_evolution(angle_evolution, interesting_angles)
