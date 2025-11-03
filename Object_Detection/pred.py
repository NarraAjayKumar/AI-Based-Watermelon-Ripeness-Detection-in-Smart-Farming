import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import os

# 🔧 Config
model_path = r"C:\MurthyLab\Project\Project\my_yolo_model2\weights\best.pt"
image_path = r"C:\MurthyLab\Object_Detection\dataset\images\a51e6744-IMG_20240603_151225.jpg"  # Your test image
output_path = r"C:\Users\narra\OneDrive\Desktop\Images\predicted_output.jpg" # Always saves here

# 🧠 Load model
model = YOLO(r"C:\MurthyLab\Object_Detection\runs\detect\my_yolo_model\weights\best.pt")


# 🎯 Run prediction
results = model(image_path, conf=0.25)

# 🖌️ Plot annotated image
output_image = results[0].plot()

# 💾 Ensure output folder exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# 💾 Save result
cv2.imwrite(output_path, output_image)
print(f"✅ Output saved to: {output_path}")

# 🖼️ Show image
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('YOLOv8 Prediction')
plt.show()