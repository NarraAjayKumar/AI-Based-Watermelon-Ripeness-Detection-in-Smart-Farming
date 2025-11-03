import cv2
import os

input_folder = r"C:\MurthyLab\project_seg\data\Water melon"
output_folder = r"C:\MurthyLab\project_seg\data\flipped"

os.makedirs(output_folder, exist_ok=True)

angles = [45, 90, 180, 270]

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, matrix, (w, h))
    return rotated

for file in os.listdir(input_folder):
    if file.endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(input_folder, file)
        image = cv2.imread(img_path)

        for angle in angles:
            rotated = rotate_image(image, angle)
            save_name = f"{os.path.splitext(file)[0]}_rot{angle}.jpg"
            save_path = os.path.join(output_folder, save_name)
            cv2.imwrite(save_path, rotated)

        # Flip Horizontally
        h_flip = cv2.flip(image, 1)
        cv2.imwrite(os.path.join(output_folder, f"{os.path.splitext(file)[0]}_hflip.jpg"), h_flip)

        # Flip Vertically
        v_flip = cv2.flip(image, 0)
        cv2.imwrite(os.path.join(output_folder, f"{os.path.splitext(file)[0]}_vflip.jpg"), v_flip)

print("✅ Flipped & rotated images saved successfully.")
