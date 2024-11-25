import torch
from PIL import Image
import os


folder_path = r"Images"
print_path = r"Identified Images"
itter = 0
for item in os.listdir(folder_path):

    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    # Images
    item_path = os.path.join(folder_path, item)
    # img = "Images\DeathStar3.jpeg"  # or file, Path, PIL, OpenCV, numpy, list

    # Inference
    model.conf = 0.2
    results = model(item_path, size=320) #old was 320

    # Results
    # results.show()  # or .show(), .save(), .crop(), .pandas(), etc.

    # Load the original image
    ori_img = Image.open(item_path)
    for i, (*box, conf, cls) in enumerate(results.xyxy[0]):  # xyxy format
        x1, y1, x2, y2 = map(int, box[:4])  # Extract bounding box
        cropped_img = ori_img.crop((x1, y1, x2, y2))  # Crop region
        cropped_img.convert("RGB").save(f"{item[:-4]}_cropped_{i}.png")  # Save the cropped image
    itter += 1

print ("\nFINISHED\n")