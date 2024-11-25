import torch
from PIL import Image
import os
import numpy as np
from sklearn.neighbors import NearestNeighbors

folder_path = r"Images"
itter = 0
for item in os.listdir(folder_path):

    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    # Images
    item_path = os.path.join(folder_path, item)


    # Inference
    model.conf = 0.2
    results = model(item_path, size=320) #old was 320

    boxes = results.xyxy[0].cpu().numpy()  # Get bounding boxes [x1, y1, x2, y2, confidence, class_id]
    centers = [[1,1]]
    knn = NearestNeighbors(n_neighbors=3, algorithm='auto', metric='euclidean')
    knn.fit(centers)

    # Results
    results.show()  # or .show(), .save(), .crop(), .pandas(), etc.

    for i, box in enumerate(boxes):
            distances, indices = knn.kneighbors([centers[i]])
            print(f"Bounding box {i}: Neighbors -> {indices[0]}, Distances -> {distances[0]}")


    # Load the original image
    ori_img = Image.open(item_path)
    for i, (*box, conf, cls) in enumerate(results.xyxy[0]):  # xyxy format
        x1, y1, x2, y2 = map(int, box[:4])  # Extract bounding box
        cropped_img = ori_img.crop((x1, y1, x2, y2))  # Crop region
        cropped_img.convert("RGB").save(f"{item[:-4]}_cropped_{i}.png")  # Save the cropped image
    itter += 1

print ("\nFINISHED\n")