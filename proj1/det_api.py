import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# STEP 2: Create an ObjectDetector object.
base_options = python.BaseOptions(model_asset_path="models\\efficientdet_lite0.tflite")
options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.5)
detector = vision.ObjectDetector.create_from_options(options)

from fastapi import FastAPI, UploadFile

app = FastAPI()

import cv2
import numpy as np


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):

    # STEP 3: Load the input image.
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    cv_mat = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv_mat)

    detection_result = detector.detect(image)
    # print(detection_result)

    # STEP 5: Process the detection result. In this case, visualize it.
    total_count = len(detection_result.detections)
    person_count = 0
    for detection in detection_result.detections:
        if detection.categories[0].category_name == "person":
            person_count += 1

    result = {"total_count": total_count, "person_count": person_count}
    return {"result": result}
