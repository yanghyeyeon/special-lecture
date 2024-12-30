#  STEP 1 : import modules
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

# STEP 2 : create inference object(instance)
app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(640,640))

# STEP 3 : load data
img1 = cv2.imread("news-p.v1.20241018.5beb2099e19b4dccb69a3a67548f71e0_P1.png")
img2 = cv2.imread("53019236.4.jpg")

# STEP 4 : inference
faces1 = app.get(img1)
faces2 = app.get(img2)
assert len(faces1)==1
assert len(faces2)==1

# STEP 5 : post processing
# STEP 5-1 : Save result image
# rimg = app.draw_on(img, faces)
# cv2.imwrite("./t1_output.jpg", rimg)

# STEP 5-2 : face recognition
# then print all-to-all face similarity


face_feat1 = faces1[0].normed_embedding
face_feat2 = faces2[0].normed_embedding
face_feat1 = np.array(face_feat1, dtype=np.float32)
face_feat2 = np.array(face_feat2, dtype=np.float32)
sims = np.dot(face_feat1, face_feat2.T)
print(sims)

