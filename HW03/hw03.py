import cv2
import numpy as np
from collections import deque


img_path = "cat.jpg"
fixed_img = cv2.imread(img_path, cv2.IMREAD_COLOR_BGR)

move = deque([0, 10, 0, 0])

video = []
for i in range(4):
    info = np.array([0, 0, 0, 0])
    for j in range(10):
        background = np.zeros(shape=(450, 400, 3), dtype=np.uint8)
        img = fixed_img

        info += np.array(move)
        s1, e1, s2, e2 = info

        background[e1:450-s1, e2:400-s2] = fixed_img[s1:450-e1, s2:400-e2]
        video.append(background)
    move.rotate(-1)

np.save("202401833.npy", video)
for img in video:
    cv2.imshow("cat", img)
    cv2.waitKey(250)



