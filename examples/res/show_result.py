import cv2
import json
import numpy as np

if __name__ == '__main__':
    with open("./alphapose-results.json") as f:
        results = json.load(f)
    if len(results)==0:
        print("no result!!!")
        exit(0)
    results = results[0]
    kpts = results["keypoints"]
    kpts = np.array(kpts).reshape((-1, 3))

    img = cv2.imread("../../images/1658416958339222600_front.jpg", cv2.IMREAD_UNCHANGED)
    for i in range(kpts.shape[0]):
        cv2.circle(img, center=(int(kpts[i][0]), int(kpts[i][1])), radius=2, color=(0, 0, 255))
        cv2.putText(img, str(i), (int(kpts[i][0]), int(kpts[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=(255, 0, 0))
    cv2.imshow("fff", img)
    cv2.waitKey(1000000)