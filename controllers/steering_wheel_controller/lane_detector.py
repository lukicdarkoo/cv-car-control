import cv2
import numpy as np
from skimage.transform import hough_line


class LaneDetector:
    def __init__(self):
        img = cv2.imread('lane_test.png')
        self.detect_lanes(img)

    def detect_lanes(self, image):
        image = image[330:500, :]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        edges = cv2.Laplacian(gray, ksize=9, ddepth=cv2.CV_16U) / 2**8
        edges = edges.astype(np.uint8)
        print(np.max(edges))
        edges[edges > 100] = 255 
        edges[edges <= 100] = 0

        out, angles, d = hough_line(edges)
        for o, theta, rho in zip(out, angles, d):
            print(theta * 180 / np.pi, rho)
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))

            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cv2.imshow('houghlines3.jpg', image)
        cv2.waitKey(2000)

        return lines[0][0][1]


LaneDetector()
