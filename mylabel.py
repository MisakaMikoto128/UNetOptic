import cv2
import numpy as np

for i in range(8):
    file = "%d.png"%(i)
    path = "data/train1/label/"
    gray = cv2.imread(path+file,cv2.IMREAD_GRAYSCALE)
    ret, binary = cv2.threshold(gray, gray.mean(), 255, cv2.THRESH_BINARY)
    contours,hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # label = np.zeros((binary.shape[0],binary.shape[1],3))
    label = np.zeros(binary.shape)
    label = cv2.drawContours(label, contours, -1, (255, 255, 255), 7)
    cv2.imwrite("data/train1/label1/"+("%d"%i)+".png",label)
    print("第%d张图片采样保存成功"%(i))


