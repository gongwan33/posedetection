import cv2

vidcap = cv2.VideoCapture('short.mp4')

if vidcap.isOpened() == False:
    print("Cannot open file")

while vidcap.isOpened():
    success, frame = vidcap.read()
    cv2.imshow('Frame', frame)
    cv2.waitKey(1)

vidcap.release()


