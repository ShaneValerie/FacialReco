import cv2

for i in range(4):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ok, frame = cap.read()
        if ok:
            cv2.imshow(f"Camera index {i} - press any key", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    cap.release()

    