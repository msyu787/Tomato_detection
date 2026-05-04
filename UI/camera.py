import cv2
for i in range(10):
    cap = cv2.VideoCapture(i)
    ok, _ = cap.read()
    print(i, "OK" if ok else "X")
    cap.release()