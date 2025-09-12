import cv2

cap = cv2.VideoCapture("/home/gongjae/PycharmProjects/rPPG-Toolbox/dataset/V4V/Phase_1_Training_Validation_sets/Videos/train/F001_T1.mkv")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
