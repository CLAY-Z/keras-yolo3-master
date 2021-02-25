import cv2

url = "rtsp://admin:HK123456@192.168.1.20:8000/Streaming/Channels/1"
cap = cv2.VideoCapture(url)

if cap.isOpened():
    flag, image = cap.read()
    while flag:
        flag, image = cap.read()
        cv2.imshow('frame', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    cap.release()
