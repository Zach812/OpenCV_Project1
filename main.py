import cv2
from PIL import Image
from util import get_limits


yellow = [0, 255, 255] # Yellow in BGR colorspace
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    hsvimage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lowerLimit, upperLimit = get_limits(color=yellow)

    # All pixels that are the desired color
    mask = cv2.inRange(hsvimage, lowerLimit, upperLimit)

    mask_ = Image.fromarray(mask)

    bbox = mask_.getbbox()
    
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cont in contours:
        # Isolating each white image within thresh
        if cv2.contourArea(cont) > 200:
            # Removing noise
            # Can see dots around the img (green little dots)
            # cv2.drawContours(img,cont, -1, (0,255, 0), 1)
            x1, y1, w, h = cv2.boundingRect(cont)
            # Building Object Detector
            cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 255), 2)
    cv2.imshow('mask', mask)

    
    if bbox is not None:
        x1, y1, x2, y2 = bbox

        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

    cv2.imshow('frame', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()

