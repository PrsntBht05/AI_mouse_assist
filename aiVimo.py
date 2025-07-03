import cv2
import time
import numpy as np
import HandTrackingModule as htm
import pyautogui  # Import pyautogui for mouse control

############################
wCam, hCam = 640, 480
frameR = 98;
smoothening = 7  # Smooth movement
############################

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)  # Set webcam width
cap.set(4, hCam)  # Set webcam height
detector = htm.handDetector(maxHands=1)
wScr, hScr = pyautogui.size()  # Get screen width and height using pyautogui

print(f"Screen Resolution: {wScr}, {hScr}")  # Check if screen resolution is correct

while True:
    success, img = cap.read()  # Capture image from webcam
    if not success:
        print("Error: Failed to capture image")
        break

    img = detector.findHands(img)  # Detect hands in the image
    lmList, bbox = detector.findPosition(img)  # Get hand landmarks

    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]  # Get index finger tip coordinates
        x2, y2 = lmList[12][1:]  # Get middle finger tip coordinates

        fingers = detector.fingersUp()  # Check which fingers are up

        # Draw a rectangle around the active area
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

        # Moving Mode: Only index finger up
        if fingers[1] == 1 and fingers[2] == 0:
            # Map camera coordinates (x1, y1) to screen coordinates
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))  # Map x1 to screen width
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))  # Map y1 to screen height

            # Smooth the mouse movement
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            # Move the mouse using pyautogui
            pyautogui.moveTo(clocX, clocY)  # Move the mouse to the new position

            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)  # Draw a circle on the index finger
            plocX, plocY = clocX, clocY  # Update previous location for smoothening

        # Clicking Mode: Both index and middle fingers up
        if fingers[1] == 1 and fingers[2] == 1:
            length, img, lineInfo = detector.findDistance(8, 12, img)  # Measure the distance between index and middle fingers
            if length < 39:  # If the distance is small enough (indicating a "click")
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)  # Draw a circle on click point
                pyautogui.click()  # Perform the click action

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (40, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)

    cv2.imshow("Img", img)  # Display the image

    # Exit when pressing the Esc key
    if cv2.waitKey(10) & 0xFF == 27:  # 27 is the keycode for the "Esc" key
        break

cap.release()
cv2.destroyAllWindows()
