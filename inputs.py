import cv2
import pathlib
import numpy as np
import pygetwindow as gw
import pyautogui

cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"
clf = cv2.CascadeClassifier(str(cascade_path))
camera = cv2.VideoCapture("smilingperson.mp4")

while True:
    _, frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = clf.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, width, height) in faces:
        cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 255, 0), 2)

    cv2.imshow("Faces", frame)  # Moved this line outside the for loop
    if cv2.waitKey(1) == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

cmd_window = gw.getWindowsWithTitle('Command Prompt')[0]

# Get the coordinates of the window
left, top, right, bottom = cmd_window.left, cmd_window.top, cmd_window.right, cmd_window.bottom

# Capture a screenshot of the command prompt window
screenshot = pyautogui.screenshot(region=(left, top, right - left, bottom - top))

# Save the screenshot
screenshot.save('cmd_screenshot.png')