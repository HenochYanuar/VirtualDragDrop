import cv2
from cvzone.HandTrackingModule import HandDetector
import cvzone
import numpy as np

cap = cv2.VideoCapture(0) # Get the input video from default camera
# Setting resolution input to 1280 x 720
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(detectionCon=0.8)
rectangleColor = (255, 0, 255) # Set the rectangle color for objects fill

class Rectangle: # Create a Rectangle class to declare the attributes of each rectangle/objects
    def __init__(self, centerPos, size=None):
        if size is None: # If the size from parameter is None
            size = [100, 100] # Set the size of rectangle/objects
        self.centerPos = centerPos # Set the objects center position according to the centerPos parameter
        self.size = size # Set the objects size according to the size parameter

    def posUpdate(self, cursor): # Create a function to update the position of objects that are drag and drop
        xCenter, yCenter, _ = self.centerPos # Get the x and y center point coordinates of the rectangle from __init__ function
        width, height = self.size # Get the width and height of the rectangle from __init__ function

        # If the coordinates of cursor(tip of index finger) on rectangle area
        if xCenter - width // 2 < cursor[0] < xCenter + width // 2 and \
                yCenter - height // 2 < cursor[1] < yCenter + height // 2:
            self.centerPos = cursor # Change the center point value of the rectangle with the coordinates of the cursor parameters

rectangleList = [] # Variable to store all created rectangles/objects

for x in range(5): # range(5) is used to set how many rectangles will be created
    rectangleList.append(Rectangle([x*120+70, 70, 0])) # Set the center position of rectangle/object using Rectangle class and append to rectangleList[]

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)  # Detected there a hand on the input

    if hands: # If a hand is detected
        hand1 = hands[0] # Get the data from detected hand
        lmList1 = hand1['lmList'] # Get the landmarks from detected hand
        x8, y8, _ = lmList1[8] # Get the x and y coordinates of the landmark point 8 (tip of index finger) on the hand1
        x12, y12, _ = lmList1[12] # Get the x and y coordinates of the landmark point 12 (tip of the middle finger) on the hand1

        length, _, _ = detector.findDistance((x8, y8), (x12, y12), img, scale=0) # Find the length of the distance between the lmList1[8] (tip of the index finger) to the lmList1[12] (tip of the middle finger)

        if length < 30: # If the length of the distance between the tip of the index finger to the tip of the middle finger is less then 30
            # Call the update position function from class Rectangle
            for rectangle in rectangleList: # Looping to get all data in the rectangleList array
                rectangle.posUpdate(lmList1[8]) # lmList1[8] is for cursor parameter

    # Draw a transparent rectangle
    imgNew = np.zeros_like(img, np.uint8) # Creates an imgNew array that has the same dimensions and data type as img, but all its elements are initialized with zero values.
    for rectangle in rectangleList: # Looping to get all data in the rectangleList array
        cx, cy, _ = rectangle.centerPos # Get the center point x and y of each rectangle in the rectangleList[]
        w, h = rectangle.size # Get the width and height of each rectangle in the rectangleList[]
        # Draw a rectangle in imgNew with the top left corner at (cx-w//2, cy-h//2) and the bottom right corner at (cx+w//2, cy+h//2). The color of the rectangle is given by rectangleColor, and the cv2.FILLED parameter is used to fill the rectangle
        cv2.rectangle(imgNew, (cx-w//2, cy-h//2), (cx+w//2, cy+h//2), rectangleColor, cv2.FILLED)
        # Add corners to a rectangle in imgNew using the cornerRect function from the cv zone library. This provides a clearer visual appearance of the corners of the rectangle
        cvzone.cornerRect(imgNew, (cx-w//2, cy-h//2, w, h),20, 3,2)
    img = img.copy() # Make a copy of the image in the img variable
    alpha = 0
    mask = imgNew.astype(bool)
    img[mask] = cv2.addWeighted(img, alpha, imgNew,1 - alpha, 0)[mask]

    cv2.imshow("Image", img)
    cv2.waitKey(1)
