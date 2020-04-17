"""
===============================================================================
Document Rectifier.
The document rectifier automatically finds the document in a picture and rectifies it.
If you wish you can use the mouse to drag the corners to correct position.
Requirements for automatic document (corner) detection: 
* Document has to be completely within the image
* Document is easly distinguasable from the background
* Only one document
* Document have to take a large portion of the image
* The document must be a white paper with dark letters


USAGE:
    python documentrectifier.py <filename>

README FIRST:
    Two windows will show up,  one for input and one for output(rectified document).


Key 'Esc'- To Exit (or close window)
Key 's' - Save rectified image to rectified_output.png
===============================================================================
"""

# Python 2/3 compatibility
from __future__ import print_function
import numpy as np
import cv2
import sys
from collections import defaultdict
import math
RAD2DEG = 180.0/3.14        # Multiply with this to convert from RADIANS to DEGREES
CORNER_CIRCLE_RADIUS = 20   # CORNER_CIRCLE_RADIUS of circles marking corners that can be dragged
RECTIFIED_DOC_WIDTH = 500   # width of rectified document
INPUT_IMAGE_MAX_HEIGHT = 1280 # If input image has height grather than this, then its scaled down.

inputImageScale = 1         # If image size is greater than INPUT_IMAGE_MAX_HEIGHT, we will get a scale of 2 or higher 
                            # Mouse corrdinates will be scaled according to this variable.
cornerPoints = []           # Contains the four corners of the document
inputWindowName = "input"
outputWindowName = "rectified document"

buttonDown = False      # If mouse button is down, this is true.

def propagateChildrenCountToUpperParent(hierar,node,children_count):
    """
        Propagetes the children count to its parents, at the end,
        all children are registered at the upper most parent. (start node).
        If all leaf nodes are given as input, then all start nodes will contain
        the sum of all children below.
    """
    h = hierar[node]
    # exit if no partents.
    if h[3] == -1:
        return
    children_count[h[3]] += children_count[node]
    children_count[node] = 0    # The count has been added to parent.
    propagateChildrenCountToUpperParent(hierar,h[3],children_count)

def sortPoints(points):
    """ We have four points, number 0 should be upper left, and
    it should go clockwise, this function sorts them accordingly"""
    
    points[:] = sorted(points, key=lambda x: x[1])
    if points[0][0] > points[1][0]:
        p = points[0].copy()
        points[0] = points[1]
        points[1] = p

    if points[3][0] > points[2][0]:
        p = points[3].copy()
        points[3] = points[2]
        points[2] = p
    return points


def getLength(p1,p2):
    return math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)

# Application Function on mouse
def onmouse(event, x, y, flags, param):
    global img, imagCopy, drawing, mask, ix, iy, cornerPoints, CORNER_CIRCLE_RADIUS, buttonDown, inputImageScale
    x = inputImageScale*x
    y = inputImageScale*y

    cornerGrabbed = False   # Will contain a corner when using mouse to correct document corners
    if (event == cv2.EVENT_LBUTTONDOWN):
        buttonDown = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        # Always grab a corner if button is down
        if buttonDown:
            for cornerP in cornerPoints:
                    if getLength(cornerP, (ix,iy)) < CORNER_CIRCLE_RADIUS:
                        cornerGrabbed = cornerP

        if cornerGrabbed != False:
            cornerGrabbed[0] += x-ix
            cornerGrabbed[1] += y-iy
            ix, iy = x, y

    elif (event == cv2.EVENT_LBUTTONUP):
        buttonDown = False

  
def getDocumentCornerPoints(imagCopy,border = [20,20,20,20]):
    mask = np.zeros(img.shape[:2], dtype = np.uint8) # mask initialized to PR_BG
    rect = (border[0], border[1], img.shape[1]-border[0]-border[2], img.shape[0]-border[0]-border[3])

    bgdmodel = np.zeros((1, 65), np.float64)
    fgdmodel = np.zeros((1, 65), np.float64)
    cv2.grabCut(imagCopy, mask, rect, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_RECT)

    # Final mask is the union of definitely foreground and probably foreground
    # mask such that all 1-pixels (cv2.GC_FGD) and 3-pixels (cv2.GC_PR_FGD) are put to 1 (ie foreground) and
    # all rest are put to 0(ie background pixels)
    mask2 = np.where((mask==1) + (mask==3), 255, 0).astype('uint8')

    # Copy the region to output
    output = cv2.bitwise_and(imagCopy, imagCopy, mask=mask2)
    
    output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    o,output = cv2.threshold(output,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    contours, hierarchy = cv2.findContours(output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # A document will have black letters, thus we want the contour that
    # have the most contours inside.
    # Find parents with most children.
    #[Next, Previous, First_Child, Parent]
    # Count children
    children_count = defaultdict(int)
    for h in hierarchy[0]:
        if h[3] >= 0:
            children_count[h[3]] += 1
    
    # Find all leaf nodes - that has a parent, but no children.
    for h in hierarchy[0]:
        if h[3] >= 0 and h[2] == -1:
            propagateChildrenCountToUpperParent(hierarchy[0],h[3],children_count)

    if len(children_count) > 0:
        # get key with highest value. If several keys with same max value, the first one is selected        
        max_key = max(children_count, key=children_count.get)

        cnt = contours[max_key]
        epsilon = 0.1*cv2.arcLength(cnt,True)
        approxCorners = cv2.approxPolyDP(cnt,epsilon,True)
    
    if len(approxCorners) != 4:
        approxCorners = np.array([[[int(output.shape[1]*0.2),int(output.shape[0]*0.3)]], [[int(output.shape[1]*0.2),int(output.shape[0]*0.9)]], [[int(output.shape[1]*0.8),int(output.shape[0]*0.2)]], [[int(output.shape[1]*0.8),int(output.shape[0]*0.8)]] ])

    return approxCorners.reshape(4,2).tolist()

def getAngle(vector_1,vector_2):
    if np.sum(vector_1-vector_2) == 0:
        return 45   # Equal vectors give a high error angle.
    if np.sum(vector_1) == 0 or np.sum(vector_2) == 0:
        return 45   # zero vector give a high error angle.

    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)
    return angle*RAD2DEG

def getError(cornerPoints):
    """Assumption that the box should have parallel lines, and
    90 degrees. If calculates the square error """
    cp = np.array(cornerPoints)
    error = 0
    # Angle between top and bottom line
    a = getAngle(cp[0]-cp[1], cp[3]-cp[2])
    error += a**2
    
    # Angle between left and right line
    a = getAngle(cp[1]-cp[2], cp[0]-cp[3])
    error += a**2
    
    # Error Angle for bottom right corner
    a = getAngle(cp[3]-cp[2], cp[2]-cp[1]) - 90
    error += a**2
    
    # Error Angle for bottom left corner
    a = getAngle(cp[2]-cp[3], cp[3]-cp[0]) - 90
    error += a**2
    return error


if __name__ == '__main__':

    # print documentation
    print(__doc__)

    # Loading images if image is given bu command line
    if len(sys.argv) == 2:
        filename = sys.argv[1] # Using file for image
    else:
        print("No input image. Correct Usage: python documentscanner.py <filename> \n")
        
    originalImg = cv2.imread(filename)
    img = originalImg.copy()

    # input window, named because of callback
    cv2.namedWindow(outputWindowName)
    cv2.namedWindow(inputWindowName)

    cv2.setMouseCallback(inputWindowName, onmouse)
    cv2.moveWindow(outputWindowName, 10, 10)
    cv2.moveWindow(inputWindowName, RECTIFIED_DOC_WIDTH+10, 10)

    # Try different grab cuts.
    # Select the one with lowest error.
    cornerPoints = []
    minError = 4*46**2 # A highvlue. Initialized to  46 degrees 4 times.
    for i in range(10,21,10):
        borders = [i,i,i,i]
        cp = getDocumentCornerPoints(img,borders)
        cp = sortPoints(cp)    
        error = getError(cp)
        if error < minError:
            minError = error
            cornerPoints = cp
    
    exitLoop = False
    while(not exitLoop):        
        
        img = originalImg.copy()
        
        cornerPoints = sortPoints(cornerPoints)
        for i,a in enumerate(cornerPoints):
            cv2.circle(img, (a[0],a[1]), CORNER_CIRCLE_RADIUS, (0,0,255), thickness=1, lineType=8, shift=0)
            cv2.circle(img, (a[0],a[1]), 6, (0,0,255), thickness=3, lineType=8, shift=0)
            cv2.putText(img, str(i) ,(a[0],a[1]), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2,cv2.LINE_AA)
        
        
        # find aspect ration
        h = getLength(cornerPoints[1],cornerPoints[2])
        w = getLength(cornerPoints[0],cornerPoints[1])
        aspectRatio = h/max(w,0.1)
        aspectRatio = max(0.1,min(aspectRatio,10))  # Limit between 0.1 and 10
        size = (RECTIFIED_DOC_WIDTH, int(RECTIFIED_DOC_WIDTH*aspectRatio) ,3)

        im_dst = np.zeros(size, np.uint8)
        pts_dst = np.array(
                       [
                        [0,0],
                        [size[0] - 1, 0],
                        [size[0] - 1, size[1] -1],
                        [0, size[1] - 1 ]
                        ], dtype=float
                       )
        pts_src = np.array(cornerPoints, float)
        
        # Calculate the homography
        h, status = cv2.findHomography(pts_src, pts_dst)
        # Warp source image to destination
        im_dst = cv2.warpPerspective(originalImg, h, size[0:2])
        
        if img.shape[0] > INPUT_IMAGE_MAX_HEIGHT:
            inputImageScale = math.ceil(img.shape[0] / INPUT_IMAGE_MAX_HEIGHT)
            scale = 1/inputImageScale
            img = cv2.resize(img,None,fx=scale,fy=scale)

        # Exit if one of the windows are closed
        if cv2.getWindowProperty(inputWindowName, 0) < 0 or cv2.getWindowProperty(outputWindowName, 0) < 0:
            exitLoop = True
        k = cv2.waitKey(1)
        if k == ord('q') or k == 27:    # Exit with q or ESC
            exitLoop = True
        elif k == ord('s'):     # save image
            cv2.imwrite('rectified_output.png', im_dst)
            print("Result saved as image \n")
        
        cv2.imshow(inputWindowName, img)
        cv2.imshow(outputWindowName, im_dst)

    cv2.destroyAllWindows()

    


