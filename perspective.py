import cv2
import numpy as np
import imutils
    
#Find the file vertex using contour approach
def get_vertex(image):
    #ratio = image.shape[0]/300.0
    orig = image.copy()
    #image = imutils.resize(image,height = 300)
    #convert into grayscale and blur
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray,11,17,17)
    edged = cv2.Canny(gray,30,200)
    #edge detect processing
    (contours,_) = cv2.findContours(edged.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #for c in contours:
     #   cv2.drawContours(image,[c],-1,(0,255,0),3)
    contours = sorted(contours,key = cv2.contourArea,reverse = True)[:10]
    fileContour=None

    #Get the contour of file
    for contour in contours:
        peri = cv2.arcLength(contour,True)
        approx = cv2.approxPolyDP(contour,0.02*peri,True)
        cv2.drawContours(image,[approx],-1,(0,255,0),3)
        if len(approx)==4:
            fileContour = approx
            break
    #show contour
    cv2.drawContours(image,[fileContour],-1,(0,255,0),3)
    cv2.imwrite("test.jpg",image)
    #cv2.imshow("Shoe contour",image)
    #cv2.waitKey(0)
    return fileContour,orig

def formate_input(raw):
    vts=np.array([raw[i][0] for i in range(4)],dtype="float32")
    #expected:top-left,top-right,bottom-right,bottomleft
    rect = np.zeros((4,2),dtype = "float32")
    #since the top-left smallest and bottom-right smallest
    s = vts.sum(axis=1)
    rect[0] = vts[np.argmin(s)]
    rect[2] = vts[np.argmax(s)]

    #top-right have smallest difference and bottom-left have the biggest

    diff = np.diff(vts,axis=1)
    rect[1] = vts[np.argmin(diff)]
    rect[3] = vts[np.argmax(diff)]
    return rect

def transform(image,vts):
    vts = formate_input(vts)
    (tl,tr,br,bl) = vts
    #compute width of new image(max distance between bottom-right and bottom-left or tr and tl)
    width_bottom = np.sqrt(((br[0]-bl[0])**2)+((br[1]-bl[1])**2))
    width_top = np.sqrt(((tr[0]-tl[0])**2)+((tl[1]-tr[1])**2))
    width = max(int(width_top),int(width_bottom))
    #so does the height
    height_left = np.sqrt(((tr[0]-br[0])**2)+((br[1]-tr[1])**2))
    height_right = np.sqrt(((tl[0]-bl[0])**2)+((bl[1]-tl[1])**2))
    height = max(int(height_right),int(height_left))
    #transform
    dst = np.array([[0,0],[width-1,0],[width-1,height-1],[0,height-1]],dtype="float32")
    temp = cv2.getPerspectiveTransform(vts,dst)
    warped = cv2.warpPerspective(image,temp,(width,height))
    #cv2.imshow("warpped",warpped)
    #cv2.waitKey(0)
    return warped

if __name__ == "__main__":
    s=raw_input();
    img=cv2.imread(s)
    vts,img2=get_vertex(img)
    warpped=transform(img2,vts)
    cv2.imwrite("new.jpg",warpped)
#    cv2.imshow("warpped",warpped)
 #   cv2.waitKey(0)

