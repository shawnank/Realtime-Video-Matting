import numpy as np
import cv2
from numba import njit

src = 0

@njit('u1[:,:,:](i8,i8,u1[:,:,:],u1[:,:])')
def cvt(width, height, frame, bw):
    for i in range(width):
        for j in range(height):
            if(bw[j,i]==0): 
                frame[j,i,0]=0
                frame[j,i,1]=0
                frame[j,i,2]=0
    return frame

cap = cv2.VideoCapture(src)
ret, frame=cap.read()
width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(width, height)
avg = cv2.blur(frame, (4, 4))
avg_float = np.float32(avg)

stay = frame.copy()
stay[0:, 0:, 0:]=0
blk = stay.copy()

while(True):
    ret, frame = cap.read()
    cur = cv2.blur(frame, (4, 4))
    diff = cv2.absdiff(avg, cur)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    frame = cvt(width, height, frame, thresh)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    frame = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    sav = frame.copy()
    #sav[0:, 0:, 0:2]=0
    cv2.imshow('show',sav)
    cv2.accumulateWeighted(cur, avg_float, 0.01)
    avg = cv2.convertScaleAbs(avg_float)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
