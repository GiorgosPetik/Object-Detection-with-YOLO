import cv2
import numpy as np

#Weight Target cause of the YOLO version
whT = 320
confidenceThreshold = 0.5
nms_threshold = 0.3


classesFile = 'coco.names'
classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
#print(classNames)
#print(len(classNames))

modelConfiguration = 'YOLO-320.cfg'
modelWeights = 'yolov3-320.weights'

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def findObjects(outputs, img):
    height,weight,channels = img.shape
    boundingBox = []
    classIds = []
    confidenceValues = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > confidenceThreshold:
                w,h = int(detection[2]*weight),int(detection[3]*height)
                x,y = int(detection[0]*weight - weight/2), int((detection[1]*height) -height/2)
                boundingBox.append([x,y,w,h])
                classIds.append(classID)
                confidenceValues.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(boundingBox,confidenceValues,confidenceThreshold,nms_threshold)

    for i in indices:
        i = i[0]
        box = boundingBox[i]
        x,y,w,h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img,(x,y), (x+w, y+h),(255,0,0),2)
        cv2.putText(img,f'{classNames[classIds[i]].upper()}{int(confidenceValues[i]*100)}%',
                    (x,y - 10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2)

# Camera activation
capture = cv2.VideoCapture(1)
while True:
    success, img = capture.read()

    blob = cv2.dnn.blobFromImage(img, 1/255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)

    layerNames = net.getLayerNames()
    #print(layerNames)

    outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]
    #print(outputNames)
    outputs = net.forward(outputNames)
    findObjects(outputs,img)

    cv2.imshow('Image',img)
    cv2.waitKey(1)

