import cv2 as cv
import torch
import torch.nn as nn
import torchvision.models as models
import mediapipe as mp


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Emotion(nn.Module):
    def __init__(self):
        super(Emotion,self).__init__()
        vgg = models.vgg13(pretrained=True)
        vgg.features[0] = nn.Conv2d(1, 64, 3,1, padding =1)
        self.features = vgg.features
        self.avgpool = vgg.avgpool
        self.classifier = nn.Sequential(nn.Linear(512*7*7, 256),nn.ReLU(),nn.Dropout(0.5),nn.Linear(256, 7))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x= self.classifier(x)
        return x

model = Emotion()
model = model.to(device)
model.load_state_dict(torch.load('fer_emotion_vgg.pth',map_location=device))
model.eval()
print("model set hai janab")

emotions = ['angry','disgust','fear','happy','sad','surprise','neutral']

mp_face=mp.solutions.face_detection
face = mp_face.FaceDetection(model_selection=0,min_detection_confidence=0.1)


capture = cv.VideoCapture(0)
capture.set(cv.CAP_PROP_FRAME_WIDTH,640)
capture.set(cv.CAP_PROP_FRAME_HEIGHT,480)

frameskip=0
while True:
    frameskip+=1
    ret , frame = capture.read()
    h,w,_ = frame.shape

    rgb = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
    result =face.process(rgb)

    if result.detections:
        for detections in result.detections:


            bbox = detections.location_data.relative_bounding_box
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            ww = int(bbox.width*w)
            hh = int(bbox.height *h)

            x=max(0,x)
            y=max(0,y)

            test = frame[y:y+hh,x:x+ww]


            if test.size ==0:
                continue

            test = cv.cvtColor(test,cv.COLOR_BGR2GRAY)
            test = cv.resize(test,(48,48))
            test = cv.equalizeHist(test)

            test = test.astype('float32')
            test = (test - 128) / 128.0


            face_tensor = torch.from_numpy(test).float().unsqueeze(0).unsqueeze(0)
            face_tensor = face_tensor.to(device)

            with torch.no_grad():
                pred = model(face_tensor)
                pred = torch.softmax(pred,dim=1)
            emotion = emotions[pred.argmax(dim=1).item()]
            confidence = pred[0,pred.argmax().item()]
            cv.rectangle(frame, (x, y), (x + ww, y + hh), (0, 255, 0), 2)
            cv.putText(frame, f"{emotion} with {confidence * 100:.1f}% confidence", (x, y - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv.imshow('detecting', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break