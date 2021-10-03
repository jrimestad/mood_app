"""
Strong inspiration from https://github.com/serengil/deepface/blob/97d0a7d1dfaa055ea2b38117ba837bd22c691a7c/deepface/commons/realtime.py#L12
author: jens.rimestad@gmail.com
"""

import cv2
from deepface import DeepFace
from deepface.commons import functions
import PIL.Image as Image
import time

# Define global variables
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']    
model_name = 'VGG-Face'
detector_backend = 'opencv'
time_threshold = 5
frame_threshold = 5
input_shape = (224, 224); input_shape_x = input_shape[0]; input_shape_y = input_shape[1]
text_color = (255,255,255)

# Init model and webcam
emotion_model = DeepFace.build_model('Emotion')
source = 0 # Webcam to use
cap = cv2.VideoCapture(source) #webcam

freeze = False
face_detected = False
face_included_frames = 0 #freeze screen if face detected sequantially 5 frames
freezed_frame = 0
tic = time.time()

while(True):
    ret, img = cap.read()
    if True:
        cv2.imshow(img)
    else:
        raw_img = img.copy()
        resolution = img.shape; resolution_x = img.shape[1]; resolution_y = img.shape[0]
        if freeze == False:
            #faces = face_cascade.detectMultiScale(img, 1.3, 5)

            #faces stores list of detected_face and region pair
            faces = FaceDetector.detect_faces(face_detector, detector_backend, img, align = False)

            if len(faces) == 0:
                face_included_frames = 0
        else:
            faces = []

        detected_faces = []
        face_index = 0
        for face, (x, y, w, h) in faces:
            if w > 130: #discard small detected faces

                face_detected = True
                if face_index == 0:
                    face_included_frames = face_included_frames + 1 #increase frame for a single face

                cv2.rectangle(img, (x,y), (x+w,y+h), (67,67,67), 1) #draw rectangle to main image

                cv2.putText(img, str(frame_threshold - face_included_frames), (int(x+w/4),int(y+h/1.5)), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 2)

                detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face

                #-------------------------------------

                detected_faces.append((x,y,w,h))
                face_index = face_index + 1

                #-------------------------------------

        if face_detected == True and face_included_frames == frame_threshold and freeze == False:
            freeze = True
            #base_img = img.copy()
            base_img = raw_img.copy()
            detected_faces_final = detected_faces.copy()
            tic = time.time()

        if freeze == True:
            toc = time.time()
            if (toc - tic) < time_threshold:
                if freezed_frame == 0:
                    freeze_img = base_img.copy()

                for detected_face in detected_faces_final:
                    x = detected_face[0]; y = detected_face[1]
                    w = detected_face[2]; h = detected_face[3]

                    cv2.rectangle(freeze_img, (x,y), (x+w,y+h), (67,67,67), 1) #draw rectangle to main image

                    #-------------------------------
                    #apply deep learning for custom_face
                    custom_face = base_img[y:y+h, x:x+w]
                    gray_img = functions.preprocess_face(img = custom_face, target_size = (48, 48), grayscale = True, enforce_detection = False, detector_backend = 'opencv')		
                    emotion_predictions = emotion_model.predict(gray_img)[0,:]
                                
            time_left = int(time_threshold - (toc - tic) + 1)

            cv2.rectangle(freeze_img, (10, 10), (90, 50), (67,67,67), -10)
            cv2.putText(freeze_img, str(time_left), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

            cv2.imshow('img', freeze_img)

            freezed_frame = freezed_frame + 1
        else:
            face_detected = False
            face_included_frames = 0
            freeze = False
            freezed_frame = 0
            cv2.imshow('img',img)

    if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
        break

#kill open cv things
cap.release()
cv2.destroyAllWindows()
