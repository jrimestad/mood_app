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
face_detector = FaceDetector.build_model(detector_backend)

time_threshold = 30
frame_threshold = 5
input_shape = (224, 224); input_shape_x = input_shape[0]; input_shape_y = input_shape[1]
text_color = (255,255,255)

# Init model and webcam
emotion_model = DeepFace.build_model('Emotion')
source = 0 # Webcam to use
cap = cv2.VideoCapture(source) #webcam

playing = False
show_stats = True
no_face_found_frames = 0
face_detected = False
tic = time.time()
toc = tic
detected_emotions = set()

while(True):
    ret, img = cap.read()

    if show_stats:
        time_used = toc - tic
        cv2.putText(img, str(time_used), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
        cv2.putText(img, str("Detected %d emotions." % (len(detected_emotions))), (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
        cv2.putText(img, str("Press 's' to start.", (40, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
        cv2.imshow(img)
    else:
        resolution = img.shape; resolution_x = img.shape[1]; resolution_y = img.shape[0]

        faces = FaceDetector.detect_faces(face_detector, detector_backend, img, align = False)

        # Find largest face
        max_face = 130
        detected_face = None
        face_detected = False
        for face, (x, y, w, h) in faces:
            if w > max_face:
                max_face = w
                face_detected = True
                cv2.rectangle(img, (x,y), (x+w,y+h), (67,67,67), 1) #draw rectangle to main image
                detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face

        if face_detected and ~playing:
            # Start game
            playing = True
            show_stats = False
            no_face_found_frames = 0
            detected_emotions = set()
            tic = time.time()

        if (toc - tic) > time_threshold or len(detected_emotions) == len(emotion_labels) or no_face_found_frames > frame_threshold:
            # Time is up, all emotions detected or no face seen for x frames 
            # Show game stats
            print("Player got %d emotions in %0.3f seconds" % (len(detected_emotions), toc - tic))
            playing = False
            show_stats = True
        else:
            if face_detected:
                # Look for emotions in detected face
                no_face_found_frames = 0
                gray_img = functions.preprocess_face(img = detected_face, target_size = (48, 48), grayscale = True, enforce_detection = False, detector_backend = 'opencv')		
                emotion_predictions = emotion_model.predict(gray_img)[0,:]
                max_emotion = np.argmax(emotion_predictions)
                if emotion_predictions[max_emotion] >= emotion_thresh:
                    detected_emotions.append(max_emotion)
            else:
                no_face_found_frames += 1
                                
            # Time left for detecting emotions
            time_left = int(time_threshold - (toc - tic) + 1)
            cv2.rectangle(img, (10, 10), (90, 50), (67,67,67), -10)
            cv2.putText(img, str(time_left), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

            # Show emotion status
            for i, emotion in enumerate(emotion_labels):
                detected = i in detected_emotions
                cv2.putText(img, str(emotion), (40, (i + 2) * 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if detected else (255, 0, 0))

        cv2.imshow('img',img)
        toc = time.time()

    if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
        break

    if cv2.waitKey(1) & 0xFF == ord('s'): #start game
        show_stats = False

#kill open cv things
cap.release()
cv2.destroyAllWindows()
