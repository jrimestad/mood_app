"""
Strong inspiration from https://github.com/serengil/deepface/blob/97d0a7d1dfaa055ea2b38117ba837bd22c691a7c/deepface/commons/realtime.py#L12
author: jens.rimestad@gmail.com
"""

import cv2
from deepface import DeepFace
from deepface.detectors import FaceDetector
from deepface.commons import functions
import PIL.Image as Image
import time
import numpy as np

def overlay_image_alpha(img, img_overlay, x, y, alpha_mask):
    """Overlay `img_overlay` onto `img` at (x, y) and blend using `alpha_mask`.

    `alpha_mask` must have same HxW as `img_overlay` and values in range [0, 1].
    """
    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    # Blend overlay within the determined ranges
    img_crop = img[y1:y2, x1:x2]
    img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]
    alpha = alpha_mask[y1o:y2o, x1o:x2o, np.newaxis]
    alpha_inv = 1.0 - alpha

    img_crop[:] = alpha * img_overlay_crop + alpha_inv * img_crop

# Define global variables
emotion_labels = ['ANGRY', 'DISGUST', 'FEAR', 'HAPPY', 'SAD', 'SURPRISE', 'NEUTRAL']    
model_name = 'OpenFace'#'VGG-Face'
detector_backend = 'opencv'
face_detector = FaceDetector.build_model(detector_backend)

time_threshold = 30
frame_threshold = 60
input_shape = (224, 224); input_shape_x = input_shape[0]; input_shape_y = input_shape[1]
text_color = (255,255,255)

# Init model and webcam
emotion_model = DeepFace.build_model('Emotion')
source = 0 # Webcam to use
cap = cv2.VideoCapture(source, cv2.CAP_DSHOW) #webcam
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

playing = False
show_stats = True
no_face_found_frames = 0
emotion_thresh = [0.5, 0.25, 0.5, 0.6, 0.4, 0.6, 0.35]
face_detected = False
tic = time.time()
toc = tic
detected_emotions = set()
text_space = 80
barwidth = 270
bar_offset = 15
bar_height = 4
logo_img = Image.open("logo.png").resize((200, 200))
emotion_predictions = np.zeros((7), np.float)

cv2.namedWindow('img',cv2.WINDOW_NORMAL)
# cv2.resizeWindow('img', 600,600)
while(True):
    ret, img = cap.read()
    img = cv2.flip(img, flipCode=1)

    if show_stats:
        time_used = toc - tic
        cv2.putText(img, "Detected %d emotions in %0.0f seconds." % (len(detected_emotions), time_used), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
        cv2.putText(img, "Press space-bar to start.", (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
        cv2.imshow('img', img)
    else:
        resolution = img.shape; resolution_x = img.shape[1]; resolution_y = img.shape[0]

        faces = FaceDetector.detect_faces(face_detector, detector_backend, img, align = False)

        # Find largest face
        max_face = 200
        detected_face = None
        face_detected = False
        max_face_dims = np.array([0,0,0,0])
        for face, (x, y, w, h) in faces:
            y -= int(h * 0.05)
            y = np.clip(y, 0, resolution_y)
            h += int(h * 0.1)
            
            if w * h > max_face:
                max_face = w * h
                face_detected = True
                max_face_dims = np.array([x, y, w, h])
                # detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face

        if face_detected:
            detected_face = img[int(max_face_dims[1]):int(max_face_dims[1]+max_face_dims[3]), int(max_face_dims[0]):int(max_face_dims[0]+max_face_dims[2])] #crop detected face
            cv2.rectangle(img, max_face_dims[0:2], max_face_dims[0:2] + max_face_dims[2:4], (180,180,180), 1) #draw rectangle to main image
            if playing==False:
                # Start game
                playing = True
                show_stats = False
                no_face_found_frames = 0
                emotion_predictions = np.zeros((7), np.float)
                detected_emotions = set()
                tic = time.time()

        toc = time.time()
        runtime = toc - tic
        if runtime > time_threshold or len(detected_emotions) == len(emotion_labels) or no_face_found_frames > frame_threshold:
            # Time is up, all emotions detected or no face seen for x frames 
            # Show game stats
            print("Player got %d emotions in %0.3f seconds" % (len(detected_emotions), runtime))
            playing = False
            show_stats = True
        else:
            if face_detected:
                # Look for emotions in detected face
                no_face_found_frames = 0
                gray_img = functions.preprocess_face(img = detected_face, target_size = (48, 48), grayscale = True, enforce_detection = False, detector_backend = 'opencv')		
                emotion_predictions = 0.85 * emotion_predictions + 0.15 * emotion_model.predict(gray_img)[0,:]
                # max_emotion = np.argmax(emotion_predictions)
                if any(emotion_predictions > emotion_thresh):
                # if emotion_predictions[max_emotion] >= emotion_thresh:
                    detected_emotions.add(np.argmax(emotion_predictions > emotion_thresh))
            else:
                no_face_found_frames += 1
                                
            # Time left for detecting emotions
            time_left = int(time_threshold - runtime + 1)
            cv2.rectangle(img, (20, 20), (370, resolution_y - 20), (67,67,67), -10)
            cv2.putText(img, "TIME:%ds" % (time_left), (40, 85), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 1)

            # Show emotion status
            for i, emotion in enumerate(emotion_labels):
                detected = i in detected_emotions
                cv2.putText(img, str(emotion), (40, (i + 2) * text_space), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0) if detected else (0, 0, 255))
                cv2.line(img, (40, (i + 2) * text_space + bar_offset), (40 + barwidth, (i + 2) * text_space + bar_offset), (255, 255, 255), bar_height)
                cv2.line(img, (40, (i + 2) * text_space + bar_offset), (int(40 + min(1.0, emotion_predictions[i] / emotion_thresh[i]) * barwidth), (i + 2) * text_space + bar_offset), (0, emotion_predictions[i] * 255, 0), bar_height)

        # img = overlay_image_alpha(img, logo_img, resolution_x - 200, 40, np.array(logo_img)[:,:,0] / 255)
        cv2.imshow('img',img)

    key = cv2.waitKey(1)
    if key:
        if key & 0xFF == ord('q'): #press q to quit
            break
        if key & 0xFF == ord(' '): #start game
            show_stats = False
        

#kill open cv things
cap.release()
cv2.destroyAllWindows()
