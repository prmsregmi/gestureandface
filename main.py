import cv2, csv, face_recognition, copy, math, itertools, os, datetime
import config
import mediapipe as mp
import numpy as np
from model import KeyPointClassifier
from pprint import pprint

#parameter definition
buffer = True #if buffer, only gives response after buffer time else continuous
buffer_duration = 5 #define in number of loops

#global variables definition
known_face_encodings = [] #encodings of faces for recognition
known_face_names = [] #names of those encodings in serial order
face_buffer = [] #buffer time (seconds) remaining in a face
face_locations = [] #location of faces in each frame during processing
face_encodings = [] #encodings of located faces



#detection model import
mp_face_detection = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

#label importing for hand gestures
with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
keypoint_classifier = KeyPointClassifier()

#function to generate face_encodings
def create_face_encodings(folder= "facial_images"):
    for filename in os.listdir(folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            tmp_img = face_recognition.load_image_file(os.path.join(folder, filename))
            known_face_encodings.append(face_recognition.face_encodings(tmp_img)[0])
            known_face_names.append(filename.split(".")[0])
            face_buffer.append(buffer_duration)
create_face_encodings()
#calculate landmarks from image in a frame
def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_face_detection.FaceDetection(
    min_detection_confidence=0.5) as face_detection, mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5, max_num_hands=10) as hands, mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:
 while cap.isOpened():
    for i in range(0, len(face_buffer)):
        if (face_buffer[i] < buffer_duration):
            face_buffer[i]+=1

    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    
    
    if config.process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(image)
        face_landmarks_list = face_recognition.face_landmarks(image)
           
        face_encodings = face_recognition.face_encodings(image, face_locations)

        face_names = []
        for i, face_encoding in enumerate(face_encodings):
            nose_x = face_landmarks_list[i]['nose_tip'][0][0]
            nose_y = face_landmarks_list[i]['nose_tip'][0][1]
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            cv2.imwrite(datetime.datetime.now().strftime("hello"), image)
            # cv2.imwrite(datetime.datetime.now().strftime("%m-%d-%Y,%H:%M:%S"), image)

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            face_recognition_accuracy = (1-face_distances[best_match_index])*100
            if face_recognition_accuracy > config.face_match_threshold and matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)
            hand_results = hands.process(image)

    config.process_this_frame = not config.process_this_frame
    
    
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    face_results = face_detection.process(image)
    hand_results = hands.process(image)
    
    if hand_results.multi_hand_landmarks:
          for hand_landmarks in hand_results.multi_hand_landmarks:
            landmark_list = calc_landmark_list(image, hand_landmarks)

            # Conversion to relative coordinates / normalized coordinates
            pre_processed_landmark_list = pre_process_landmark(landmark_list)
            hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
#             print(keypoint_classifier_labels[hand_sign_id])
            hand_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x * image.shape[1]
            hand_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * image.shape[0]
            dist = math.sqrt(pow((hand_x-nose_x),2)+pow((hand_y-nose_y),2))
    #         print('distance is')
#             print(dist)
            if(dist < config.threshold):
                if not buffer:
                    print(name, keypoint_classifier_labels[hand_sign_id])
                else:
                    if face_recognition_accuracy > config.face_match_threshold and matches[best_match_index]:
                        if(face_buffer[best_match_index] == buffer_duration):
                            print(name, keypoint_classifier_labels[hand_sign_id])
                            face_buffer[best_match_index] = 0
                
            if (config.display_window and config.display_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    
    
#     holistic_results = holistic.process(image)

    # Draw the face detection annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if face_results.detections:
      for detection in face_results.detections:
          if (config.display_window and config.display_bounding_box):
            mp_drawing.draw_detection(image, detection)
    if(config.display_window):    
        cv2.imshow('Face and Hand Detection', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cv2.destroyAllWindows()
cap.release()