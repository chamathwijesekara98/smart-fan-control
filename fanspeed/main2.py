import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def detect_gesture(landmarks):
    thumb_tip = landmarks[4]  # Thumb tip
    index_tip = landmarks[8]  # Index finger tip
    middle_tip = landmarks[12] # Middle finger tip
    ring_tip = landmarks[16]   # Ring finger tip
    pinky_tip = landmarks[20]  # Pinky finger tip
    
    # Thumbs Up
    if thumb_tip.y < index_tip.y and thumb_tip.y < middle_tip.y and thumb_tip.y < ring_tip.y and thumb_tip.y < pinky_tip.y:
        return "Thumbs Up"
    
    # Thumbs Down
    elif thumb_tip.y > index_tip.y and thumb_tip.y > middle_tip.y and thumb_tip.y > ring_tip.y and thumb_tip.y > pinky_tip.y:
        return "Thumbs Down"
    
    # Victory (Peace)
    elif (index_tip.y < middle_tip.y < ring_tip.y and 
          np.linalg.norm(np.array([index_tip.x, index_tip.y]) - np.array([middle_tip.x, middle_tip.y])) > 
          np.linalg.norm(np.array([middle_tip.x, middle_tip.y]) - np.array([ring_tip.x, ring_tip.y]))):
        return "Victory"
    
    # Pointing Up
    elif index_tip.y < thumb_tip.y and index_tip.y < middle_tip.y and index_tip.y < ring_tip.y and index_tip.y < pinky_tip.y:
        return "Pointing Up"
    
    # Fist
    elif (np.linalg.norm(np.array([thumb_tip.x, thumb_tip.y]) - np.array([landmarks[0].x, landmarks[0].y])) < 0.05 and
          np.linalg.norm(np.array([index_tip.x, index_tip.y]) - np.array([landmarks[0].x, landmarks[0].y])) < 0.05 and
          np.linalg.norm(np.array([middle_tip.x, middle_tip.y]) - np.array([landmarks[0].x, landmarks[0].y])) < 0.05):
        return "Fist"
    
    return "Unknown"

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = hand_landmarks.landmark
            gesture = detect_gesture(landmarks)
            
            # Print the detected gesture
            print(f"Identified gesture: {gesture}")
            
            # Display the gesture on the video frame
            cv2.putText(image, gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    cv2.imshow('Hand Gesture Recognition', image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
