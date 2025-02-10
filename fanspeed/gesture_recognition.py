import cv2
import mediapipe as mp
import serial
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Initialize Serial Communication with Arduino
arduino = serial.Serial('/dev/cu.usbmodemXXX', 9600, timeout=1)  # Replace 'XXX' with your port number
time.sleep(2)  # Wait for Arduino to reset

def send_command(command):  
    arduino.write(f"{command}\n".encode())
    time.sleep(0.1)
    print(f"Command Sent: {command}")

def recognize_gesture(hand_landmarks):
    # Placeholder for actual gesture recognition logic
   
    return 'Gesture1'

def map_gesture(gesture):
    gesture_map = {     
        'Gesture1': 'On',
        'Gesture2':  'Off',
        'Gesture3': 'Increase Speed',
        'Gesture4': 'Decrease Speed',
        'Gesture5': 'Set Speed',
    }
    return gesture_map.get(gesture, 'Unknown')

# Capture video from the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, img = cap.read()
    if not success:
        continue

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)
            gesture = recognize_gesture(handLms)  # Recognize the gesture
            mapped_gesture = map_gesture(gesture)  # Map to the desired command
            if mapped_gesture != 'Unknown':
                send_command(mapped_gesture)

    cv2.imshow("Hand Gesture Recognition", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
