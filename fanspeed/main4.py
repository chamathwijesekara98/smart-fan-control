import cv2
import mediapipe as mp
import serial
import time

from main import detect_gesture

# Set up serial communication with Arduino
arduino = serial.Serial(port='/dev/cu.usbmodem11301', baudrate=9600, timeout=.1)  # Adjust the port based on your OS
time.sleep(2)  # Give Arduino time to reset

def send_command(command):
    arduino.write(bytes(command, 'utf-8'))
    time.sleep(0.05)

# Initialize MediaPipe Hand Model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Initialize Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB for hand detection
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Detect gestures here (simplified example)
            gesture = detect_gesture(hand_landmarks)  # Assuming you have a detect_gesture function

            # Mapping gestures to commands
            if gesture == 'Thumbs Up':  # Turn fan ON
                send_command('O')
                print("Turn fan ON")
            elif gesture == 'Thumbs Down':  # Turn fan OFF
                send_command('F')
                print("Turn fan OFF")
            elif gesture == 'Victory':  # Increase fan speed
                send_command('I')
                print("Increase fan speed")
            elif gesture == 'Fist':  # Decrease fan speed
                send_command('D')
                print("Decrease fan speed")
            elif gesture == 'Pointing Up':  # Set speed
                speed = 150  # Example speed
                send_command(f'S{speed}')
                print(f"Set fan speed to {speed}")

    # Display the frame
    cv2.imshow('Hand Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
