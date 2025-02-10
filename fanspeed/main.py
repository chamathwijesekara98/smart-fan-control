import cv2
import mediapipe as mp
import serial
import time

# Initialize MediaPipe Hand Model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    
    static_image_mode=False,       # Video stream; set to False
    max_num_hands=1,               # Maximum number of hands to detect
    min_detection_confidence=0.7,  # Minimum confidence for detection
    min_tracking_confidence=0.7    # Minimum confidence for tracking
)
mp_drawing = mp.solutions.drawing_utils

# Function to check if a finger is open
def is_finger_open(landmarks, finger_tip_idx, finger_dip_idx):
    return landmarks[finger_tip_idx].y < landmarks[finger_dip_idx].y

# Define Gestures and map them to commands
def detect_gesture(landmarks):
    thumb_open = is_finger_open(landmarks, 4, 2)    # Thumb (Landmark 4, Landmark 2)
    index_open = is_finger_open(landmarks, 8, 6)    # Index finger (Landmark 8, Landmark 6)
    middle_open = is_finger_open(landmarks, 12, 10) # Middle finger (Landmark 12, Landmark 10)
    ring_open = is_finger_open(landmarks, 16, 14)   # Ring finger (Landmark 16, Landmark 14)
    pinky_open = is_finger_open(landmarks, 20, 18)  # Pinky finger (Landmark 20, Landmark 18)
    
    thumb_tip = landmarks[4].y   # y-coordinate of thumb tip
    thumb_base = landmarks[2].y  # y-coordinate of thumb base

    # Define gestures and map them to commands
    if thumb_open and not index_open and not middle_open and not ring_open and not pinky_open and thumb_tip < thumb_base:
        return 'increase'  # "Thumbs Up" gesture -> Increase Speed
    elif  thumb_open and  index_open and  middle_open and not ring_open and not pinky_open:
        return 'off'        # "victory" gesture -> Turn Fan Off
    elif thumb_open and not index_open and not middle_open and not ring_open and not pinky_open and thumb_tip > thumb_base:
        return 'dicrease'  # Thumbs Down gesture -> Turn Fan Off
    elif index_open and middle_open and ring_open and pinky_open and thumb_open:
        return 'set_speed'  # "Open Hand" gesture -> Set Fan Speed
    elif index_open and not middle_open and not ring_open and not pinky_open:
        return 'on'         # "Pointing" gesture -> Turn Fan On
    else:
        return 'unknown'
    

# Setup serial communication (adjust the port and baud rate accordingly)
try:
    # arduino = serial.Serial(port='COM9', baudrate=9600, timeout=.1)  #Windows
    arduino = serial.Serial(port='/dev/cu.usbmodem11401', baudrate=9600, timeout=.1)  #Mac
    time.sleep(2)  # Wait for the serial connection to initialize
    print("Serial connection established.")
except serial.SerialException:
    print("Could not open serial port. Please check the port name and try again.")
    arduino = None

def send_command(command):
    if arduino:
        command_str = command + '\n'  # Adding newline as a delimiter
        arduino.write(command_str.encode('utf-8'))
        print(f"__________________________Sent command: {command}")
    else:
        print("Arduino serial connection not established.")

# Start webcam feed
cap = cv2.VideoCapture(0)

# To prevent multiple rapid command sends, keep track of the last sent command
last_command = None
command_delay = 2  # seconds
last_command_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    # Convert image to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame to detect hand gestures
    results = hands.process(image_rgb)

    gesture = 'No Hand Detected'

    # Draw landmarks and detect gestures
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )

            # Detect hand gesture and get corresponding command
            gesture = detect_gesture(hand_landmarks.landmark)
            print("Detected Gesture:", gesture)

            # Get bounding box for placing the gesture text
            h, w, _ = frame.shape
            x_coords = [landmark.x for landmark in hand_landmarks.landmark]
            y_coords = [landmark.y for landmark in hand_landmarks.landmark]
            xmin, xmax = int(min(x_coords) * w), int(max(x_coords) * w)
            ymin, ymax = int(min(y_coords) * h), int(max(y_coords) * h)

            # Define position for the gesture text
            text_position = (xmin, ymin - 10 if ymin - 10 > 10 else ymin + 10)

            # Display the gesture name on the frame
            cv2.putText(frame, gesture, text_position, cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 0, 0), 2, cv2.LINE_AA)

            # Send the command to Arduino with a delay to prevent flooding
            current_time = time.time()
            if gesture != 'unknown' and gesture != 'No Hand Detected':
                if gesture != last_command or (current_time - last_command_time) > command_delay:
                    send_command(gesture)
                    last_command = gesture
                    last_command_time = current_time

    else:
        # If no hand is detected, you can choose to display a message or do nothing
        gesture = 'No Hand Detected'
        cv2.putText(frame, gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow("Hand Gesture Recognition", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
if arduino:
    arduino.close()


