# Description: This file contains the code for the webcam demo of the hand gesture recognition system.
import cv2
import mediapipe as mp

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Define a function to calculate the distance between two points
def calculate_distance(point1, point2):
    """Calculate the distance between two points"""
    x_dist = point2.x - point1.x
    y_dist = point2.y - point1.y
    z_dist = point2.z - point1.z
    return (x_dist**2 + y_dist**2 + z_dist**2)**0.5

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Initialize mediapipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize mediapipe drawing module
mp_drawing = mp.solutions.drawing_utils

print("Initialization done...")
# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Webcam not accessible!")
else:
    print("Webcam accessed successfully...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error reading frame from webcam!")
        continue

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame and get the hand landmarks
    results = hands.process(rgb_frame)

    # Check if hand(s) detected in the frame
    if results.multi_hand_landmarks:
        print(f"Detected {len(results.multi_hand_landmarks)} hands in the frame.")
        
        for hand_number, hand_landmarks in enumerate(results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # tuple of finger tip and base landmark IDs
            finger_pairs = {
                'Index': (8, 5),
                'Middle': (12, 9),
                'Ring': (16, 13),
                'Pinky': (20, 17)
            }

            all_tips_touching_bases = True  # We assume initially that all tips are touching their bases
            # define  the landmarks for each finger and check if the tips are touching their bases
            for finger, (tip_id, base_id) in finger_pairs.items():
                tip = hand_landmarks.landmark[tip_id]
                base = hand_landmarks.landmark[base_id]

                # Check if the tip is below the base
                if calculate_distance(tip, base) >= 0.05:  # Adjust the threshold if needed
                    all_tips_touching_bases = False
                    break

            
            thumb_tip = hand_landmarks.landmark[4]  # Thumb tip
            index_tip = hand_landmarks.landmark[8]  # Index finger tip

            # Displaying open/close status of each finger
            finger_tips = [4, 8, 12, 16, 20]
            open_fingers = []

            for tip in finger_tips:
                if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip-2].y:
                    open_fingers.append(True)
                else:
                    open_fingers.append(False)

            finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
            finger_status = dict(zip(finger_names, open_fingers))

            #check if thumb and index are touching
            if calculate_distance(thumb_tip, index_tip) < 0.05:  # You may need to adjust this threshold
                print("Thumb and Index are touching!")
                cv2.putText(frame, "OKAY!", (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            #Check if 'A' condition is met
            if all_tips_touching_bases:
                print("All finger tips are touching their bases!")
                cv2.putText(frame, "A", (400, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Check if 'U' condition is met
            if (finger_status['Index'] and finger_status['Middle']) and (not finger_status['Ring'] and not finger_status['Pinky']):
                cv2.putText(frame, "U", (300, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            
            y_offset = 20 + (hand_number * 100)  # Change '100' if you want more or less spacing between hands

            # Displaying open/close status of each finger
            for finger, status in finger_status.items():
                status_text = f"{finger}: {'Open' if status else 'Closed'}"
                color = (0, 255, 0) if status else (0, 0, 255)  # green for open, red for closed
                cv2.putText(frame, status_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                y_offset += 20
                
    cv2.imshow('Test', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#Close the webcam and destroy all windows
cap.release()
cv2.destroyAllWindows()
