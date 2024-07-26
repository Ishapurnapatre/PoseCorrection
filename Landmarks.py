import cv2
import mediapipe as mp

# Set up the MediaPipe Pose and Hands models and the drawing utility
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5)

# Set up the camera
cap = cv2.VideoCapture(0)  # 0 for the default camera, you can use other camera indexes if available

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to RGB (MediaPipe requires RGB format)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and obtain the pose and hand landmarks
    results_pose = pose.process(rgb_frame)
    results_hands = hands.process(rgb_frame)

    # Visualize the pose landmarks on the frame
    if results_pose.pose_landmarks:
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Visualize the hand landmarks on the frame
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the captured frame
    cv2.imshow('Pose and Hand Landmarks', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
