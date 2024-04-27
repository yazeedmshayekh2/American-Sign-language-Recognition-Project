import os
import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

video_path = os.path.join('SnapSave.io-car-CRASH -[car accident, car wreck]-(1080p).mp4')
if not os.path.exists(video_path):
    print(f"Video file {video_path} does not exist.")
    exit(1)

cap = cv2.VideoCapture(video_path)

frame_number = 0

# Initialize a list to hold all frames' landmarks data
all_frames_data = []

# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            print("Failed to capture frame or end of video reached.")
            break

        # Make Detections
        results = holistic.process(frame)

        # Dictionaries to store x, y, and z coordinates separately
        x_data = {}
        y_data = {}
        z_data = {}

        # Collect data for face landmarks
        if results.face_landmarks:
            for idx, landmark in enumerate(results.face_landmarks.landmark):
                x_data[f'x_face_{idx}'] = landmark.x
                y_data[f'y_face_{idx}'] = landmark.y
                z_data[f'z_face_{idx}'] = landmark.z

        # Collect data for right hand landmarks
        if results.right_hand_landmarks:
            for idx, landmark in enumerate(results.right_hand_landmarks.landmark):
                x_data[f'x_right_hand_{idx}'] = landmark.x
                y_data[f'y_right_hand_{idx}'] = landmark.y
                z_data[f'z_right_hand_{idx}'] = landmark.z

        # Collect data for left hand landmarks
        if results.left_hand_landmarks:
            for idx, landmark in enumerate(results.left_hand_landmarks.landmark):
                x_data[f'x_left_hand_{idx}'] = landmark.x
                y_data[f'y_left_hand_{idx}'] = landmark.y
                z_data[f'z_left_hand_{idx}'] = landmark.z

        # Collect data for pose landmarks
        if results.pose_landmarks:
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                x_data[f'x_pose_{idx}'] = landmark.x
                y_data[f'y_pose_{idx}'] = landmark.y
                z_data[f'z_pose_{idx}'] = landmark.z

        # Create a dictionary for the current frame's landmarks and update with collected data
        frame_data = {'Frame_number': frame_number}
        frame_data.update(x_data)
        frame_data.update(y_data)
        frame_data.update(z_data)

        # Add the frame's landmark data to the list
        all_frames_data.append(frame_data)

        # Increment the frame number
        frame_number += 1

        # Draw landmarks
        if results.face_landmarks:
            mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        # Display the annotated frame
        cv2.imshow('Raw Webcam Feed', frame)

        # Exit on 'q' key press
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# Convert the list of frames data to a pandas DataFrame
df_landmarks = pd.DataFrame(all_frames_data)

columns_ordered = ['Frame_number'] + \
                  [col for col in df_landmarks.columns if 'x_face' in col] + \
                  [col for col in df_landmarks.columns if 'y_face' in col] + \
                  [col for col in df_landmarks.columns if 'z_face' in col] + \
                  [col for col in df_landmarks.columns if 'x_left_hand' in col] + \
                  [col for col in df_landmarks.columns if 'y_left_hand' in col] + \
                  [col for col in df_landmarks.columns if 'z_left_hand' in col] + \
                  [col for col in df_landmarks.columns if 'x_pose' in col] + \
                  [col for col in df_landmarks.columns if 'y_pose' in col] + \
                  [col for col in df_landmarks.columns if 'z_pose' in col] + \
                  [col for col in df_landmarks.columns if 'x_right_hand' in col] + \
                  [col for col in df_landmarks.columns if 'y_right_hand' in col] + \
                  [col for col in df_landmarks.columns if 'z_right_hand' in col]

df_landmarks = df_landmarks[columns_ordered]

df_landmarks['Sequence_ID'] = 1

df_landmarks.set_index('Sequence_ID', inplace=True)

# Now df_landmarks is a DataFrame where each row represents a frame's landmarks
print(df_landmarks)

table = pa.Table.from_pandas(df_landmarks)

pq.write_table(table, 'test_case.parquet')
