import os
import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

def process_video(video_path, detection=False):
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic

    video_path = video_path
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

            # Collect data for face landmarks
            if results.face_landmarks:
                for idx, landmark in enumerate(results.face_landmarks.landmark):
                    all_frames_data.append({
                        'frame': frame_number,
                        'row_id': f'{frame_number}-face-{idx}',
                        'type': 'face',
                        'landmark_index': idx,
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z if landmark.HasField('z') else None
                    })

            # Collect data for left hand landmarks
            if results.left_hand_landmarks:
                for idx, landmark in enumerate(results.left_hand_landmarks.landmark):
                    all_frames_data.append({
                        'frame': frame_number,
                        'row_id': f'{frame_number}-left_hand-{idx}',
                        'type': 'left_hand',
                        'landmark_index': idx,
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z if landmark.HasField('z') else None
                    })

            # Collect data for pose landmarks
            if results.pose_landmarks:
                for idx, landmark in enumerate(results.pose_landmarks.landmark):
                    all_frames_data.append({
                        'frame': frame_number,
                        'row_id': f'{frame_number}-pose-{idx}',
                        'type': 'pose',
                        'landmark_index': idx,
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z if landmark.HasField('z') else None
                    })

            # Collect data for right hand landmarks
            if results.right_hand_landmarks:
                for idx, landmark in enumerate(results.right_hand_landmarks.landmark):
                    all_frames_data.append({
                        'frame': frame_number,
                        'row_id': f'{frame_number}-right_hand-{idx}',
                        'type': 'right_hand',
                        'landmark_index': idx,
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z if landmark.HasField('z') else None
                    })

            if detection == True:
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

                # Increment the frame number
                frame_number += 1

            else:

                # Increment the frame number
                frame_number += 1

            # Exit on 'q' key press
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    # Convert the list of frames data to a pandas DataFrame
    df_landmarks = pd.DataFrame(all_frames_data)

    # Define the order of landmark types
    landmark_types = ['face', 'left_hand', 'pose', 'right_hand']

    # Initialize a list to hold the reordered rows
    reordered_rows = []

    # Iterate over the landmark types in the specified order
    for landmark_type in landmark_types:
        # Select rows that belong to the current landmark type
        type_rows = df_landmarks[df_landmarks['type'].str.contains(landmark_type)]

        # Append the selected rows to the reordered list
        reordered_rows.extend(type_rows.to_dict('records'))

    # Create a new DataFrame with the reordered rows
    df_landmarks_reordered = pd.DataFrame(reordered_rows)

    # Now df_landmarks_reordered is a DataFrame with rows ordered by landmark type
    print(df_landmarks_reordered)

    # Write the DataFrame to a Parquet file
    table = pa.Table.from_pandas(df_landmarks_reordered)
    pq.write_table(table, 'test_case.parquet')

    return 'The video has been successfully processed', df_landmarks_reordered


process_video('SnapSave.io-WARM -[warmth, heat]-(1080p).mp4', True)
