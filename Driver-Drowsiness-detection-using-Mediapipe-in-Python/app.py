import cv2
import time
import os
from drowsy_detection import VideoFrameHandler
from audio_handling import AudioFrameHandler

alarm_file_path = os.path.join("audio", "wake_up.wav")
# Initialize the VideoFrameHandler class
video_frame_handler = VideoFrameHandler()
audio_handler = AudioFrameHandler(sound_file_path=alarm_file_path)

# Thresholds
EAR_THRESH = 0.18
WAIT_TIME = 1.0

# Start the video capture
cap = cv2.VideoCapture(0)

# Process each frame
while True:
    # Capture a frame
    ret, frame = cap.read()

    # If frame is not captured, break the loop
    if not ret:
        break

    # Process the frame
    frame, play_alarm = video_frame_handler.process(frame, {"EAR_THRESH": EAR_THRESH, "WAIT_TIME": WAIT_TIME})

    # Display the frame
    cv2.imshow("Drowsiness Detection", frame)

    # If alarm should be played, play it
    if play_alarm:
        audio_handler
        # ...

    # Press `q` to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture
cap.release()

# Destroy all windows
cv2.destroyAllWindows()

