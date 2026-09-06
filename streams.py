import cv2
import numpy as np

# Standardized size for each individual stream box
TARGET_WIDTH = 640
TARGET_HEIGHT = 360

def create_2x2_grid(frame1, frame2, frame3, frame4):
    """Resizes 4 frames and stacks them into a 2x2 grid."""
    
    # 1. Resize all frames to matching dimensions
    f1 = cv2.resize(frame1, (TARGET_WIDTH, TARGET_HEIGHT))
    f2 = cv2.resize(frame2, (TARGET_WIDTH, TARGET_HEIGHT))
    f3 = cv2.resize(frame3, (TARGET_WIDTH, TARGET_HEIGHT))
    f4 = cv2.resize(frame4, (TARGET_WIDTH, TARGET_HEIGHT))
    # 3. Stack frames horizontally and then vertically
    top_row = np.hstack((f1, f2))
    bottom_row = np.hstack((f3, f4))
    grid = np.vstack((top_row, bottom_row))

    return grid

# --- Example Usage in a Loop ---
cap1 = cv2.VideoCapture("video1.mp4")
cap2 = cv2.VideoCapture("video2.mp4")
cap3 = cv2.VideoCapture("video3.mp4")
cap4 = cv2.VideoCapture("video4.mp4")

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    ret3, frame3 = cap3.read()
    ret4, frame4 = cap4.read()

    # Loop videos back to start if any stream reaches the end
    if not ret1: cap1.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
    if not ret2: cap2.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
    if not ret3: cap3.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
    if not ret4: cap4.set(cv2.CAP_PROP_POS_FRAMES, 0); continue

    # Combine into a single frame (1280x720 total output window)
    grid_display = create_2x2_grid(frame1, frame2, frame3, frame4)

    # Display the merged single window
    cv2.imshow("Multi-cam View", grid_display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
cap2.release()
cap3.release()
cap4.release()
cv2.destroyAllWindows()