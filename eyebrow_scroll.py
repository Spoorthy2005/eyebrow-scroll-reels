import cv2
import mediapipe as mp
import pyautogui

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Webcam feed
cap = cv2.VideoCapture(0)

# Eyebrow landmark indexes
left_eyebrow = [70, 63, 105, 66, 107]
right_eyebrow = [336, 296, 334, 293, 300]
eyebrow_pairs = list(zip(left_eyebrow, left_eyebrow[1:])) + list(zip(right_eyebrow, right_eyebrow[1:]))

# Eyebrow baseline height to compare
baseline = None
scroll_threshold = 5  # Adjust for sensitivity

while True:
    success, frame = cap.read()
    if not success:
        break

    # Flip and convert to RGB
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = face_mesh.process(rgb)

    eyebrow_color = (255, 255, 255)  # Default: white

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            h, w, _ = frame.shape

            # Extract eyebrow points
            points = {}
            for idx in left_eyebrow + right_eyebrow:
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)
                points[idx] = (x, y)

            # Compute average eyebrow height
            left_avg = sum(points[i][1] for i in left_eyebrow) / len(left_eyebrow)
            right_avg = sum(points[i][1] for i in right_eyebrow) / len(right_eyebrow)
            avg_height = (left_avg + right_avg) / 2

            if baseline is None:
                baseline = avg_height

            # Compare with baseline
            if avg_height < baseline - scroll_threshold:
                pyautogui.scroll(-30)  # Scroll down
                action = "Eyebrows raised - Scrolling down"
                eyebrow_color = (0, 255, 0)  # Green
            elif avg_height > baseline + scroll_threshold:
                pyautogui.scroll(30)  # Scroll up
                action = "Eyebrows lowered - Scrolling up"
                eyebrow_color = (0, 0, 255)  # Red
            else:
                action = "Neutral"

            # Draw eyebrow points
            for idx in left_eyebrow + right_eyebrow:
                cv2.circle(frame, points[idx], 3, eyebrow_color, -1)

            # Connect with lines
            for p1, p2 in eyebrow_pairs:
                cv2.line(frame, points[p1], points[p2], eyebrow_color, 1)

            # Display action
            cv2.putText(frame, action, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, eyebrow_color, 2)

    # Show frame
    cv2.imshow("Eyebrow Scroll Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
