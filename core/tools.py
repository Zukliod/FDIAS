import cv2

def extract_color_features(image, bbox):
    """Extracts color histogram from the bounding box region."""
    x1, y1, x2, y2 = bbox
    roi = image[y1:y2, x1:x2]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_roi], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def is_blurry(image, threshold=100):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold

def is_far(image, face_box, min_face_area=30000):
    x1, y1, x2, y2 = face_box
    face_area = (x2 - x1) * (y2 - y1)
    return face_area < min_face_area

def is_moving(previous_frame, current_frame, motion_threshold=60000):
    gray_prev = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    gray_curr = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    frame_diff = cv2.absdiff(gray_prev, gray_curr)
    _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
    motion_score = cv2.countNonZero(thresh)

    return motion_score > motion_threshold