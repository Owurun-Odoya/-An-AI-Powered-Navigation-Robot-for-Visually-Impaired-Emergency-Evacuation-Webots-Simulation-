from controller import Robot
import numpy as np
import cv2
import pytesseract
from difflib import SequenceMatcher

# --------------------------------------------------
# Windows Tesseract path
# --------------------------------------------------
pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"

# --------------------------------------------------
# Constants
# --------------------------------------------------
TIME_STEP = 64
MAX_SPEED = 6.28

# Wall-follow tuning
FRONT_BLOCK_THRESHOLD = 80
SIDE_TARGET = 70
WALL_FOLLOW_GAIN = 0.003

# Exit approach tuning
EXIT_CONFIRM_FRAMES = 3

# Increase this so it doesn't stop in the hallway
STOP_SIGN_AREA = 180000

# Require the sign to be centered before stopping
STOP_CENTER_TOLERANCE = 35

APPROACH_BASE_SPEED = 2.0
CENTERING_GAIN = 0.004

# Final safety stop if extremely close to wall/sign
FRONT_HARD_STOP = 120

# --------------------------------------------------
# Robot init
# --------------------------------------------------
robot = Robot()

# Motors
left_motor = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")
left_motor.setPosition(float("inf"))
right_motor.setPosition(float("inf"))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

# Camera
camera = robot.getDevice("camera")
camera.enable(TIME_STEP)

# Speaker
speaker = robot.getDevice("speaker")
try:
    speaker.setEngine("microsoft")
    speaker.setLanguage("en-UK")
except Exception:
    speaker.setEngine("pico")
    speaker.setLanguage("en-UK")

# Proximity sensors ps0..ps7
ps = []
for i in range(8):
    sensor = robot.getDevice(f"ps{i}")
    sensor.enable(TIME_STEP)
    ps.append(sensor)

# --------------------------------------------------
# State variables
# --------------------------------------------------
state = "SEARCH"
previous_state = "SEARCH"
exit_votes = 0
frame_count = 0
announced_exit = False
stop_steps = 0

# --------------------------------------------------
# Utility functions
# --------------------------------------------------
def clamp(value, low, high):
    return max(low, min(high, value))

def set_speed(left, right):
    left_motor.setVelocity(clamp(left, -MAX_SPEED, MAX_SPEED))
    right_motor.setVelocity(clamp(right, -MAX_SPEED, MAX_SPEED))

def get_proximity_values():
    return [sensor.getValue() for sensor in ps]

def get_camera_frame_bgr():
    raw = camera.getImage()
    if raw is None:
        return None

    width = camera.getWidth()
    height = camera.getHeight()

    # Webots camera image is BGRA
    frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 4))
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame_bgr

def say(text):
    try:
        speaker.speak(text, 1.0)
    except Exception as e:
        print("Speak failed:", e)

# --------------------------------------------------
# EXIT sign detection
# --------------------------------------------------
def detect_exit_text(frame_bgr):
    h, w = frame_bgr.shape[:2]

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    lower_green = np.array([40, 60, 60])
    upper_green = np.array([95, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return {
            "exit_seen": False,
            "text": "",
            "score": 0.0,
            "bbox": None,
            "area": 0,
            "center_error": 0.0
        }

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)

    if area < 500:
        return {
            "exit_seen": False,
            "text": "",
            "score": 0.0,
            "bbox": None,
            "area": area,
            "center_error": 0.0
        }

    x, y, bw, bh = cv2.boundingRect(largest)

    sign_center_x = x + bw / 2.0
    image_center_x = w / 2.0
    center_error = sign_center_x - image_center_x

    pad = 5
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w, x + bw + pad)
    y2 = min(h, y + bh + pad)

    roi = frame_bgr[y1:y2, x1:x2]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    _, proc = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

    config = '--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    text = pytesseract.image_to_string(proc, config=config).strip()

    normalized = ''.join(ch for ch in text.upper() if ch.isalpha())
    score = SequenceMatcher(None, normalized, "EXIT").ratio()

    exit_seen = (
        normalized == "EXIT" or
        "EXIT" in normalized or
        score >= 0.72
    )

    return {
        "exit_seen": exit_seen,
        "text": normalized,
        "score": score,
        "bbox": (x, y, bw, bh),
        "area": area,
        "center_error": center_error
    }

# --------------------------------------------------
# Navigation
# --------------------------------------------------
def wall_follow_right(ps_vals):
    """
    e-puck proximity order:
    ps0 right front
    ps1 right-front diagonal
    ps2 right side
    ps3 right back
    ps4 left back
    ps5 left side
    ps6 left-front diagonal
    ps7 left front
    """
    front = max(ps_vals[0], ps_vals[1], ps_vals[6], ps_vals[7])
    right_side = max(ps_vals[1], ps_vals[2])
    left_side = max(ps_vals[5], ps_vals[6])

    if front > FRONT_BLOCK_THRESHOLD:
        if left_side < right_side:
            return -2.0, 2.0
        else:
            return 2.0, -2.0

    error = right_side - SIDE_TARGET
    correction = WALL_FOLLOW_GAIN * error

    base = 3.5
    left_speed = base + correction
    right_speed = base - correction
    return left_speed, right_speed

def approach_exit(sign_info, ps_vals):
    """
    Move toward the sign.
    Only stop when:
      - sign is big enough
      - sign is centered enough
    Front sensor is only a final safety check.
    """
    front = max(ps_vals[0], ps_vals[1], ps_vals[6], ps_vals[7])
    area = sign_info["area"]
    center_error = sign_info["center_error"]

    # True stop condition: close and centered
    if area >= STOP_SIGN_AREA and abs(center_error) <= STOP_CENTER_TOLERANCE:
        return 0.0, 0.0, True

    # Safety stop only if extremely close AND reasonably aligned
    if front > FRONT_HARD_STOP and abs(center_error) <= STOP_CENTER_TOLERANCE and area >= STOP_SIGN_AREA * 0.7:
        return 0.0, 0.0, True

    # Steering toward sign
    steering = CENTERING_GAIN * center_error

    left_speed = APPROACH_BASE_SPEED + steering
    right_speed = APPROACH_BASE_SPEED - steering

    return left_speed, right_speed, False

# --------------------------------------------------
# Main loop
# --------------------------------------------------
print("robot_guide controller started")

while robot.step(TIME_STEP) != -1:
    ps_vals = get_proximity_values()
    frame_bgr = get_camera_frame_bgr()

    sign_info = {
        "exit_seen": False,
        "text": "",
        "score": 0.0,
        "bbox": None,
        "area": 0,
        "center_error": 0.0
    }

    if frame_bgr is not None:
        sign_info = detect_exit_text(frame_bgr)

        if sign_info["exit_seen"]:
            exit_votes += 1
        else:
            exit_votes = max(0, exit_votes - 1)

        print(
            f"STATE={state} | OCR='{sign_info['text']}' | "
            f"score={sign_info['score']:.2f} | area={sign_info['area']:.0f} | "
            f"center_error={sign_info['center_error']:.1f} | "
            f"votes={exit_votes}"
        )

    # ---------------------------
    # State transitions
    # ---------------------------
    if state == "SEARCH":
        if exit_votes >= EXIT_CONFIRM_FRAMES:
            state = "APPROACH"
            print("Exit detected. Approaching sign.")

    if state != previous_state:
        print(f"STATE CHANGED: {previous_state} -> {state}")
        if state == "STOP":
            stop_steps = 0
        previous_state = state

    # ---------------------------
    # State actions
    # ---------------------------
    if state == "SEARCH":
        left_speed, right_speed = wall_follow_right(ps_vals)
        set_speed(left_speed, right_speed)

    elif state == "APPROACH":
        left_speed, right_speed, reached = approach_exit(sign_info, ps_vals)
        set_speed(left_speed, right_speed)
        if reached:
            state = "STOP"

    elif state == "STOP":
        set_speed(0.0, 0.0)
        stop_steps += 1

        print(f"STOP state active. stop_steps={stop_steps}")

        if not announced_exit and stop_steps >= 10:
            say("Exit reached.")
            announced_exit = True

        print("Exit reached. Robot stopped.")

    frame_count += 1
