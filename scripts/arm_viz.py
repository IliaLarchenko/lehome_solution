#!/usr/bin/env python3
"""SO101 arm side-view drawing library for the DAgger UI.

Renders the robot arm state as a 2D side-view stick figure with joints.
Imported by dagger_collect.py.
"""
import math
import numpy as np
import cv2

# SO101 link lengths (meters, from URDF)
L_BASE = 0.062    # base to shoulder height
L_UPPER = 0.116   # shoulder to elbow
L_FOREARM = 0.135  # elbow to wrist
L_HAND = 0.065     # wrist to gripper tip

# Table height reference — the base sits on the table
TABLE_Y = 0.0

# Joint limits in radians (from USD_JOINT_LIMITS in degrees)
JOINT_LIMITS_RAD = {
    "shoulder_pan": (-110 * math.pi / 180, 110 * math.pi / 180),
    "shoulder_lift": (-100 * math.pi / 180, 100 * math.pi / 180),
    "elbow_flex": (-100 * math.pi / 180, 90 * math.pi / 180),
    "wrist_flex": (-95 * math.pi / 180, 95 * math.pi / 180),
    "wrist_roll": (-160 * math.pi / 180, 160 * math.pi / 180),
    "gripper": (-10 * math.pi / 180, 100 * math.pi / 180),
}


def arm_fk_side_view(joints_rad: np.ndarray) -> list[tuple[float, float]]:
    """Compute 2D side-view joint positions for one SO101 arm.

    Args:
        joints_rad: 6 joint angles in radians
            [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]

    Returns:
        List of (x, y) positions in meters: [base, shoulder, elbow, wrist, gripper_tip]
        x = horizontal distance from base (forward), y = height above table.
    """
    pan, lift, elbow, wrist_f, wrist_r, grip = joints_rad

    # Horizontal projection factor from shoulder_pan
    # When pan=0 the arm points straight forward in the view plane
    # pan rotates left/right so we see cos(pan) of the reach
    h_proj = math.cos(pan)

    # Base position (on table)
    base = (0.0, 0.0)
    # Shoulder is above the base
    shoulder = (0.0, L_BASE)

    # Cumulative angle in the side-view plane
    # angle=0 means pointing right (horizontal forward), angle=pi/2 means up
    # Real robot: at lift=0 the upper arm points UP (vertical), not forward
    # So we start at pi/2 and lift rotates from there
    angle = -math.pi / 2 + lift

    # Shoulder -> Elbow
    ex = shoulder[0] + L_UPPER * math.cos(angle) * h_proj
    ey = shoulder[1] - L_UPPER * math.sin(angle)
    elbow_pos = (ex, ey)

    # Cancel the -pi/2 offset so elbow/wrist deltas stay correct
    # (user confirmed these segments look right with the original angles)
    angle += math.pi / 2

    # Elbow flex: positive = bends inward (forearm folds toward upper arm)
    angle += elbow

    # Elbow -> Wrist
    wx = elbow_pos[0] + L_FOREARM * math.cos(angle) * h_proj
    wy = elbow_pos[1] - L_FOREARM * math.sin(angle)
    wrist_pos = (wx, wy)

    # Wrist flex: direction flipped compared to elbow
    angle += wrist_f

    # Wrist -> Gripper tip
    gx = wrist_pos[0] + L_HAND * math.cos(angle) * h_proj
    gy = wrist_pos[1] - L_HAND * math.sin(angle)
    grip_pos = (gx, gy)

    return [base, shoulder, elbow_pos, wrist_pos, grip_pos]


def draw_arm_side_view(
    img: np.ndarray,
    joints_rad: np.ndarray,
    region: tuple[int, int, int, int],  # (x, y, w, h) in pixels
    label: str = "",
    gripper_open: float = 0.0,  # 0=closed, 1=fully open
    flip: bool = False,  # horizontally mirror (for right arm)
) -> np.ndarray:
    """Draw a side-view schematic of one SO101 arm.

    Args:
        img: Image to draw on (modified in place)
        joints_rad: 6 joint angles in radians
        region: (x, y, w, h) pixel region to draw in
        label: "LEFT" or "RIGHT"
        gripper_open: gripper openness 0-1
        flip: if True, mirror horizontally (right arm faces left)
    """
    rx, ry, rw, rh = region

    # Compute FK
    positions = arm_fk_side_view(joints_rad)

    # Find bounds for auto-scaling
    max_reach = L_UPPER + L_FOREARM + L_HAND + 0.02  # ~0.32m + margin
    view_x_min = -0.05
    view_x_max = max_reach + 0.02
    view_y_min = -0.15  # below table to show when gripper goes low
    view_y_max = max_reach

    # Pixel mapping
    margin = 8
    draw_w = rw - 2 * margin
    draw_h = rh - 2 * margin
    x_scale = draw_w / (view_x_max - view_x_min)
    y_scale = draw_h / (view_y_max - view_y_min)
    scale = min(x_scale, y_scale)

    def to_px(x_m: float, y_m: float) -> tuple[int, int]:
        px_raw = int((x_m - view_x_min) * scale)
        if flip:
            px_raw = draw_w - px_raw
        px = rx + margin + px_raw
        py = ry + margin + int((view_y_max - y_m) * scale)
        return (px, py)

    # Background
    cv2.rectangle(img, (rx, ry), (rx + rw, ry + rh), (25, 25, 30), -1)

    # Table surface line
    table_left = to_px(view_x_min, TABLE_Y)
    table_right = to_px(view_x_max, TABLE_Y)
    cv2.line(img, table_left, table_right, (80, 60, 40), 2)

    # Grid lines for height reference (every 5cm)
    for h_cm in range(-10, 35, 5):
        h_m = h_cm / 100.0
        p1 = to_px(view_x_min, h_m)
        p2 = to_px(view_x_max, h_m)
        color = (50, 50, 50) if h_cm != 0 else (80, 60, 40)
        cv2.line(img, p1, p2, color, 1)
        if h_cm % 10 == 0 and h_cm != 0:
            cv2.putText(img, f"{h_cm}cm", (rx + 2, p1[1] - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.25, (80, 80, 80), 1)

    # Draw arm segments
    colors = [
        (100, 100, 100),  # base post (gray)
        (70, 180, 255),   # upper arm (orange)
        (70, 255, 70),    # forearm (green)
        (255, 100, 100),  # hand (blue)
    ]
    thickness = [3, 3, 3, 2]
    joint_names = ["base", "shoulder", "elbow", "wrist", "gripper"]

    for i in range(len(positions) - 1):
        p1 = to_px(*positions[i])
        p2 = to_px(*positions[i + 1])
        cv2.line(img, p1, p2, colors[i], thickness[i])

    # Draw joints as circles
    for i, (name, pos) in enumerate(zip(joint_names, positions)):
        px = to_px(*pos)
        if name == "base":
            # Base: small square on table
            cv2.rectangle(img, (px[0] - 4, px[1] - 2), (px[0] + 4, px[1] + 2),
                          (150, 150, 150), -1)
        elif name == "gripper":
            # Gripper tip: small diamond
            pts = np.array([
                [px[0], px[1] - 4],
                [px[0] + 4, px[1]],
                [px[0], px[1] + 4],
                [px[0] - 4, px[1]],
            ], dtype=np.int32)
            color = (100, 255, 100) if gripper_open > 0.3 else (100, 100, 255)
            cv2.fillPoly(img, [pts], color)
        else:
            # Regular joint: filled circle
            cv2.circle(img, px, 4, (200, 200, 200), -1)
            cv2.circle(img, px, 4, (100, 100, 100), 1)

    # Gripper jaws — one fixed, one moving (connected to opposite sides of last motor)
    if len(positions) >= 5:
        wrist_pos = positions[3]
        grip_pos = positions[4]
        # Direction from wrist to gripper (forward axis of hand)
        dx = grip_pos[0] - wrist_pos[0]
        dy = grip_pos[1] - wrist_pos[1]
        length = math.sqrt(dx * dx + dy * dy) + 1e-6
        # Unit forward and perpendicular vectors
        fx, fy = dx / length, dy / length
        nx, ny = -fy, fx  # perpendicular
        finger_len = 0.025  # 2.5cm fingers
        jaw_color = (100, 255, 100) if gripper_open > 0.3 else (100, 100, 255)

        # Fixed jaw — extends straight forward from grip_pos (one side)
        fixed_start = grip_pos
        fixed_end = (grip_pos[0] + fx * finger_len, grip_pos[1] + fy * finger_len)
        cv2.line(img, to_px(*fixed_start), to_px(*fixed_end), jaw_color, 2)

        # Moving jaw — on opposite side, angle depends on gripper_open
        # When closed (0): parallel to fixed jaw; when open (1): splayed 90 degrees
        jaw_angle = gripper_open * (math.pi / 2)  # max 90 degrees open
        # Rotate forward direction by jaw_angle (opposite side)
        mjx = fx * math.cos(jaw_angle) + nx * math.sin(jaw_angle)
        mjy = fy * math.cos(jaw_angle) + ny * math.sin(jaw_angle)
        moving_end = (grip_pos[0] + mjx * finger_len, grip_pos[1] + mjy * finger_len)
        cv2.line(img, to_px(*grip_pos), to_px(*moving_end), jaw_color, 2)

    # Gripper height indicator — dotted line from gripper to table
    grip_px = to_px(*positions[4])
    table_below = to_px(positions[4][0], TABLE_Y)
    grip_height_cm = positions[4][1] * 100
    for y in range(min(grip_px[1], table_below[1]), max(grip_px[1], table_below[1]), 4):
        cv2.circle(img, (grip_px[0], y), 1, (100, 100, 150), -1)
    # Height label
    h_text = f"{grip_height_cm:.0f}cm"
    cv2.putText(img, h_text, (grip_px[0] + 5, grip_px[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 255), 1)

    # Label
    if label:
        cv2.putText(img, label, (rx + 4, ry + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

    return img
