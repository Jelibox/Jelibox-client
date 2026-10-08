"""
Class selector window handling
"""
import cv2
import numpy as np
from .config import CLASSLIST, CLASS_HEIGHT, CLASS_WINDOW_H, CLASS_WINDOW_W, state

def class_mouse_event(event, x, y, flags, param):
    """Handle mouse events on class selector window"""
    MAX_PER_COL = 10
    col_width = 220
    total_cols = (len(CLASSLIST) + MAX_PER_COL - 1) // MAX_PER_COL

    if event == cv2.EVENT_LBUTTONDOWN:
        if y >= CLASS_WINDOW_H:
            return

        col = max(0, min(x // col_width, total_cols - 1))
        if col >= total_cols:
            return

        start_row = state.scroll_offset // CLASS_HEIGHT
        row_local = y // CLASS_HEIGHT

        if row_local < 0 or row_local >= MAX_PER_COL:
            return

        absolute_row = start_row + row_local
        if absolute_row < 0 or absolute_row >= MAX_PER_COL:
            return

        idx = col * MAX_PER_COL + absolute_row

        if 0 <= idx < len(CLASSLIST):
            local_x = x - col * col_width

            # Click on eye icon
            if local_x < 30:
                cls = CLASSLIST[idx]
                state.visible_class[cls] = not state.visible_class[cls]
                print(f"[INFO] Toggle visibility {cls}: {state.visible_class[cls]}")
                return

            # Click on class name
            state.current_class = CLASSLIST[idx]
            print(f"[INFO] Selected class: {state.current_class}")

    elif event == cv2.EVENT_MOUSEWHEEL:
        try:
            steps = int(flags / 120)
        except:
            steps = 0

        max_off = max(0, MAX_PER_COL * CLASS_HEIGHT - CLASS_WINDOW_H)
        state.scroll_offset -= steps * CLASS_HEIGHT
        state.scroll_offset = max(0, min(state.scroll_offset, max_off))

def draw_class_window(images, current_index):
    """Draw class selector window"""
    MAX_PER_COL = 10
    col_width = 220
    rows_per_col = MAX_PER_COL

    total_classes = len(CLASSLIST)
    total_cols = (total_classes + rows_per_col - 1) // rows_per_col

    canvas_w = total_cols * col_width
    canvas_h = CLASS_WINDOW_H + 60

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # Count classes
    class_counts = {cls: 0 for cls in CLASSLIST}
    for _, _, _, _, cls in state.bboxes:
        if cls in class_counts:
            class_counts[cls] += 1

    # Compute visible row range
    start_row = state.scroll_offset // CLASS_HEIGHT
    end_row = min(start_row + CLASS_WINDOW_H // CLASS_HEIGHT + 1, rows_per_col)

    # Draw each column
    for col in range(total_cols):
        x0 = col * col_width

        for row in range(start_row, end_row):
            idx = col * rows_per_col + row
            if idx >= total_classes:
                continue

            cls = CLASSLIST[idx]
            is_selected = (cls == state.current_class)
            is_visible = state.visible_class.get(cls, True)

            y_pos = (row - start_row) * CLASS_HEIGHT

            # Background
            bg_color = (0, 0, 255) if is_selected else (60, 60, 60)
            cv2.rectangle(canvas, (x0, y_pos), 
                         (x0 + col_width, y_pos + CLASS_HEIGHT - 2), bg_color, -1)

            # Eye icon
            cx = x0 + 18
            cy = y_pos + 18
            eye_w, eye_h = 16, 8
            eye_thickness = 2

            if is_visible:
                cv2.circle(canvas, (cx, cy), 3, (200, 80, 80), -1)
                cv2.ellipse(canvas, (cx, cy), (eye_w // 2, eye_h), 
                           0, 0, 360, (60, 200, 60), eye_thickness)
            else:
                cv2.ellipse(canvas, (cx, cy), (eye_w // 2, eye_h), 
                           0, 0, 360, (60, 60, 200), eye_thickness)
                cv2.line(canvas, (cx - eye_w // 2, cy - eye_h // 2),
                        (cx + eye_w // 2, cy + eye_h // 2), (255, 255, 255), 2)
                cv2.circle(canvas, (cx, cy), 3, (80, 80, 200), -1)

            # Text
            text = f"{cls} ({class_counts[cls]})"
            cv2.putText(canvas, text, (x0 + 45, y_pos + 22),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Bottom info
    total_images = len(images)
    info_text = f"Image {current_index + 1}/{total_images}"

    cv2.rectangle(canvas, (0, CLASS_WINDOW_H), 
                 (canvas_w, CLASS_WINDOW_H + 60), (40, 40, 40), -1)
    cv2.putText(canvas, info_text, (10, CLASS_WINDOW_H + 35),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

    cv2.imshow("ClassSelector", canvas)