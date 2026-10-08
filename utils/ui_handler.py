"""
UI and mouse event handling
"""
import cv2
from .config import CLASSLIST, colorsPalette, state

def draw_all(frame_draw):
    """Draw all bounding boxes on frame"""
    for i, (x1, y1, x2, y2, cls) in enumerate(state.bboxes):
        if not state.visible_class.get(cls, True):
            continue
        classIndex = CLASSLIST.index(cls)
        color = colorsPalette[classIndex] if i != state.selected_bbox else (60, 60, 200)
        cv2.rectangle(frame_draw, (x1, y1), (x2, y2), color, 1)
        cv2.putText(frame_draw, cls, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # Indicator for new box mode
    if state.force_new_bbox:
        cv2.rectangle(frame_draw, (5, 5), 
                     (state.frame.shape[1] - 5, state.frame.shape[0] - 5), 
                     (200, 60, 60), 2)

def mouse_event(event, x, y, flags, param):
    """Handle mouse events on main window"""
    if event == cv2.EVENT_LBUTTONDOWN:
        state.ix, state.iy = x, y
        state.selected_bbox = None
        state.resizing = False
        state.resize_mode = None

        # Mode for creating new bbox
        if state.force_new_bbox:
            state.drawing = True
            return

        # Find all clicked bboxes → select smallest one
        clicked = []
        for i, (x1, y1, x2, y2, cls) in enumerate(state.bboxes):
            if x1 <= x <= x2 and y1 <= y <= y2:
                area = (x2 - x1) * (y2 - y1)
                clicked.append((area, i))
        
        if clicked:
            _, state.selected_bbox = min(clicked, key=lambda a: a[0])
            x1, y1, x2, y2, _ = state.bboxes[state.selected_bbox]

            # Check if clicked on corner handle
            handle_size = 10
            if abs(x - x1) < handle_size and abs(y - y1) < handle_size:
                state.resizing = True
                state.resize_mode = 'tl'
            elif abs(x - x2) < handle_size and abs(y - y1) < handle_size:
                state.resizing = True
                state.resize_mode = 'tr'
            elif abs(x - x1) < handle_size and abs(y - y2) < handle_size:
                state.resizing = True
                state.resize_mode = 'bl'
            elif abs(x - x2) < handle_size and abs(y - y2) < handle_size:
                state.resizing = True
                state.resize_mode = 'br'
            else:
                state.moving = True
            return

        state.drawing = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if state.drawing:
            temp = state.frame.copy()
            cv2.rectangle(temp, (state.ix, state.iy), (x, y), (255, 0, 0), 2)
            draw_all(temp)
            cv2.imshow("Annotator", temp)

        elif state.moving and state.selected_bbox is not None:
            dx, dy = x - state.ix, y - state.iy
            state.bboxes[state.selected_bbox][0] += dx
            state.bboxes[state.selected_bbox][1] += dy
            state.bboxes[state.selected_bbox][2] += dx
            state.bboxes[state.selected_bbox][3] += dy
            state.ix, state.iy = x, y

        elif state.resizing and state.selected_bbox is not None:
            x1, y1, x2, y2, cls = state.bboxes[state.selected_bbox]
            if state.resize_mode == 'tl':
                state.bboxes[state.selected_bbox][0] = min(x, x2 - 5)
                state.bboxes[state.selected_bbox][1] = min(y, y2 - 5)
            elif state.resize_mode == 'tr':
                state.bboxes[state.selected_bbox][2] = max(x, x1 + 5)
                state.bboxes[state.selected_bbox][1] = min(y, y2 - 5)
            elif state.resize_mode == 'bl':
                state.bboxes[state.selected_bbox][0] = min(x, x2 - 5)
                state.bboxes[state.selected_bbox][3] = max(y, y1 + 5)
            elif state.resize_mode == 'br':
                state.bboxes[state.selected_bbox][2] = max(x, x1 + 5)
                state.bboxes[state.selected_bbox][3] = max(y, y1 + 5)

    elif event == cv2.EVENT_LBUTTONUP:
        if state.drawing:
            x1 = min(state.ix, x)
            y1 = min(state.iy, y)
            x2 = max(state.ix, x)
            y2 = max(state.iy, y)
            w, h = abs(x2 - x1), abs(y2 - y1)
            if w >= 8 and h >= 8:
                state.bboxes.append([x1, y1, x2, y2, state.current_class])
            else:
                print("[INFO] Skipped tiny bbox (<8px).")
        state.drawing = state.moving = state.resizing = False
        state.force_new_bbox = False