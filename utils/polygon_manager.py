"""
Polygon annotation utilities for drawing, editing, and manipulation
Handles polygon operations: creation, editing, validation
"""
import cv2
import numpy as np
from typing import List, Tuple

class PolygonManager:
    """Manage polygon drawing and editing operations"""
    
    def __init__(self):
        self.point_radius = 5
        self.point_color = (0, 255, 255)  # Cyan for control points
        self.completed_color = (0, 255, 0)  # Green for completed polygon
    
    def draw_polygon(self, img_array: np.ndarray, points: List[Tuple[int, int]], 
                    color: Tuple[int, int, int], thickness: int = 2, 
                    is_selected: bool = False, show_points: bool = True) -> np.ndarray:
        """
        Draw a polygon on image with optional control points
        
        Args:
            img_array: Input image array (RGB)
            points: List of (x, y) coordinate tuples
            color: BGR color for polygon outline
            thickness: Line thickness
            is_selected: Whether polygon is selected (draw different color)
            show_points: Whether to show control points
        
        Returns:
            Modified image array
        """
        if len(points) < 2:
            return img_array
        
        # Convert points to numpy array format for drawContours
        pts = np.array(points, dtype=np.int32)
        
        # Use selected color if chosen
        if is_selected:
            draw_color = (60, 60, 200)  # Red for selected
        else:
            draw_color = color
        
        # Draw polygon lines
        if len(points) >= 2:
            for i in range(len(points)):
                pt1 = (int(points[i][0]), int(points[i][1]))  # Convert to int for cv2
                pt2 = (int(points[(i + 1) % len(points)][0]), int(points[(i + 1) % len(points)][1]))
                cv2.line(img_array, pt1, pt2, draw_color, thickness)
        
        # Draw control points if available
        if show_points:
            for i, pt in enumerate(points):
                pt_int = (int(pt[0]), int(pt[1]))  # Convert to int for cv2
                cv2.circle(img_array, pt_int, self.point_radius, self.point_color, -1)
                # Draw small index number
                cv2.putText(img_array, str(i), 
                          (pt_int[0] + 10, pt_int[1] - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.point_color, 1)
        
        return img_array
    
    def draw_preview_polygon(self, img_array: np.ndarray, completed_points: List[Tuple[int, int]], 
                            current_mouse_pos: Tuple[int, int] = None, color: Tuple[int, int, int] = None) -> np.ndarray:
        """
        Draw polygon being drawn with preview line to mouse cursor
        
        Args:
            img_array: Input image array
            completed_points: Points already placed
            current_mouse_pos: Current mouse position for preview line
            color: Optional color for polygon lines (B, G, R). Default (0, 255, 0) green
        
        Returns:
            Modified image array
        """
        if len(completed_points) < 1:
            return img_array
        
        # Default color is green for annotation polygons, red for mask polygons
        if color is None:
            color = (0, 255, 0)  # Green
        
        # Draw completed lines
        for i in range(len(completed_points) - 1):
            pt1 = (int(completed_points[i][0]), int(completed_points[i][1]))  # Convert to int
            pt2 = (int(completed_points[i + 1][0]), int(completed_points[i + 1][1]))
            cv2.line(img_array, pt1, pt2, color, 2)
        
        # Draw preview line from last point to mouse
        if current_mouse_pos and len(completed_points) > 0:
            pt_last = (int(completed_points[-1][0]), int(completed_points[-1][1]))  # Convert to int
            pt_mouse = (int(current_mouse_pos[0]), int(current_mouse_pos[1]))
            cv2.line(img_array, pt_last, pt_mouse, color, 1)
        
        # Draw completed points
        for pt in completed_points:
            pt_int = (int(pt[0]), int(pt[1]))  # Convert to int
            cv2.circle(img_array, pt_int, self.point_radius, color, -1)
        
        return img_array
    
    def point_in_polygon(self, point: Tuple[int, int], polygon: List[Tuple[int, int]]) -> bool:
        """
        Check if point is inside polygon using ray casting algorithm
        
        Args:
            point: (x, y) coordinate to test
            polygon: List of (x, y) polygon vertices
        
        Returns:
            True if point is inside polygon
        """
        if len(polygon) < 3:
            return False
        
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def get_closest_point(self, point: Tuple[int, int], polygon: List[Tuple[int, int]], 
                         threshold: int = 10) -> int:
        """
        Find closest control point in polygon to given point
        
        Args:
            point: (x, y) coordinate
            polygon: List of polygon vertices
            threshold: Maximum distance to consider as "close"
        
        Returns:
            Index of closest point, or -1 if none within threshold
        """
        if not polygon:
            return -1
        
        min_dist = float('inf')
        closest_idx = -1
        
        for i, poly_pt in enumerate(polygon):
            dist = np.sqrt((point[0] - poly_pt[0]) ** 2 + (point[1] - poly_pt[1]) ** 2)
            if dist < min_dist and dist <= threshold:
                min_dist = dist
                closest_idx = i
        
        return closest_idx
    
    def move_polygon(self, polygon: List[Tuple[int, int]], dx: int, dy: int) -> List[Tuple[int, int]]:
        """
        Move entire polygon by offset
        
        Args:
            polygon: List of (x, y) vertices
            dx: X offset
            dy: Y offset
        
        Returns:
            New polygon with moved coordinates
        """
        return [(x + dx, y + dy) for x, y in polygon]
    
    def move_polygon_point(self, polygon: List[Tuple[int, int]], point_idx: int, 
                          new_pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Move single control point of polygon
        
        Args:
            polygon: List of (x, y) vertices
            point_idx: Index of point to move
            new_pos: New (x, y) position
        
        Returns:
            New polygon with moved point
        """
        if point_idx < 0 or point_idx >= len(polygon):
            return polygon
        
        new_polygon = polygon.copy()
        new_polygon[point_idx] = new_pos
        return new_polygon
    
    def delete_polygon_point(self, polygon: List[Tuple[int, int]], point_idx: int) -> List[Tuple[int, int]]:
        """
        Delete a control point from polygon
        
        Args:
            polygon: List of (x, y) vertices
            point_idx: Index of point to delete
        
        Returns:
            New polygon without deleted point
        """
        if len(polygon) <= 3:  # Need at least 3 points for polygon
            return polygon
        
        if point_idx < 0 or point_idx >= len(polygon):
            return polygon
        
        new_polygon = polygon[:point_idx] + polygon[point_idx + 1:]
        return new_polygon
    
    def simplify_polygon(self, polygon: List[Tuple[int, int]], epsilon: float = 1.0) -> List[Tuple[int, int]]:
        """
        Simplify polygon by removing points that are too close (Ramer-Douglas-Peucker algorithm)
        
        Args:
            polygon: List of (x, y) vertices
            epsilon: Distance threshold for point removal
        
        Returns:
            Simplified polygon
        """
        if len(polygon) < 4:
            return polygon
        
        # Convert to numpy for easier processing
        pts = np.array(polygon, dtype=np.float32)
        
        # Use OpenCV's approxPolyDP
        epsilon_px = epsilon
        simplified = cv2.approxPolyDP(pts, epsilon_px, True)
        
        # Convert back to list of tuples
        return [tuple(pt[0].astype(int)) for pt in simplified]
    
    def get_polygon_bounds(self, polygon: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
        """
        Get bounding box of polygon
        
        Args:
            polygon: List of (x, y) vertices
        
        Returns:
            (x_min, y_min, x_max, y_max)
        """
        if not polygon:
            return (0, 0, 0, 0)
        
        xs = [pt[0] for pt in polygon]
        ys = [pt[1] for pt in polygon]
        
        return (min(xs), min(ys), max(xs), max(ys))
    
    def polygon_to_bbox(self, polygon: List[Tuple[int, int]]) -> List[int]:
        """
        Convert polygon to bounding box
        
        Args:
            polygon: List of (x, y) vertices
        
        Returns:
            [x1, y1, x2, y2] bounding box
        """
        x_min, y_min, x_max, y_max = self.get_polygon_bounds(polygon)
        return [x_min, y_min, x_max, y_max]
    
    def is_valid_polygon(self, polygon: List[Tuple[int, int]]) -> bool:
        """
        Check if polygon is valid (has at least 3 points and is not degenerate)
        
        Args:
            polygon: List of (x, y) vertices
        
        Returns:
            True if polygon is valid
        """
        if len(polygon) < 3:
            return False
        
        # Check if all points are not collinear (simple check)
        if len(set(polygon)) < 3:  # Has duplicate points
            return False
        
        return True


# Global instance for utility access
polygon_manager = PolygonManager()
