"""
Export helper for Jelibox using Ultralytics YOLO model export API.
Provides `export_model` which runs `YOLO(model_path).export(...)` with common options.
"""
import os
import traceback

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


def export_model(model_path: str,
                 fmt: str = 'onnx',
                 imgsz: int = 640,
                 optimize: bool = False,
                 keras: bool = False,
                 half: bool = False,
                 int8: bool = False,
                 dynamic: bool = False,
                 simplify: bool = False,
                 end2end: bool = False,
                 save_dir: str = None):
    """Export the model using ultralytics.YOLO.export.

    Returns (success: bool, message: str, out_path: str|None)
    """
    if YOLO is None:
        return False, "ultralytics package not installed", None

    if not os.path.exists(model_path):
        return False, f"Model file not found: {model_path}", None

    try:
        model = YOLO(model_path)

        kwargs = {
            'format': fmt,
            'imgsz': imgsz,
            'optimize': optimize,
            'keras': keras,
            'half': half,
            'int8': int8,
            'dynamic': dynamic,
            'simplify': simplify,
            'end2end': end2end,
        }

        # Remove keys with None or False defaults are fine to pass
        # ultralytics.export expects keyword names similar to these
        out = model.export(**kwargs)

        # `export` may return path or list of paths; normalize to string
        out_path = None
        if isinstance(out, (list, tuple)) and len(out) > 0:
            out_path = out[0]
        elif isinstance(out, str):
            out_path = out

        if save_dir and out_path:
            try:
                os.makedirs(save_dir, exist_ok=True)
                dest = os.path.join(save_dir, os.path.basename(out_path))
                os.replace(out_path, dest)
                out_path = dest
            except Exception:
                # ignore
                pass

        return True, "Export completed", out_path

    except Exception as e:
        tb = traceback.format_exc()
        return False, f"Export failed: {str(e)}\n{tb}", None
