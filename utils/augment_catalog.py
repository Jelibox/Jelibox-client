"""What the augmentation picker shows: one tile per kind of augmentation (Flip, Brightness, Blur ...), what each tile
controls, how its preview pictures are made, and which tiles Ultralytics already does while it trains.

No Tk here: the window (AugmentationPanel) draws what this module describes, and the tests check it alone.
"""
from . import augmentation as aug
from .augmentation import AugmentConfig, OpSetting

# One tile = one picture in the grid and one settings window.
#   ops       the augmentation operations it switches on (a tile with two, like Flip, has a tick per operation)
#   slider    the operation whose magnitude the slider sets, and the words around it
#   choices   (operation, option, text): yes/no options of an operation; option None = the operation itself
#   previews  (label, operation, high, option overrides): the examples shown next to the original
#   ultralytics  what Ultralytics already does while it trains (the tile is greyed out in the training window), or None
TILES = [
    {"key": "flip", "title": "Flip", "ops": ("flip_h", "flip_v"), "slider": None,
     "description": "Flipping pictures teaches your model that an object can face either way, so it does not "
                    "depend on which side of the picture it usually appears.",
     "choices": [("flip_h", None, "Horizontal (left-right)"), ("flip_v", None, "Vertical (top-bottom)")],
     "previews": [("horizontal", "flip_h", True, {}), ("vertical", "flip_v", True, {})],
     "ultralytics": "Ultralytics already flips pictures left-right at random while it trains (fliplr = 0.5) and can "
                    "flip them upside down (flipud). Adding it here would apply it twice."},
    {"key": "rot90", "title": "90° Rotate", "ops": ("rot90",), "slider": None,
     "description": "Turning pictures by 90 degrees helps when the camera can be mounted or held in any direction. "
                    "Pick the turns that can really happen.",
     "choices": [("rot90", "cw", "Clockwise"), ("rot90", "ccw", "Counter-clockwise"),
                 ("rot90", "upside_down", "Upside down")],
     "previews": [("clockwise", "rot90", True, {"cw": True, "ccw": False, "upside_down": False}),
                  ("counter-clockwise", "rot90", True, {"cw": False, "ccw": True, "upside_down": False}),
                  ("upside down", "rot90", True, {"cw": False, "ccw": False, "upside_down": True})],
     "ultralytics": None},
    {"key": "rotate", "title": "Rotation", "ops": ("rotate",),
     "slider": ("rotate", "Between -N° and +N°", "°"),
     "description": "Tilts pictures a little, so your model copes with cameras that are not perfectly level. The "
                    "canvas grows to keep the whole picture; the new corners are black.",
     "choices": [],
     "previews": [("-N°", "rotate", False, {}), ("+N°", "rotate", True, {})],
     "ultralytics": "Ultralytics can already rotate pictures while it trains (degrees). Use its own setting instead of "
                    "adding a second rotation here."},
    {"key": "shear", "title": "Shear", "ops": ("shear",), "slider": ("shear", "Between -N° and +N°", "°"),
     "description": "Slants pictures sideways or up and down, like looking at the scene from an angle. The canvas "
                    "grows to keep the whole picture.",
     "choices": [("shear", "horizontal", "Horizontal"), ("shear", "vertical", "Vertical")],
     "previews": [("-N°", "shear", False, {}), ("+N°", "shear", True, {})],
     "ultralytics": "Ultralytics can already shear pictures while it trains (shear). Use its own setting instead of "
                    "adding a second shear here."},
    {"key": "grayscale", "title": "Grayscale", "ops": ("grayscale",), "slider": None,
     "description": "Turns pictures black and white, so your model does not lean on colour alone. It is applied to a "
                    "share of the outputs, not all of them.",
     "choices": [], "previews": [("grayscale", "grayscale", True, {})], "ultralytics": None},
    {"key": "hue", "title": "Hue", "ops": ("hue",), "slider": ("hue", "Between -N° and +N°", "°"),
     "description": "Shifts the colours around the colour wheel, so your model copes with different white balance and "
                    "colour casts.",
     "choices": [], "previews": [("-N°", "hue", False, {}), ("+N°", "hue", True, {})],
     "ultralytics": "Ultralytics already shifts hue at random while it trains (hsv_h). Adding it here would apply "
                    "it twice."},
    {"key": "saturation", "title": "Saturation", "ops": ("saturation",),
     "slider": ("saturation", "Between -N% and +N%", "%"),
     "description": "Makes colours more vivid or more washed out, so your model copes with different cameras and "
                    "lighting.",
     "choices": [], "previews": [("-N%", "saturation", False, {}), ("N%", "saturation", True, {})],
     "ultralytics": "Ultralytics already changes saturation at random while it trains (hsv_s). Adding it here "
                    "would apply it twice."},
    {"key": "brightness", "title": "Brightness", "ops": ("brightness",),
     "slider": ("brightness", "Between -N% and +N%", "%"),
     "description": "Add variability to image brightness to help your model be more resilient to lighting and camera "
                    "setting changes.",
     "choices": [("brightness", "brighten", "Brighten"), ("brightness", "darken", "Darken")],
     "previews": [("-N%", "brightness", False, {}), ("N%", "brightness", True, {})],
     "ultralytics": "Ultralytics already changes brightness (the value channel) at random while it trains (hsv_v). "
                    "Adding it here would apply it twice. Exposure below is a different effect and is available."},
    {"key": "exposure", "title": "Exposure", "ops": ("exposure",),
     "slider": ("exposure", "Between -N% and +N%", "%"),
     "description": "Changes the exposure the way a camera does (a gamma curve): bright pictures keep their highlights "
                    "while the dark parts change most, which plain brightness does not do.",
     "choices": [("exposure", "brighten", "Overexpose"), ("exposure", "darken", "Underexpose")],
     "previews": [("-N%", "exposure", False, {}), ("N%", "exposure", True, {})], "ultralytics": None},
    {"key": "contrast", "title": "Contrast", "ops": ("contrast",), "slider": ("contrast", "Between -N% and +N%", "%"),
     "description": "Flattens or sharpens the difference between light and dark areas.",
     "choices": [], "previews": [("-N%", "contrast", False, {}), ("N%", "contrast", True, {})], "ultralytics": None},
    {"key": "blur", "title": "Blur", "ops": ("blur",), "slider": ("blur", "Up to N pixels", "px"),
     "description": "Softens pictures, so your model copes with out-of-focus or low quality cameras.",
     "choices": [], "previews": [("N px", "blur", True, {})], "ultralytics": None},
    {"key": "motion_blur", "title": "Motion Blur", "ops": ("motion_blur",),
     "slider": ("motion_blur", "Up to N pixels", "px"),
     "description": "Smears pictures in one direction, like a moving camera or a fast object.",
     "choices": [], "previews": [("N px", "motion_blur", True, {})], "ultralytics": None},
    {"key": "noise", "title": "Noise", "ops": ("noise",), "slider": ("noise", "Up to N levels", "σ"),
     "description": "Adds grain, so your model copes with low light and cheap sensors.",
     "choices": [], "previews": [("N", "noise", True, {})], "ultralytics": None},
    {"key": "cutout", "title": "Cutout", "ops": ("cutout",), "slider": ("cutout", "Box size up to N% of the side", "%"),
     "description": "Paints black boxes over random parts of the picture, so your model learns to recognise objects "
                    "that are partly hidden. The labels are left as they are.",
     "choices": [], "count": ("cutout", "count", "Boxes", 1, 10),
     "previews": [("cutout", "cutout", True, {})], "ultralytics": None},
    {"key": "jpeg", "title": "JPEG compression", "ops": ("jpeg",), "slider": None,
     "description": "Re-saves pictures as low quality JPEG (quality 70), so your model copes with compressed video "
                    "frames and web pictures.",
     "choices": [], "previews": [("quality 70", "jpeg", True, {})], "ultralytics": None},
]
TILE_BY_KEY = {t["key"]: t for t in TILES}

OUTPUTS_RANGE = (1, 20)
DEFAULT_OUTPUTS = 3


def default_state():
    """{operation name: OpSetting} with nothing switched on and every setting at its default."""
    return {name: OpSetting(False, spec["default"], 50.0, dict(spec.get("options") or {}))
            for name, spec in aug.OPERATIONS.items()}


def tile_enabled(tile, state):
    return any(state[op].enabled for op in tile["ops"])


def selected_tiles(state):
    return [t for t in TILES if tile_enabled(t, state)]


def summary(state):
    """["Flip", "Brightness ±30%", "Blur up to 5px", ...] for the picked tiles."""
    out = []
    for tile in selected_tiles(state):
        slider = tile["slider"]
        if not slider:
            out.append(tile["title"])
            continue
        amount = f"{int(state[slider[0]].magnitude)}{slider[2]}"
        out.append(f"{tile['title']} ±{amount}" if "Between" in slider[1] else f"{tile['title']} up to {amount}")
    return out


def to_config(state, outputs, keep_original, seed=None, splits=("train",)):
    """The AugmentConfig for these picks. `outputs` = pictures per training image; with `keep_original` one of them is
    the original, so there are outputs - 1 augmented copies."""
    outputs = max(OUTPUTS_RANGE[0], min(OUTPUTS_RANGE[1], int(outputs)))
    copies = outputs - 1 if keep_original else outputs
    ops = {name: OpSetting(s.enabled, s.magnitude, s.probability, dict(s.options)) for name, s in state.items()}
    return AugmentConfig(ops=ops, copies=copies, splits=tuple(splits), seed=seed, keep_original=bool(keep_original))


def from_config(config):
    """The picker's state for an AugmentConfig (the other way round of to_config)."""
    state = default_state()
    for name, s in (config.ops or {}).items():
        if name in state:
            state[name] = OpSetting(s.enabled, s.magnitude, s.probability, {**state[name].options, **s.options})
    return state


def setting_for_preview(tile_op, state, overrides=None):
    """The setting a preview picture is made with: the current one, switched on, plus the preview's own options."""
    cur = state[tile_op]
    options = {**cur.options, **(overrides or {})}
    return OpSetting(True, cur.magnitude, 100.0, options)


def previews(tile, state, image):
    """[(label, image)] for the settings window: the original, then the tile's examples at the current setting."""
    out = [("original", image)]
    for label, op, high, overrides in tile["previews"]:
        shown = label
        if "N" in label and tile["slider"]:
            amount = int(state[tile["slider"][0]].magnitude)
            shown = label.replace("-N", f"-{amount}").replace("N", str(amount))
        out.append((shown, aug.preview_operation(image, op, setting_for_preview(op, state, overrides), high)))
    return out


def thumbnail(tile, state, image):
    """The one picture a tile shows in the grid: its strongest example (the first one when it has two)."""
    label, op, high, overrides = tile["previews"][-1]
    return aug.preview_operation(image, op, setting_for_preview(op, state, overrides), high)


def usable_tiles(training):
    """(tile, note): every tile, with the reason it is greyed out when `training` and Ultralytics already does it."""
    return [(t, t["ultralytics"] if training else None) for t in TILES]
