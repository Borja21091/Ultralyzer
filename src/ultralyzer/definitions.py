import os

# ROOT DIRECTORY
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# DB DIRECTORY
DB_DIR = os.path.join(ROOT_DIR, ".db")

# MODEL DIRECTORY
UNET_MODEL_DIR = os.path.join(ROOT_DIR, "backend", "models", "unet", "weights")
VESSEL_MODEL_DIR = os.path.join(ROOT_DIR, "backend", "models", "vessel")

# MODEL DOWNLOAD BASE URL
MODEL_BASE_URL_UWF = 'https://github.com/Borja21091/palloryzer/releases/download/uwf_model_weights'

# SEGMENTATION DIRECTORY
SEG_DIR = os.path.join(ROOT_DIR, ".seg")

# IMAGE FORMATS
IMAGE_FORMATS = (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".tif")

# BLANK STATE QUALITY-CONTROL
BLANK_STATE = {
    'filename': '', 
    'decision': None, 
    'notes': ''
}

# CANVAS BORDER COLORS
CANVAS_BORDER_COLORS = {
    "default": "#cccccc",    # Light gray
    "pass": "#2f9e44",       # Green
    "borderline": "#e6a500", # Orange
    "reject": "#dc2626",     # Red
    }

# IMAGE CHANNELS
IMAGE_CHANNEL_MAP = {
    "color": 0,
    "red": 1,
    "green": 2,
    "blue": 3
}

# OVERLAY OPTIONS
OVERLAY_MAP = {
    "arteries": 0,
    "veins": 1,
    "both": 2,
    "none": 3
}

# EDIT TOOLS
EDIT_TOOLS = {
    "brush": 0,
    "eraser": 1
    }