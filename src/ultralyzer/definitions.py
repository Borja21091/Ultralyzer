import os

# ROOT DIRECTORY
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# DB DIRECTORY
DB_DIR = os.path.join(ROOT_DIR, ".db")

# MODEL DIRECTORY
MODELS_DIR = os.path.join(ROOT_DIR, "backend", "models")

# MODEL DOWNLOAD BASE URL
MODEL_BASE_URL_UWF = 'https://github.com/Borja21091/Ultralyzer/releases/download/model_weights/'

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
    "default": "#cccccc",   # Light gray
    "pass": "#059669",      # Emerald
    "borderline": "#d97706",# Amber
    "reject": "#e11d48"     # Rose
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
    "red": 0,
    "green": 1,
    "blue": 2,
    "vessels": 3,
    "all": 4,
    "none": 5
}

# EDIT TOOLS
EDIT_TOOLS = {
    "brush": 0,
    "eraser": 1
    }

# METRICS
METRIC_DICTIONARY = {
    # GENERAL
    "laterality" : "Image eye laterality (left/right)",
    # OPTIC DISC
    "disc_center_x" : "X coordinate of optic disc center",
    "disc_center_y" : "Y coordinate of optic disc center",
    "disc_diameter_px" : "Diameter of equivalent optic disc circle (circle with same area as optic disc) in pixels",
    "disc_diameter_um" : "Diameter of equivalent optic disc circle (circle with same area as optic disc) in micrometers",
    "disc_area_px" : "Area of optic disc in pixels",
    "disc_area_um" : "Area of optic disc in micrometers",
    "disc_major_axis_px" : "Length of the major axis of the fitted ellipse in pixels",
    "disc_major_axis_um" : "Length of the major axis of the fitted ellipse in micrometers",
    "disc_minor_axis_px" : "Length of the minor axis of the fitted ellipse in pixels",
    "disc_minor_axis_um" : "Length of the minor axis of the fitted ellipse in micrometers",
    "disc_orientation_deg" : "Angle between the disc-centroid -- fovea line and the major axis of the fitted ellipse in degrees",
    "disc_circularity" : "Roundness of optic disc [0, 1]",
    "disc_eccentricity" : "Eccentricity of the ellipse that has the same second-moments as the region. The eccentricity is the ratio of the focal distance (distance between focal points) over the major axis length [0, 1)",
    # FOVEA
    "fovea_center_x" : "X coordinate of fovea center",
    "fovea_center_y" : "Y coordinate of fovea center",
    # OPTIC DISC - FOVEA RELATIONSHIP
    "disc_fovea_distance_px" : "Distance between optic disc and fovea centers in pixels",
    "disc_fovea_distance_um" : "Distance between optic disc and fovea centers in micrometers",
    "disc_fovea_angle_deg" : "Angle between horizontal axis and line connecting optic disc and fovea centers in degrees",
    # VESSELS
    "vessel_density" : "Ratio of vessel pixels to total image pixels",
    "vessel_tortuosity_density" : "Measure of how twisted the vessels are",
    "vessel_tortuosity_fft" : "Frequency domain measure of vessel tortuosity",
    "vessel_fractal_dimension_sandbox" : "Fractal dimension of vessels using sandbox method",
    "vessel_fractal_dimension_boxcount" : "Fractal dimension of vessels using box-counting method",
    "vessel_width_px" : "Average vessel width in pixels",
    "vessel_width_um" : "Average vessel width in micrometers",
    "vessel_width_gradient_px" : "Gradient (unitless) of vessel width along the vessel length in image space (pixels)",
    "vessel_width_gradient_um" : "Gradient (um/mm) of vessel width along the vessel length in real space (micrometers)",
    "vessel_width_intercept_px" : "Y-intercept of vessel width linear fit. Theoretically, vessel width at distance 0 from OD center in pixels",
    "vessel_width_intercept_um" : "Y-intercept of vessel width linear fit. Theoretically, vessel width at distance 0 from OD center in micrometers",
    # ARTERIES
    "crae" : "Central Retinal Artery Equivalent",
    "a_density" : "Ratio of artery pixels to total image pixels",
    "a_tortuosity_density" : "Measure of how twisted the arteries are",
    "a_tortuosity_fft" : "Frequency domain measure of artery tortuosity",
    "a_fractal_dimension_sandbox" : "Fractal dimension of arteries using sandbox method",
    "a_fractal_dimension_boxcount" : "Fractal dimension of arteries using box-counting method",
    "a_width_px" : "Average artery width in pixels",
    "a_width_um" : "Average artery width in micrometers",
    "a_width_gradient_px" : "Gradient (unitless) of artery width along the artery length in image space (pixels)",
    "a_width_gradient_um" : "Gradient (um/mm) of artery width along the artery length in real space (micrometers)",
    "a_width_intercept_px" : "Y-intercept of artery width linear fit. Theoretically, artery width at distance 0 from OD center in pixels",
    "a_width_intercept_um" : "Y-intercept of artery width linear fit. Theoretically, artery width at distance 0 from OD center in micrometers",
    "a_groups" : "Number of independent artery groups",
    "a_branching_points" : "Average number of artery branching points per independent artery group",
    "a_branches" : "Average number of artery branches per independent artery group",
    # VEINS
    "crve" : "Central Retinal Vein Equivalent",
    "v_density" : "Ratio of vein pixels to total image pixels",
    "v_tortuosity_density" : "Measure of how twisted the veins are",
    "v_tortuosity_fft" : "Frequency domain measure of vein tortuosity",
    "v_fractal_dimension_sandbox" : "Fractal dimension of veins using sandbox method",
    "v_fractal_dimension_boxcount" : "Fractal dimension of veins using box-counting method",
    "v_width_px" : "Average vein width in pixels",
    "v_width_um" : "Average vein width in micrometers",
    "v_width_gradient_px" : "Gradient (unitless) of vein width along the vein length in image space (pixels)",
    "v_width_gradient_um" : "Gradient (um/mm) of vein width along the vein length in real space (micrometers)",
    "v_width_intercept_px" : "Y-intercept of vein width linear fit. Theoretically, vein width at distance 0 from OD center in pixels",
    "v_width_intercept_um" : "Y-intercept of vein width linear fit. Theoretically, vein width at distance 0 from OD center in micrometers",
    "v_groups" : "Number of independent vein groups",
    "v_branching_points" : "Average number of vein branching points per independent vein group",
    "v_branches" : "Average number of vein branches per independent vein group",
    # ARTERIES - VEINS RELATIONSHIP
    "av_ratio" : "Ratio of artery pixels to vein pixels",
    "av_crossings" : "Number of vessel artery-vein crossings",
    "av_arcade_concavity" : "Concavity of the main vessel arcades",
}