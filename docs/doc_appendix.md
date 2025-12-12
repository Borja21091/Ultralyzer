# Appendix

## Keyboard Shortcuts

The following keyboard shortcuts are available to enhance your workflow within the application.

### View & Overlay Controls

| **View / Overlays** | **Shortcut** |
| :--- | :--- |
| Show Red Overlay | `1` |
| Show Green Overlay | `2` |
| Show Blue Overlay | `3` |
| Show Vessels Only | `4` |
| Show All | `5` |
| Hide Overlay | `6` |

| **Image Channels** | **Shortcut** |
| :--- | :--- |
| Color Image | `C` |
| Red Channel | `R` |
| Green Channel | `G` |
| Blue Channel | `B` |

### Editing Tools

| **Tools** | **Shortcut** |
| :--- | :--- |
| Brush Tool | `Ctrl + B` |
| Smart Paint | `Ctrl + Shift + B` |
| Eraser | `Ctrl + E` |
| Color Switch | `Ctrl + C` |
| Save Edits | `Ctrl + S` |
| Undo | `Ctrl + Z` |
| Redo | `Ctrl + Shift + Z` |
| Brush Size Up | `+` or `=` |
| Brush Size Down | `-` |

### Navigation

| **Navigation** | **Shortcut** |
| :--- | :--- |
| Next Image | `Right Arrow` |
| Previous Image | `Left Arrow` |

## Metric Definitions

This section details all the metrics calculated by Ultralyzer, organized by category.

Pixel coordinates start at the top-left corner of the image (0,0). X increases to the right, and Y increases downwards. Therefore, a coordinate `(x, y)` refers to a point `x` pixels to the right and `y` pixels down from the top-left corner.

### General

| Metric Key | Description |
| :--- | :--- |
| `laterality` | Image eye laterality (left/right) |

### Optic Disc

| Metric Key | Description |
| :--- | :--- |
| `disc_center_x` | X coordinate of optic disc center |
| `disc_center_y` | Y coordinate of optic disc center |
| `disc_diameter_px` | Diameter of equivalent optic disc circle (circle with same area as optic disc) in pixels |
| `disc_diameter_um` | Diameter of equivalent optic disc circle (circle with same area as optic disc) in micrometers |
| `disc_area_px` | Area of optic disc in pixels |
| `disc_area_um` | Area of optic disc in micrometers |
| `disc_major_axis_px` | Length of the major axis of the fitted ellipse in pixels |
| `disc_major_axis_um` | Length of the major axis of the fitted ellipse in micrometers |
| `disc_minor_axis_px` | Length of the minor axis of the fitted ellipse in pixels |
| `disc_minor_axis_um` | Length of the minor axis of the fitted ellipse in micrometers |
| `disc_orientation_deg` | Angle between the disc-centroid -- fovea line and the major axis of the fitted ellipse in degrees |
| `disc_circularity` | Roundness of optic disc [0, 1] |
| `disc_eccentricity` | Eccentricity of the ellipse that has the same second-moments as the region. The eccentricity is the ratio of the focal distance (distance between focal points) over the major axis length [0, 1) |

### Fovea

| Metric Key | Description |
| :--- | :--- |
| `fovea_center_x` | X coordinate of fovea center |
| `fovea_center_y` | Y coordinate of fovea center |

### Optic Disc - Fovea Relationship

| Metric Key | Description |
| :--- | :--- |
| `disc_fovea_distance_px` | Distance between optic disc and fovea centers in pixels |
| `disc_fovea_distance_um` | Distance between optic disc and fovea centers in micrometers |
| `disc_fovea_angle_deg` | Angle between horizontal axis and line connecting optic disc and fovea centers in degrees |

### Vessels (Arteries + Veins)

| Metric Key | Description |
| :--- | :--- |
| `vessel_density` | Ratio of vessel pixels to total image pixels |
| `vessel_tortuosity_density` | Measure of how twisted the vessels are |
| `vessel_tortuosity_fft` | Frequency domain measure of vessel tortuosity |
| `vessel_fractal_dimension_sandbox` | Fractal dimension of vessels using sandbox method |
| `vessel_fractal_dimension_boxcount` | Fractal dimension of vessels using box-counting method |
| `vessel_width_px` | Average vessel width in pixels |
| `vessel_width_um` | Average vessel width in micrometers |
| `vessel_width_gradient_px` | Gradient (unitless) of vessel width along the vessel length in image space (pixels) |
| `vessel_width_gradient_um` | Gradient (um/mm) of vessel width along the vessel length in real space (micrometers) |
| `vessel_width_intercept_px` | Y-intercept of vessel width linear fit. Theoretically, vessel width at distance 0 from OD center in pixels |
| `vessel_width_intercept_um` | Y-intercept of vessel width linear fit. Theoretically, vessel width at distance 0 from OD center in micrometers |

### Arteries

| Metric Key | Description |
| :--- | :--- |
| `crae` | Central Retinal Artery Equivalent |
| `a_density` | Ratio of artery pixels to total image pixels |
| `a_tortuosity_density` | Measure of how twisted the arteries are |
| `a_tortuosity_fft` | Frequency domain measure of artery tortuosity |
| `a_fractal_dimension_sandbox` | Fractal dimension of arteries using sandbox method |
| `a_fractal_dimension_boxcount` | Fractal dimension of arteries using box-counting method |
| `a_width_px` | Average artery width in pixels |
| `a_width_um` | Average artery width in micrometers |
| `a_width_gradient_px` | Gradient (unitless) of artery width along the artery length in image space (pixels) |
| `a_width_gradient_um` | Gradient (um/mm) of artery width along the artery length in real space (micrometers) |
| `a_width_intercept_px` | Y-intercept of artery width linear fit. Theoretically, artery width at distance 0 from OD center in pixels |
| `a_width_intercept_um` | Y-intercept of artery width linear fit. Theoretically, artery width at distance 0 from OD center in micrometers |
| `a_groups` | Number of independent artery groups |
| `a_branching_points` | Average number of artery branching points per independent artery group |
| `a_branches` | Average number of artery branches per independent artery group |

### Veins

| Metric Key | Description |
| :--- | :--- |
| `crve` | Central Retinal Vein Equivalent |
| `v_density` | Ratio of vein pixels to total image pixels |
| `v_tortuosity_density` | Measure of how twisted the veins are |
| `v_tortuosity_fft` | Frequency domain measure of vein tortuosity |
| `v_fractal_dimension_sandbox` | Fractal dimension of veins using sandbox method |
| `v_fractal_dimension_boxcount` | Fractal dimension of veins using box-counting method |
| `v_width_px` | Average vein width in pixels |
| `v_width_um` | Average vein width in micrometers |
| `v_width_gradient_px` | Gradient (unitless) of vein width along the vein length in image space (pixels) |
| `v_width_gradient_um` | Gradient (um/mm) of vein width along the vein length in real space (micrometers) |
| `v_width_intercept_px` | Y-intercept of vein width linear fit. Theoretically, vein width at distance 0 from OD center in pixels |
| `v_width_intercept_um` | Y-intercept of vein width linear fit. Theoretically, vein width at distance 0 from OD center in micrometers |
| `v_groups` | Number of independent vein groups |
| `v_branching_points` | Average number of vein branching points per independent vein group |
| `v_branches` | Average number of vein branches per independent vein group |

### Arteries - Veins Relationship

| Metric Key | Description |
| :--- | :--- |
| `av_ratio` | Ratio of artery pixels to vein pixels |
| `av_crossings` | Number of vessel artery-vein crossings |
| `av_arcade_concavity` | Concavity of the main vessel arcades |
