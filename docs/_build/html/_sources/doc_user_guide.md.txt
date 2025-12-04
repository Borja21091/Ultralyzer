# User Guide

## 1. Getting Started

### Loading Data

1. **Images**: Go to `File > Open Image Folder`. Select the directory containing your retinal images (supported formats: PNG, JPG, TIFF, BMP).
2. **Existing Masks**: If you have pre-computed masks, go to `File > Load Mask Folder`. The app will match masks to images by filename.

### Navigation

* **Next/Previous Image**: Use the `◀ Prev` / `Next ▶` buttons or the **Left/Right Arrow keys**.
* **Dropdown**: Jump to a specific image using the dropdown menu at the top of the window.

---

## 2. Segmentation Workflow

### Automated Segmentation

You can run the AI models directly within the app:

* **Single Image**: Click `⏩ Segment Current` in the right-hand panel.
* **Batch Processing**: Click `⏩ Segment All` to process every image in the loaded folder.
* **Specific Models**: Use the `Segmentation` menu to run specific tasks like `A/V Segment` or `Disc Segment`.

### Visualization Controls

* **Opacity**: Use the vertical slider on the left to adjust the transparency of the segmentation overlay.
* **Channels**: Toggle specific overlay views (Arteries, Veins, Optic Disc) using the dropdown or shortcuts (see below).

---

## 3. Editing & Correction

If the automated segmentation needs correction, enter **Edit Mode** by clicking `✏️ Edit Mask`.

### Tools

* **🖌️ Brush (`Ctrl+B`)**: Add to the mask.
* **✨ Smart Paint (`Ctrl+Shift+B`)**: Semi-automated painting that sticks to vessel edges.
* **🧹 Eraser (`Ctrl+E`)**: Remove parts of the mask.
* **⇄ Color Switch (`Ctrl+C`)**: Click on a vessel segment to swap it between Artery (Red) and Vein (Blue).
* **🎯 Fovea Location**: Click to manually set the fovea center point.

### Brush Controls

* **Size**: Adjust using the slider or **`+` / `-` keys**.

### Saving Edits

* **Save (`Ctrl+S`)**: Saves the current mask to the disk.
* **Undo/Redo**: `Ctrl+Z` / `Ctrl+Shift+Z`.

---

## 4. Quality Control (QC)

Classify images to filter them for analysis:

1. Review the segmentation.
2. Add optional **Notes** in the text box.
3. Click a decision button:
    * **✅ PASS**: Good quality, ready for metrics.
    * **⚠️ BORDERLINE**: Usable but has minor issues.
    * **❌ REJECT**: Poor quality or segmentation failure.

---

## 5. Exporting Results

Go to the `Database` menu:

* **Export QC Results**: Generates a CSV file with filenames, decisions, and notes.
* **Export Metrics**: Generates a CSV file with calculated vascular metrics (tortuosity, fractal dimension, etc.).

---

## Appendix: Keyboard Shortcuts

| **Navigation** | **Shortcut** |
| :--- | :--- |
| Next Image | `Right Arrow` |
| Previous Image | `Left Arrow` |

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
