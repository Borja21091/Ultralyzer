from PySide6.QtGui import QImage, QPixmap, QPainter, QUndoStack, QPen
from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
from frontend.models.edit_commands import BrushStrokeCommand, EraseCommand, ColorSwitchCommand, SmartPaintCommand
from PySide6.QtCore import Qt, Signal
import numpy as np
import cv2
from definitions import IMAGE_CHANNEL_MAP, OVERLAY_MAP
from typing import Optional, Dict


class ChannelBuffer:
    """
    Lightweight wrapper around a single-channel QImage for efficient editing.
    
    Stores a single channel (R, G, B, or A) as a QImage in RGB888 format
    (with all three bytes identical) and provides both Qt and numpy interfaces
    for editing operations.
    """
    
    def __init__(self, 
                 channel_array: Optional[np.ndarray] = None, 
                 width: Optional[int] = None, 
                 height: Optional[int] = None):
        """
        Initialize ChannelBuffer.
        
        Args:
            channel_array: Optional uint8 numpy array (H, W) for single channel. If provided, width/height ignored.
            width, height: Dimensions for empty buffer if channel_array is None.
        """
        if channel_array is not None:
            # Create QImage from existing channel data
            h, w = channel_array.shape[:2]
            # Convert grayscale to RGB (stacked copies) for QImage RGB888 format
            rgb_array = np.stack([channel_array, channel_array, channel_array], axis=-1).astype(np.uint8)
            bytes_per_line = 3 * w
            q_img = QImage(rgb_array.tobytes(), w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self._qimage = q_img.copy()  # Detach from numpy buffer
        else:
            # Create empty buffer
            if width is None or height is None:
                raise ValueError("Must provide either channel_array or (width, height)")
            self._qimage = QImage(width, height, QImage.Format.Format_RGB888)
            self._qimage.fill(0)  # Black (transparent/empty)
    
    @property
    def qimage(self) -> QImage:
        """Get the QImage for direct QPainter editing. Read-only."""
        return self._qimage
    
    @property
    def numpy_view(self) -> np.ndarray:
        """
        Get a numpy view of the buffer (grayscale, single channel).
        
        Returns:
            np.ndarray of shape (height, width) with uint8 values.
        """
        ptr = self._qimage.bits()
        h, w = self._qimage.height(), self._qimage.width()
        
        # QImage Format_RGB888: 3 bytes per pixel (all identical for grayscale)
        byte_count = self._qimage.bytesPerLine() * h
        pixels = np.ndarray((byte_count,), dtype=np.uint8, buffer=ptr)
        pixels = pixels.reshape((h, w, 3))
        
        # Extract first channel (all three are identical for grayscale)
        return pixels[:, :, 0].copy()
    
    def copy(self) -> 'ChannelBuffer':
        """Create a deep copy of this buffer."""
        new_buffer = ChannelBuffer.__new__(ChannelBuffer)
        new_buffer._qimage = self._qimage.copy()
        return new_buffer
    
    def to_numpy(self) -> np.ndarray:
        """
        Extract buffer as grayscale numpy array.
        
        Returns:
            np.ndarray of shape (height, width) with uint8 values.
        """
        return self.numpy_view.copy()
    
    def get_dimensions(self) -> tuple:
        """Return (width, height) of buffer."""
        return (self._qimage.width(), self._qimage.height())
    
    @staticmethod
    def compose_rgba(r_buffer: 'ChannelBuffer', 
                     g_buffer: 'ChannelBuffer', 
                     b_buffer: 'ChannelBuffer', 
                     a_buffer: 'ChannelBuffer') -> QImage:
        """
        Compose four ChannelBuffers into a single RGBA QImage.
        
        Args:
            r_buffer, g_buffer, b_buffer, a_buffer: ChannelBuffer instances
            
        Returns:
            QImage in ARGB32 format ready for display
        """
        w = r_buffer._qimage.width()
        h = r_buffer._qimage.height()
        
        # Extract numpy arrays from each channel (first byte of RGB888)
        r_ptr = r_buffer._qimage.bits()
        r_pixels = np.ndarray((h, w, 3), dtype=np.uint8, buffer=r_ptr)
        r_data = r_pixels[:, :, 0]
        
        g_ptr = g_buffer._qimage.bits()
        g_pixels = np.ndarray((h, w, 3), dtype=np.uint8, buffer=g_ptr)
        g_data = g_pixels[:, :, 0]
        
        b_ptr = b_buffer._qimage.bits()
        b_pixels = np.ndarray((h, w, 3), dtype=np.uint8, buffer=b_ptr)
        b_data = b_pixels[:, :, 0]
        
        a_ptr = a_buffer._qimage.bits()
        a_pixels = np.ndarray((h, w, 3), dtype=np.uint8, buffer=a_ptr)
        a_data = a_pixels[:, :, 0]
        
        # Stack into RGBA
        rgba = np.stack([r_data, g_data, b_data, a_data], axis=-1).astype(np.uint8)
        
        # Convert RGB to BGR for Qt (Qt uses BGRA internally for ARGB32)
        bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
        
        bytes_per_line = 4 * w
        q_img = QImage(bgra.tobytes(), w, h, bytes_per_line, QImage.Format.Format_ARGB32)
        return q_img.copy()  # Detach from numpy buffer


class ImageLayer:
    """Manages the base image with channel selection"""
    
    def __init__(self, image_array: np.ndarray):
        """
        Args:
            image_array: RGB uint8 numpy array (H, W, 3)
        """
        self._image = image_array  # Never modified
        self._channel = "color"
        self._pixmap = None
        self._update_display()
    
    @property
    def pixmap(self) -> Optional[QPixmap]:
        if self._pixmap is None:
            self._update_display()
        return self._pixmap
    
    @property
    def channel(self) -> str:
        return self._channel
    
    @channel.setter
    def channel(self, value: str):
        """Set display channel: 'color', 'red', 'green', 'blue'"""
        if value not in IMAGE_CHANNEL_MAP.keys():
            return
        self._channel = value
        self._update_display()
    
    def _update_display(self):
        """Convert current channel to QPixmap"""
        if self._channel == "color":
            display = self._image
        else:
            # Extract single channel as grayscale RGB
            channel_idx = {"red": 0, "green": 1, "blue": 2}[self._channel]
            gray = self._image[:, :, channel_idx]
            display = np.stack([gray, gray, gray], axis=-1)
        
        self._pixmap = self._numpy_to_qpixmap(display)
    
    @staticmethod
    def _numpy_to_qpixmap(array: np.ndarray) -> QPixmap:
        """Convert RGB uint8 numpy to QPixmap"""
        h, w = array.shape[:2]
        bytes_per_line = 3 * w
        q_image = QImage(array.tobytes(), w, h, bytes_per_line, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(q_image)
    

class OverlayLayer:
    """Manages the editable overlay with channel-based architecture using ChannelBuffer"""
    
    def __init__(self, overlay_array: np.ndarray):
        """
        Args:
            overlay_array: RGB uint8 numpy array (H, W, 3) with arteries in red, veins in blue and disc + fovea in green
        """
        h, w = overlay_array.shape[:2]
        
        # Store original for reset
        self._overlay_array_original = overlay_array.copy()
        
        # Create ChannelBuffers for each RGBA channel
        self._channels = {
            'r': ChannelBuffer(overlay_array[:, :, 0]),
            'g': ChannelBuffer(overlay_array[:, :, 1]),
            'b': ChannelBuffer(overlay_array[:, :, 2]),
            'a': self._generate_alpha_channel(overlay_array)
        }
        
        # Cache empty buffer
        self._empty_buffer = None
        
        self._channel = "all"
        self._opacity = 0.75
        self._pixmap = None
        self._pixmap_dirty = True  # Flag to track if pixmap needs recomposition
        self._is_dirty = False
        self._undo_stack = QUndoStack()
    
    ############ PROPERTIES ############
    
    @property
    def pixmap(self) -> Optional[QPixmap]:
        """Get the current display pixmap, recomposing if needed"""
        if self._pixmap_dirty or self._pixmap is None:
            self._update_display()
        return self._pixmap
    
    @property
    def channel(self) -> str:
        return self._channel
    
    @channel.setter
    def channel(self, value: str):
        """Set display channel: 'red', 'green', 'blue', 'vessels', 'all', 'none'"""
        valid_channels = OVERLAY_MAP.keys()
        val = value.lower()
        if val not in valid_channels or self._channel == val:
            return
        self._channel = val
        self._pixmap_dirty = True
    
    @property
    def opacity(self) -> float:
        """Get overlay opacity (0.0 to 1.0)"""
        return self._opacity
    
    @opacity.setter
    def opacity(self, value: float):
        """Set overlay opacity (0.0 to 1.0)"""
        self._opacity = max(0.0, min(1.0, value))
        self._pixmap_dirty = True
    
    ############ GETTER/SETTER ############
    
    def get_array(self) -> np.ndarray:
        """Return the RGB numpy array (composed from channel buffers)"""
        r_data = self._channels['r'].to_numpy()
        g_data = self._channels['g'].to_numpy()
        b_data = self._channels['b'].to_numpy()
        
        rgb_array = np.stack([r_data, g_data, b_data], axis=-1).astype(np.uint8)
        return rgb_array

    def get_switch_target_color(self, color_bgr: np.ndarray) -> np.ndarray:
        """
        Determine the target color after switch.
        BGR format: Red=(0,0,255), Blue=(255,0,0)
        
        Returns the new color in BGR format
        """
        # Check if predominantly red or blue
        b, r = color_bgr[0], color_bgr[2]
        
        # Red channel is dominant -> switch to blue
        if r > b:
            return np.array([255, 0, 0], dtype=np.uint8)  # Blue in BGR
        # Blue channel is dominant -> switch to red
        elif b > r:
            return np.array([0, 0, 255], dtype=np.uint8)  # Red in BGR
        # Other colors (green or low intensity) -> default to red
        else:
            return np.array([0, 0, 255], dtype=np.uint8)  # Red in BGR
    
    ############ PUBLIC METHODS ############
    
    def undo(self):
        """Undo last edit"""
        self._undo_stack.undo()
        self._is_dirty = True
        self._pixmap_dirty = True
        if self._undo_stack.count() == 0:
            self._is_dirty = False
    
    def redo(self):
        """Redo last undone edit"""
        self._undo_stack.redo()
        self._is_dirty = True
        self._pixmap_dirty = True
    
    def reset(self):
        """Reset overlay to original segmentation mask state"""
        # Restore channels from the original array
        self._channels = {
            'r': ChannelBuffer(self._overlay_array_original[:, :, 0]),
            'g': ChannelBuffer(self._overlay_array_original[:, :, 1]),
            'b': ChannelBuffer(self._overlay_array_original[:, :, 2]),
            'a': self._generate_alpha_channel(self._overlay_array_original)
        }
        self._undo_stack.clear()
        self._is_dirty = False
        self._pixmap_dirty = True
    
    def can_undo(self) -> bool:
        """Check if undo is available"""
        return self._undo_stack.canUndo()
    
    def can_redo(self) -> bool:
        """Check if redo is available"""
        return self._undo_stack.canRedo()
    
    def has_changes(self) -> bool:
        """Check if overlay has unsaved changes"""
        return self._is_dirty
    
    def mark_saved(self):
        """Mark overlay as saved"""
        self._is_dirty = False
        self._undo_stack.clear()
    
    def perform_color_switch(self, x: int, y: int) -> bool:
        """
        Perform flood fill color switch at given coordinates.
        Red (artery) -> Blue (vein), Blue -> Red, Green/other -> Red
        
        Args:
            x, y: Pixel coordinates in the overlay
            
        Returns:
            True if a switch was performed, False if out of bounds or no color to switch
        """
        self._is_dirty = True
        self._pixmap_dirty = True
        
        # Get dimensions
        w, h = self._channels['r'].get_dimensions()
        
        # Bounds check
        if x < 0 or x >= w or y < 0 or y >= h:
            return False
        
        # Get pixel colors at click position
        r_ptr = self._channels['r']._qimage.bits()
        r_pixels = np.ndarray((h, w, 3), dtype=np.uint8, buffer=r_ptr)
        r_val = r_pixels[int(y), int(x), 0]
        
        g_ptr = self._channels['g']._qimage.bits()
        g_pixels = np.ndarray((h, w, 3), dtype=np.uint8, buffer=g_ptr)
        g_val = g_pixels[int(y), int(x), 0]
        
        b_ptr = self._channels['b']._qimage.bits()
        b_pixels = np.ndarray((h, w, 3), dtype=np.uint8, buffer=b_ptr)
        b_val = b_pixels[int(y), int(x), 0]
        
        clicked_color = np.array([b_val, g_val, r_val], dtype=np.uint8)
        
        # If clicked on transparent (all black), no action
        if np.all(clicked_color == 0):
            return False
        
        # Determine target color based on clicked color (BGR format)
        target_color = self.get_switch_target_color(clicked_color)
        target_b, target_g, target_r = target_color[0], target_color[1], target_color[2]
        
        # Create masks for flood fill (using R and B channels to identify vessel pixels)
        r_data = r_pixels[:, :, 0]
        b_data = b_pixels[:, :, 0]
        
        mask = np.zeros((h, w), dtype=np.uint8)
        self._flood_fill_mask(r_data, b_data, mask, int(x), int(y), r_val, b_val, tolerance=5)
        
        # Apply color switch to all masked pixels in R, G, B channels
        for ch_name, target_val in [('r', target_r), ('g', target_g), ('b', target_b)]:
            ch_ptr = self._channels[ch_name]._qimage.bits()
            ch_pixels = np.ndarray((h, w, 3), dtype=np.uint8, buffer=ch_ptr)
            ch_pixels[mask > 0, :] = target_val
        
        return True
    
    ############ PRIVATE METHODS ############
    
    def _get_empty_buffer(self) -> ChannelBuffer:
        """Get or create a cached empty buffer matching current dimensions"""
        if self._empty_buffer is None:
            w, h = self._channels['r'].get_dimensions()
            self._empty_buffer = ChannelBuffer(width=w, height=h)
        return self._empty_buffer
    
    def _generate_alpha_channel(self, overlay_array: np.ndarray) -> ChannelBuffer:
        """Generate alpha channel: 255 where any RGB channel is non-zero, 0 otherwise"""
        h, w = overlay_array.shape[:2]
        alpha = np.bitwise_or(np.bitwise_or(overlay_array[:, :, 0], overlay_array[:, :, 1]), 
                              overlay_array[:, :, 2]).astype(np.uint8)
        return ChannelBuffer(alpha)
    
    def _update_display(self):
        """Update pixmap based on current channel selection and opacity"""
        if self._channel == "none":
            transparent = QPixmap(self._channels['r']._qimage.width(), 
                                 self._channels['r']._qimage.height())
            transparent.fill(Qt.GlobalColor.transparent)
            self._pixmap = transparent
            self._pixmap_dirty = False
            return
        
        # Compose channels based on selection
        composed_qimage = self._compose_channels()
        
        # Apply opacity to alpha channel
        self._apply_opacity_to_buffer(composed_qimage)
        
        # Convert to pixmap
        self._pixmap = QPixmap.fromImage(composed_qimage)
        self._pixmap_dirty = False
    
    def _compose_vessels_mode(self) -> QImage:
        """
        Optimized composition for 'vessels' mode (Red + Blue).
        Calculates Alpha = R | B on the fly without intermediate object creation.
        """
        w, h = self._channels['r'].get_dimensions()
        
        # 1. Get direct views of R and B channels
        r_ptr = self._channels['r']._qimage.bits()
        r_arr = np.ndarray((h, w, 3), dtype=np.uint8, buffer=r_ptr)
        r_data = r_arr[:, :, 0]
        
        b_ptr = self._channels['b']._qimage.bits()
        b_arr = np.ndarray((h, w, 3), dtype=np.uint8, buffer=b_ptr)
        b_data = b_arr[:, :, 0]
        
        # 2. Calculate Alpha and Green
        a_data = np.bitwise_or(r_data, b_data)
        g_data = np.zeros((h, w), dtype=np.uint8)
        
        # 3. Stack into RGBA (Single Allocation)
        rgba = np.stack([r_data, g_data, b_data, a_data], axis=-1).astype(np.uint8)
        
        # 4. Convert to BGR for Qt (Qt uses BGRA internally for ARGB32)
        bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
        
        bytes_per_line = 4 * w
        q_img = QImage(bgra.tobytes(), w, h, bytes_per_line, QImage.Format.Format_ARGB32)
        return q_img.copy()
    
    def _compose_channels(self) -> QImage:
        """Compose ChannelBuffers into RGBA QImage based on channel selection"""
        empty = self._get_empty_buffer()
        
        if self._channel == "red":
            # Show only red (arteries)
            composed = ChannelBuffer.compose_rgba(
                self._channels['r'], 
                empty,
                empty,
                self._channels['r']  # Alpha from red
            )
        elif self._channel == "blue":
            # Show only blue (veins)
            composed = ChannelBuffer.compose_rgba(
                empty,
                empty,
                self._channels['b'],
                self._channels['b']  # Alpha from blue
            )
        elif self._channel == "green":
            # Show only green (disc + fovea)
            composed = ChannelBuffer.compose_rgba(
                empty,
                self._channels['g'],
                empty,
                self._channels['g']  # Alpha from green
            )
        elif self._channel == "vessels":
            # Show red + blue (arteries + veins)
            composed = self._compose_vessels_mode()
        else:  # "all"
            # Show all channels
            composed = ChannelBuffer.compose_rgba(
                self._channels['r'], 
                self._channels['g'],
                self._channels['b'],
                self._channels['a']
            )
        
        return composed
    
    def _apply_opacity_to_buffer(self, qimage: QImage):
        """Modify QImage alpha channel based on opacity setting"""
        ptr = qimage.bits()
        h, w = qimage.height(), qimage.width()
        byte_count = qimage.bytesPerLine() * h
        
        # Create numpy view of pixel data (ARGB32)
        pixels = np.ndarray((byte_count,), dtype=np.uint8, buffer=ptr)
        pixels = pixels.reshape((h, w, 4))
        
        # Scale alpha channel by opacity
        pixels[:, :, 3] = (pixels[:, :, 3] * self.opacity).astype(np.uint8)

    def _draw_pen_stroke(self, pen: QPen, 
                         points: list,
                         mode: QPainter.CompositionMode = QPainter.CompositionMode.CompositionMode_SourceOver):
        """Draw a pen stroke"""
        # Determine which channels to paint based on current overlay channel
        paint_r = self.channel in ['red', 'vessels', 'all']
        paint_g = self.channel in ['green', 'all']
        paint_b = self.channel in ['blue', 'vessels', 'all']
        
        if not paint_r and not paint_g and not paint_b:
            return
        
        channels_to_paint = [ch for ch, paint in zip(['r', 'g', 'b'], [paint_r, paint_g, paint_b]) if paint]
        
        for ch_name in channels_to_paint:
            painter = QPainter(self._channels[ch_name]._qimage)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setCompositionMode(mode)
            painter.setPen(pen)
            for i in range(len(points) - 1):
                painter.drawLine(int(points[i][0]), int(points[i][1]), int(points[i+1][0]), int(points[i+1][1]))
            painter.end()
            
        # OPTIMIZATION: Paint directly on alpha channel for instant feedback
        # This avoids the expensive _update_alpha_channel() call during the stroke
        painter = QPainter(self._channels['a']._qimage)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
        painter.setPen(pen)
        for i in range(len(points) - 1):
            painter.drawLine(int(points[i][0]), int(points[i][1]), int(points[i+1][0]), int(points[i+1][1]))
        painter.end()
    
    def _draw_brush_segment(self, points, radius: float):
        """Draw a single line segment on the R and B channels"""
        self._is_dirty = True
        self._pixmap_dirty = True
        
        # Create pen
        pen = QPainter(self._channels['r']._qimage).pen()  # Get default pen
        pen.setWidth(int(2 * radius))
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        pen.setColor(Qt.GlobalColor.white)

        # Paint on selected channels
        self._draw_pen_stroke(pen, points)

    def _draw_erase_segment(self, points, radius: float):
        """Erase a single line segment, respecting the current channel"""
        self._is_dirty = True
        self._pixmap_dirty = True
        
        # Create pen
        pen = QPainter(self._channels['r']._qimage).pen()  # Get default pen
        pen.setWidth(int(2 * radius))
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        pen.setColor(Qt.GlobalColor.black)

        # Paint on selected channels
        self._draw_pen_stroke(pen, points)
    
    def _draw_smart_paint_segment(self, points, radius: float):
        """
        Draw a smart paint segment: paint over existing vessels only.
        Only non-black pixels under the brush stroke are painted with the target color.
        Paints on R and B channels only.
        """
        self._is_dirty = True
        self._pixmap_dirty = True
        
        # 1. Calculate Bounding Box (ROI)
        pad = int(radius) + 2
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        
        w, h = self._channels['r'].get_dimensions()
        
        min_x = max(0, int(min(x_coords)) - pad)
        max_x = min(w, int(max(x_coords)) + pad)
        min_y = max(0, int(min(y_coords)) - pad)
        max_y = min(h, int(max(y_coords)) + pad)
        bbox = (min_x, min_y, max_x, max_y)
        
        w_roi = max_x - min_x
        h_roi = max_y - min_y
        
        if w_roi <= 0 or h_roi <= 0:
            return

        # 2. Create Local Mask
        mask_roi = np.zeros((h_roi, w_roi), dtype=np.uint8)
        
        # Adjust points to ROI coordinates
        x0, y0 = int(points[0][0]) - min_x, int(points[0][1]) - min_y
        x1, y1 = int(points[1][0]) - min_x, int(points[1][1]) - min_y
        
        cv2.line(mask_roi, (x0, y0), (x1, y1), (255,), int(2 * radius), cv2.LINE_AA)
        
        # 3. Get Local Views of Channels
        r_roi = self._get_roi_view('r', bbox)
        b_roi = self._get_roi_view('b', bbox)
        
        # 4. Local Logic
        # Check R or B > 0 in ROI
        vessel_roi = np.bitwise_or(r_roi[:, :, 0] > 0, b_roi[:, :, 0] > 0)
        paint_area = (mask_roi > 0) & vessel_roi
        
        # 5. Write Result Locally
        if self.channel == 'red':
            r_roi[paint_area, :] = 255
            b_roi[paint_area, :] = 0
        elif self.channel == 'blue':
            r_roi[paint_area, :] = 0
            b_roi[paint_area, :] = 255
    
    def _get_roi_view(self, channel_name: str, bbox: tuple):
        """Get a numpy view of a channel buffer for the specified ROI"""
        w_img, h_img = self._channels[channel_name].get_dimensions()
        min_x, min_y, max_x, max_y = bbox
        ptr = self._channels[channel_name]._qimage.bits()
        full_arr = np.ndarray((h_img, w_img, 3), dtype=np.uint8, buffer=ptr)
        return full_arr[min_y:max_y, min_x:max_x, :]
    
    def _update_alpha_channel(self):
        """Regenerate alpha channel based on current R, G, B state"""
        w, h = self._channels['r'].get_dimensions()
        
        r_ptr = self._channels['r']._qimage.bits()
        r_pixels = np.ndarray((h, w, 3), dtype=np.uint8, buffer=r_ptr)
        r_data = r_pixels[:, :, 0]
        
        g_ptr = self._channels['g']._qimage.bits()
        g_pixels = np.ndarray((h, w, 3), dtype=np.uint8, buffer=g_ptr)
        g_data = g_pixels[:, :, 0]
        
        b_ptr = self._channels['b']._qimage.bits()
        b_pixels = np.ndarray((h, w, 3), dtype=np.uint8, buffer=b_ptr)
        b_data = b_pixels[:, :, 0]
        
        # Compute alpha: 255 where any channel is non-zero
        alpha = np.bitwise_or(np.bitwise_or(r_data, g_data), b_data).astype(np.uint8)
        
        # Update alpha channel
        self._channels['a'] = ChannelBuffer(alpha)
    
    def _flood_fill_mask(self, r_data: np.ndarray, b_data: np.ndarray, mask: np.ndarray, 
                         x: int, y: int, target_r: int, target_b: int, tolerance: int = 5):
        """
        BFS flood fill to mark all connected pixels of similar color.
        
        Args:
            r_data: Red channel data
            b_data: Blue channel data
            mask: Output mask to mark filled pixels
            x, y: Starting coordinates
            target_r, target_b: Color to match
            tolerance: Color tolerance for matching
        """
        h, w = r_data.shape
        visited = np.zeros((h, w), dtype=bool)
        queue = [(x, y)]
        visited[y, x] = True
        
        while queue:
            cx, cy = queue.pop(0)
            
            # Check if current pixel matches target color (within tolerance)
            current_r = r_data[cy, cx]
            current_b = b_data[cy, cx]
            
            if (abs(int(current_r) - int(target_r)) <= tolerance and 
                abs(int(current_b) - int(target_b)) <= tolerance):
                mask[cy, cx] = 1
                
                # Add neighbors to queue
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((nx, ny))


class Canvas(QGraphicsView):
    """Main canvas for displaying and editing image layers"""
    
    signal_zoom_changed = Signal()
    
    def __init__(self, image_layer: ImageLayer, overlay_layer: OverlayLayer):
        super().__init__()
        self.image_layer = image_layer
        self.overlay_layer = overlay_layer
        
        # Create scene
        self._scene = QGraphicsScene()
        self.setScene(self._scene)
        
        # Create pixmap items for each layer
        self.image_item = QGraphicsPixmapItem(image_layer.pixmap)
        self.overlay_item = QGraphicsPixmapItem(overlay_layer.pixmap)
        
        # Add items to scene (image first, overlay on top)
        self._scene.addItem(self.image_item)
        self._scene.addItem(self.overlay_item)
        
        # Position overlay on top of image
        self.overlay_item.setPos(0, 0)
        
        self._edit_mode = False
        
        # Pan and zoom variables
        self._pan_start = None
        self._is_panning = False
        self._zoom_level = 1.0
        self.MIN_ZOOM = 0.1
        self.MAX_ZOOM = 10.0
        
        # Enable drag mode for panning
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        
        # Mouse interaction variables
        self.current_tool = None
        self.current_channel = None
        self.brush_radius = 5
        self.stroke_points = []
        self._temp_channels_copy: Optional[Dict[str, ChannelBuffer]] = None  # Temporary channels copy for undo commands
    
    ################ GETTER/SETTER ################
    
    def get_image_channel(self) -> str:
        """Get the current image layer channel"""
        return self.image_layer.channel

    def get_overlay_channel(self) -> str:
        """Get the current overlay layer channel"""
        return self.overlay_layer.channel
    
    def set_edit_mode(self, enabled: bool):
        """Enable or disable edit mode"""
        self._edit_mode = enabled
    
    def set_tool(self, tool: str):
        """Set the current tool: 'brush', 'smart_paint', 'eraser' or 'change'"""
        self.current_tool = tool
        if tool and tool.lower() in ['brush', 'smart_paint']:
            self.set_brush_channel(self.get_overlay_channel())
        
        # Set focus to capture keyboard events
        self.setFocus()

    def set_brush_channel(self, channel: str):
        """Set the current brush channel"""
        self.current_channel = channel

    def set_brush_radius(self, radius: float):
        """Set the brush radius"""
        self.brush_radius = radius
    
    def set_image_channel(self, channel: str):
        """Set the display channel for the image layer"""
        if channel not in IMAGE_CHANNEL_MAP.keys():
            return
        self.image_layer.channel = channel
        self.update_image_display()

    def set_overlay_channel(self, channel: str):
        """Set the display channel for the overlay layer"""
        if channel not in OVERLAY_MAP.keys():
            return
        self.overlay_layer.channel = channel
        self.update_overlay_display()

    def set_overlay_opacity(self, opacity: float):
        """Set overlay opacity (0.0 to 1.0) and update display"""
        self.overlay_layer.opacity = opacity
        self.update_overlay_display()

    ################ PUBLIC METHODS ################
    
    def update_image_display(self):
        """Update image layer display"""
        pixmap = self.image_layer.pixmap
        if pixmap:
            self.image_item.setPixmap(pixmap)
    
    def update_overlay_display(self):
        """Update overlay layer display"""
        pixmap = self.overlay_layer.pixmap
        if pixmap:
            self.overlay_item.setPixmap(pixmap)

    def undo(self):
        """Undo last operation"""
        if self.overlay_layer.can_undo():
            self.overlay_layer.undo()
            self.update_overlay_display()

    def redo(self):
        """Redo last undone operation"""
        if self.overlay_layer.can_redo():
            self.overlay_layer.redo()
            self.update_overlay_display()

    def reset_layers(self, image_layer: ImageLayer, overlay_layer: OverlayLayer):
        """Update canvas with new image and overlay layers without recreating the canvas"""
        self.image_layer = image_layer
        self.overlay_layer = overlay_layer
        
        # Update pixmap items with new layer pixmaps
        img_pixmap = image_layer.pixmap
        if img_pixmap:
            self.image_item.setPixmap(img_pixmap)
        
        overlay_pixmap = overlay_layer.pixmap
        if overlay_pixmap:
            self.overlay_item.setPixmap(overlay_pixmap)
        
        # Reset tool state
        self.current_tool = None
        self.stroke_points = []
    
    ################ PRIVATE METHODS ################
    
    def _handle_zoom(self, event):
        """Zoom in/out on mouse cursor position"""
        # Get zoom factor
        zoom_factor = 1.1 if event.angleDelta().y() > 0 else 0.9
        new_zoom = self._zoom_level * zoom_factor
        
        # Clamp to min/max
        new_zoom = max(self.MIN_ZOOM, min(self.MAX_ZOOM, new_zoom))
        
        # Get mouse position in scene
        scene_pos = self.mapToScene(event.position().toPoint())
        
        # Scale the view
        scale_factor = new_zoom / self._zoom_level
        self.scale(scale_factor, scale_factor)
        
        # Keep the scene position under the cursor
        new_mouse_scene_pos = self.mapToScene(event.position().toPoint())
        delta = new_mouse_scene_pos - scene_pos
        self.translate(delta.x(), delta.y())
        
        self._zoom_level = new_zoom
        event.accept()
    
    def _handle_vertical_scroll(self, event):
        """Scroll vertically"""
        scrollbar = self.verticalScrollBar()
        scrollbar.setValue(scrollbar.value() - event.angleDelta().y())
        event.accept()
    
    def _handle_horizontal_scroll(self, event):
        """Scroll horizontally (Shift + wheel)"""
        scrollbar = self.horizontalScrollBar()
        scrollbar.setValue(scrollbar.value() - event.angleDelta().y())
        event.accept()
    
    ################ EVENTS ################
    
    def mousePressEvent(self, event):
        """Start collecting stroke points"""
        if self._is_panning or self.current_tool is None:
            # Let parent class handle panning/default behavior
            super().mousePressEvent(event)
            return

        # Store temporary channels copy prior to stroke for undo
        self._temp_channels_copy = {
            k: v.copy() for k, v in self.overlay_layer._channels.items()
        }
        pos = self.mapToScene(event.position().toPoint())
        
        # Handle color switch tool (single click, no stroke)
        if self.current_tool == "change":
            command = ColorSwitchCommand(self.overlay_layer, self._temp_channels_copy, 
                                        int(pos.x()), int(pos.y()))
            self.overlay_layer._undo_stack.push(command)
            self.update_overlay_display()
            return
        
        self.stroke_points = [(pos.x(), pos.y())]

    def mouseMoveEvent(self, event):
        """Add points to current stroke"""
        if self._is_panning or self.current_tool is None or not self.stroke_points:
            super().mouseMoveEvent(event)
            return
        pos = self.mapToScene(event.position().toPoint())
        self.stroke_points.append((pos.x(), pos.y()))
        
        # Draw only the new segment (last 2 points)
        if len(self.stroke_points) >= 2:
            segment = self.stroke_points[-2:]
            if self.current_tool == "brush" and self.overlay_layer.channel != 'none':
                self.overlay_layer._draw_brush_segment(segment, self.brush_radius)
            elif self.current_tool == "smart_paint" and self.overlay_layer.channel != 'none':
                self.overlay_layer._draw_smart_paint_segment(segment, self.brush_radius)
            elif self.current_tool == "eraser" and self.overlay_layer.channel != 'none':
                self.overlay_layer._draw_erase_segment(segment, self.brush_radius)
            self.update_overlay_display()

    def mouseReleaseEvent(self, event):
        """Finalize the stroke by pushing to undo stack"""
        if self.current_tool is None or len(self.stroke_points) < 2:
            self.stroke_points = []
            super().mouseReleaseEvent(event)
            return
        
        # Push entire stroke to undo stack once
        if self.current_tool == "brush" and self.overlay_layer.channel != 'none':
            command = BrushStrokeCommand(self.overlay_layer, self._temp_channels_copy, 
                                         self.stroke_points, self.brush_radius)
            self.overlay_layer._undo_stack.push(command)
        elif self.current_tool == "smart_paint" and self.overlay_layer.channel in ['red', 'blue']:
            command = SmartPaintCommand(self.overlay_layer, self._temp_channels_copy, 
                                         self.stroke_points, self.brush_radius)
            self.overlay_layer._undo_stack.push(command)
        elif self.current_tool == "eraser" and self.overlay_layer.channel != 'none':
            command = EraseCommand(self.overlay_layer, self._temp_channels_copy, 
                                   self.stroke_points, self.brush_radius)
            self.overlay_layer._undo_stack.push(command)
        
        if self.current_tool != "smart_paint":
            self.overlay_layer._update_alpha_channel()
        self.stroke_points = []

    def wheelEvent(self, event):
        """Handle mouse wheel for zoom or scroll"""
        modifiers = event.modifiers()
        
        if modifiers == Qt.KeyboardModifier.ControlModifier:
            self._handle_horizontal_scroll(event)
        elif modifiers == Qt.KeyboardModifier.ShiftModifier:
            self._handle_vertical_scroll(event)
        else:
            self._handle_zoom(event)
            self.signal_zoom_changed.emit()

    def keyPressEvent(self, event):
        """Handle keyboard events"""
        # Spacebar for pan mode
        if event.key() == Qt.Key.Key_Space and self._edit_mode and not event.isAutoRepeat():
            self._is_panning = True
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            event.accept()
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        """Handle keyboard release"""
        # Exit pan mode when spacebar is released
        if event.key() == Qt.Key.Key_Space and self._edit_mode and not event.isAutoRepeat():
            self._is_panning = False
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            event.accept()
        else:
            super().keyReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        """Reset zoom to fit view on double click"""
        self.fitInView(self._scene.itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)
        transform = self.transform()
        self._zoom_level = transform.m11()
        event.accept()