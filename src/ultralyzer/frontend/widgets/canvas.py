from PySide6.QtGui import QImage, QPixmap, QPainter, QUndoStack, QPen, QColor, QBrush, QPainterPath
from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsEllipseItem, QGraphicsPathItem
from frontend.models.edit_commands import BrushStrokeCommand, EraseCommand, ColorSwitchCommand, SmartPaintCommand
from PySide6.QtCore import Qt, Signal, QPointF
import numpy as np
import cv2
from definitions import IMAGE_CHANNEL_MAP, OVERLAY_MAP
from typing import Optional, Dict

from PySide6.QtOpenGLWidgets import QOpenGLWidget


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
            
            # Ensure contiguous and uint8
            if not channel_array.flags['C_CONTIGUOUS']:
                channel_array = np.ascontiguousarray(channel_array)
            if channel_array.dtype != np.uint8:
                channel_array = channel_array.astype(np.uint8)
                
            # QImage Format_Grayscale8: 1 byte per pixel
            # Must align bytes_per_line to 4 bytes for QImage if we were creating it manually,
            # but passing data directly lets QImage handle it or we specify stride.
            # Ideally, let's use the array's stride if compatible, or copy.
            bytes_per_line = w  # Assuming tightly packed for input
            
            # Note: QImage constructor with data does not copy. We must copy() to own the data.
            q_img = QImage(channel_array.tobytes(), w, h, bytes_per_line, QImage.Format.Format_Grayscale8)
            self._qimage = q_img.copy()
        else:
            # Create empty buffer
            if width is None or height is None:
                raise ValueError("Must provide either channel_array or (width, height)")
            self._qimage = QImage(width, height, QImage.Format.Format_Grayscale8)
            self._qimage.fill(0)
    
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
        bytes_per_line = self._qimage.bytesPerLine()
        
        # Map the whole buffer including padding
        pixels = np.ndarray((h, bytes_per_line), dtype=np.uint8, buffer=ptr)
        
        # Return view of the valid width
        return pixels[:, :w]
    
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
        
        # Get views (handling stride)
        r_data = r_buffer.numpy_view
        g_data = g_buffer.numpy_view
        b_data = b_buffer.numpy_view
        a_data = a_buffer.numpy_view
        
        # Stack into BGRA (Qt uses BGRA byte order for Format_ARGB32 on Little Endian)
        # This avoids cv2.cvtColor(RGBA -> BGRA)
        bgra = np.dstack((b_data, g_data, r_data, a_data))
        
        # Create QImage
        # We need to ensure the data persists or is copied. 
        # QImage(bytes, ...) does not copy.
        q_img = QImage(bgra.tobytes(), w, h, QImage.Format.Format_ARGB32)
        return q_img.copy()


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
        
        self._composed_qimage: Optional[QImage] = None
        
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
    
    ############ GETTER/SETTER ############
    
    def get_array(self) -> np.ndarray:
        """Return the RGB numpy array (composed from channel buffers)"""
        r_data = self._channels['r'].numpy_view
        g_data = self._channels['g'].numpy_view
        b_data = self._channels['b'].numpy_view
        
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
        if not self._undo_stack.canUndo():
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
        
        # Get pixel colors at click position using numpy views
        r_view = self._channels['r'].numpy_view
        g_view = self._channels['g'].numpy_view
        b_view = self._channels['b'].numpy_view
        
        r_val = r_view[int(y), int(x)]
        g_val = g_view[int(y), int(x)]
        b_val = b_view[int(y), int(x)]
        
        clicked_color = np.array([b_val, g_val, r_val], dtype=np.uint8)
        
        # If the clicked pixel doesn't contain red or blue, no action
        if r_val != 255 and b_val != 255:
            return False
        
        # Determine target color based on clicked color (BGR format)
        target_color = self.get_switch_target_color(clicked_color)
        target_b, _, target_r = target_color[0], target_color[1], target_color[2]
        
        # Determine which channel to flood fill (the source channel)
        if r_val > b_val:
            # Clicked Red (Artery), fill only Red component
            source_view = r_view
        elif b_val > r_val:
            # Clicked Blue (Vein), fill only Blue component
            source_view = b_view
        else:
            # Ambiguous (Purple). Default to Blue -> Red switch (fill Blue)
            source_view = b_view
        
        # Use OpenCV floodFill for performance
        # Mask must be 2 pixels larger than image
        mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        
        # Flood fill on the SPECIFIC channel view
        # flags=8 means 8-connected. (255 << 8) fills the mask with 255.
        # FLOODFILL_MASK_ONLY prevents modifying the source image.
        flags = 8 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY
        
        # We use the source_view (R or B) as the reference image.
        # It will only fill connected pixels of that specific color.
        cv2.floodFill(source_view, mask, (int(x), int(y)), 255, loDiff=0, upDiff=0, flags=flags)
        
        # Extract the relevant part of the mask (remove padding)
        fill_mask = mask[1:-1, 1:-1].astype(bool)
        
        # Apply color switch to all masked pixels in R, B channels
        # We write directly to the numpy views which map to QImage memory
        if target_r > 0:
            r_view[fill_mask] = 255
        else:
            r_view[fill_mask] = 0
            
        if target_b > 0:
            b_view[fill_mask] = 255
        else:
            b_view[fill_mask] = 0
        
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
            self._composed_qimage = None
            return
        
        # Compose channels based on selection
        self._composed_qimage = self._compose_channels()
        
        # Convert to pixmap
        self._pixmap = QPixmap.fromImage(self._composed_qimage)
        self._pixmap_dirty = False
    
    def _compose_vessels_mode(self) -> QImage:
        """
        Optimized composition for 'vessels' mode (Red + Blue).
        Calculates Alpha = R | B on the fly without intermediate object creation.
        """
        w, h = self._channels['r'].get_dimensions()
        
        # 1. Get direct views of R and B channels
        r_data = self._channels['r'].numpy_view
        b_data = self._channels['b'].numpy_view
        
        # 2. Calculate Alpha and Green
        a_data = np.bitwise_or(r_data, b_data)
        g_data = np.zeros_like(r_data)
        
        # 3. Stack into BGRA (Qt uses BGRA byte order for Format_ARGB32)
        bgra = np.dstack((b_data, g_data, r_data, a_data))
        
        # 4. Create QImage
        q_img = QImage(bgra.tobytes(), w, h, QImage.Format.Format_ARGB32)
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
    
    def _draw_pen_stroke(self, pen: QPen, 
                         points: list,
                         mode: QPainter.CompositionMode = QPainter.CompositionMode.CompositionMode_SourceOver,
                         update_alpha: bool = True):
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
            painter.setCompositionMode(mode)
            painter.setPen(pen)
            for i in range(len(points) - 1):
                painter.drawLine(int(points[i][0]), int(points[i][1]), int(points[i+1][0]), int(points[i+1][1]))
            painter.end()
            
        if update_alpha:
            # OPTIMIZATION: Paint directly on alpha channel for instant feedback
            painter = QPainter(self._channels['a']._qimage)
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

        # Paint on selected channels (skip alpha update as we use incremental paint for display)
        self._draw_pen_stroke(pen, points, update_alpha=False)
        
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
        
        if len(points) < 2:
            return

        # 1. Calculate Bounding Box (ROI)
        pad = int(radius) + 2
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        
        w, h = self._channels['r'].get_dimensions()
        
        min_x = max(0, int(min(x_coords)) - pad)
        max_x = min(w, int(max(x_coords)) + pad)
        min_y = max(0, int(min(y_coords)) - pad)
        max_y = min(h, int(max(y_coords)) + pad)
        
        w_roi = max_x - min_x
        h_roi = max_y - min_y
        
        if w_roi <= 0 or h_roi <= 0:
            return
            
        bbox = (min_x, min_y, max_x, max_y)

        # 2. Create Local Mask using QPainter
        mask_image = QImage(w_roi, h_roi, QImage.Format.Format_Grayscale8)
        mask_image.fill(0)
        
        painter = QPainter(mask_image)
        painter.setPen(QPen(Qt.GlobalColor.white, int(2 * radius), 
                           Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        
        # Convert points to local coordinates
        local_points = [QPointF(p[0] - min_x, p[1] - min_y) for p in points]
        painter.drawPolyline(local_points)
        painter.end()
        
        # Convert QImage to numpy array
        ptr = mask_image.bits()
        bytes_per_line = mask_image.bytesPerLine()
        
        if bytes_per_line != w_roi:
             full_buffer = np.ndarray((h_roi, bytes_per_line), dtype=np.uint8, buffer=ptr)
             mask_roi = full_buffer[:, :w_roi]
        else:
             mask_roi = np.ndarray((h_roi, w_roi), dtype=np.uint8, buffer=ptr)

        # 3. Get Local Views of Channels
        r_roi = self._get_roi_view('r', bbox)
        b_roi = self._get_roi_view('b', bbox)
        
        # 4. Local Logic
        # Check R or B > 0 in ROI
        vessel_roi = np.bitwise_or(r_roi > 0, b_roi > 0)
        paint_area = (mask_roi > 0) & vessel_roi
        
        # 5. Write Result Locally
        if self.channel == 'red':
            r_roi[paint_area] = 255
            b_roi[paint_area] = 0
        elif self.channel == 'blue':
            r_roi[paint_area] = 0
            b_roi[paint_area] = 255
    
    def _get_roi_view(self, channel_name: str, bbox: tuple):
        """Get a numpy view of a channel buffer for the specified ROI"""
        min_x, min_y, max_x, max_y = bbox
        full_arr = self._channels[channel_name].numpy_view
        return full_arr[min_y:max_y, min_x:max_x]
    
    def _update_alpha_channel(self):
        """Regenerate alpha channel based on current R, G, B state"""
        r_data = self._channels['r'].numpy_view
        g_data = self._channels['g'].numpy_view
        b_data = self._channels['b'].numpy_view
        
        # Compute alpha: 255 where any channel is non-zero
        alpha = np.bitwise_or(np.bitwise_or(r_data, g_data), b_data).astype(np.uint8)
        
        # Update alpha channel
        self._channels['a'] = ChannelBuffer(alpha)


class Canvas(QGraphicsView):
    """Main canvas for displaying and editing image layers"""
    
    signal_zoom_changed = Signal()
    signal_fovea_selected = Signal(float, float)  # x, y coordinates of fovea selection
    
    def __init__(self, image_layer: Optional[ImageLayer]=None, overlay_layer: Optional[OverlayLayer]=None):
        super().__init__()
        
        # Enable hardware acceleration
        self.setViewport(QOpenGLWidget())
        
        self.image_layer = image_layer
        self.overlay_layer = overlay_layer
        
        # Create scene
        self._scene = QGraphicsScene()
        self.setScene(self._scene)
        
        # Create pixmap items for each layer
        self.image_item = QGraphicsPixmapItem(image_layer.pixmap) if image_layer else QGraphicsPixmapItem()
        self.overlay_item = QGraphicsPixmapItem(overlay_layer.pixmap) if overlay_layer else QGraphicsPixmapItem()
        
        # Add items to scene (image first, overlay on top)
        self._scene.addItem(self.image_item)
        self._scene.addItem(self.overlay_item)
        
        # Position overlay on top of image
        self.overlay_item.setPos(0, 0)
        
        # Create temporary stroke item for hardware accelerated preview
        self._temp_stroke_item = QGraphicsPathItem()
        self._temp_stroke_item.setZValue(100) # On top of everything
        self._scene.addItem(self._temp_stroke_item)
        self._temp_stroke_item.setVisible(False)
        self._current_path = QPainterPath()

        # Create fovea marker (hidden by default)
        self.fovea_item = QGraphicsEllipseItem(0, 0, 20, 20) # 20px diameter
        color = QColor(67, 220, 250)
        self.fovea_item.setPen(QPen(color, 2))
        self.fovea_item.setBrush(QBrush(color))
        self.fovea_item.setZValue(10) # Ensure it's on top
        self.fovea_item.setVisible(False)
        self._scene.addItem(self.fovea_item)
        
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
        """Set the current tool: 'brush', 'smart_paint', 'eraser', 'change' or 'fovea_location'"""
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
        self.overlay_item.setOpacity(opacity)

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

    def update_fovea(self, x: float, y: float):
        """Update fovea marker position"""
        if x is not None and y is not None:
            # Center the 20x20 circle on the coordinates
            self.fovea_item.setRect(x - 10, y - 10, 20, 20)
            self.set_fovea_visibility(True)
    
    def set_fovea_visibility(self, visible: bool):
        """Set fovea marker visibility"""
        self.fovea_item.setVisible(visible)
    
    ################ PRIVATE METHODS ################
    
    def _setup_preview_stroke(self):
        """Configure the vector path item for the current tool"""
        self._current_path = QPainterPath()
        
        # Determine color
        if self.current_tool == "eraser":
            # Eraser preview: Semi-transparent black
            color = QColor(0, 0, 0, 128) 
        else:
            # Brush/Smart Paint: Use channel color
            c = self.overlay_layer.channel
            if c == 'red': 
                color = QColor(255, 0, 0)
            elif c == 'blue': 
                color = QColor(0, 0, 255)
            elif c == 'green': 
                color = QColor(0, 255, 0)
            elif c == 'vessels': 
                color = QColor(255, 0, 255) 
            else: 
                color = QColor(255, 255, 255)
            # Dial down alpha if smart-paint
            if self.current_tool == "smart_paint":
                color.setAlpha(128)
                
        
        pen = QPen(color)
        pen.setWidth(int(2 * self.brush_radius))
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        
        self._temp_stroke_item.setPen(pen)
        self._temp_stroke_item.setPath(self._current_path)
        self._temp_stroke_item.setVisible(True)

    def _update_preview_stroke(self, x, y, start=False):
        if start:
            self._current_path.moveTo(x, y)
        else:
            self._current_path.lineTo(x, y)
        self._temp_stroke_item.setPath(self._current_path)

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
        
        # Handle fovea location tool (single click, no stroke)
        if self.current_tool == "fovea_location":
            self.update_fovea(pos.x(), pos.y())
            self.signal_fovea_selected.emit(pos.x(), pos.y())
            return
        
        self.stroke_points = [(pos.x(), pos.y())]

        # Setup hardware accelerated preview
        if self.current_tool in ["brush", "smart_paint", "eraser"]:
            self._setup_preview_stroke()
            self._update_preview_stroke(pos.x(), pos.y(), start=True)

    def mouseMoveEvent(self, event):
        """Add points to current stroke"""
        if self._is_panning or self.current_tool is None or not self.stroke_points:
            super().mouseMoveEvent(event)
            return
        pos = self.mapToScene(event.position().toPoint())
        self.stroke_points.append((pos.x(), pos.y()))
        
        # Update preview instead of rasterizing immediately
        if self.current_tool in ["brush", "smart_paint", "eraser"]:
            self._update_preview_stroke(pos.x(), pos.y())

    def mouseReleaseEvent(self, event):
        """Finalize the stroke by pushing to undo stack"""
        # Hide preview
        self._temp_stroke_item.setVisible(False)
        self._temp_stroke_item.setPath(QPainterPath())

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
        
        # Ensure display is updated after the command runs
        self.update_overlay_display()
        
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