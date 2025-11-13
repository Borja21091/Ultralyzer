from PySide6.QtGui import QImage, QPixmap, QPainter, QUndoStack, QColor
from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
from frontend.models.edit_commands import BrushStrokeCommand, EraseCommand, ColorSwitchCommand, SmartPaintCommand
from PySide6.QtCore import Qt, Signal
import numpy as np
import cv2

from definitions import IMAGE_CHANNEL_MAP, OVERLAY_MAP


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
    def pixmap(self) -> QPixmap:
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
    """Manages the editable overlay with channel selection"""
    
    def __init__(self, overlay_array: np.ndarray):
        """
        Args:
            overlay_array: RGB uint8 numpy array (H, W, 3) with arteries in red, veins in blue
        """
        self._overlay_array_original = overlay_array.copy() # Keep original for reset
        self._overlay_array = overlay_array  # RGB, kept for data persistence
        self._qimage = self._array_to_qimage(overlay_array) # RGBA for fast editing
        self._channel = "both"
        self._opacity = 0.75
        self._pixmap = None
        self._undo_stack = QUndoStack()
        self._update_display()
    
    ############ PROPERTIES ############
    
    @property
    def pixmap(self) -> QPixmap:
        return self._pixmap
    
    @property
    def channel(self) -> str:
        return self._channel
    
    @channel.setter
    def channel(self, value: str):
        """Set display channel: 'arteries', 'veins', 'both'"""
        valid_channels = OVERLAY_MAP.keys()
        if value not in valid_channels:
            return
        if self._channel == value:
            return
        self._channel = value
        self._update_display()
    
    @property
    def opacity(self) -> float:
        """Get overlay opacity (0.0 to 1.0)"""
        return self._opacity
    
    @opacity.setter
    def opacity(self, value: float):
        """Set overlay opacity (0.0 to 1.0)"""
        self._opacity = max(0.0, min(1.0, value))
        self._update_display()
    
    ############ GETTER/SETTER ############
    
    def get_array(self) -> np.ndarray:
        """Return the RGB numpy array (synced from QImage)"""
        self._sync_qimage_to_array()
        return self._overlay_array

    def get_switch_target_color(self, color_bgr: np.ndarray) -> np.ndarray:
        """
        Determine the target color after switch.
        BGR format: Red=(0,0,255), Blue=(255,0,0)
        
        Returns the new color in RGB format
        """
        # Check if predominantly red or blue
        b, g, r = color_bgr[0], color_bgr[1], color_bgr[2]
        
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
    
    def redo(self):
        """Redo last undone edit"""
        self._undo_stack.redo()
    
    def reset(self):
        """Reset overlay to original segmentation mask state"""
        # Restore from the original array
        self._qimage = self._array_to_qimage(self._overlay_array_original)
        self._undo_stack.clear()
    
    def can_undo(self) -> bool:
        """Check if undo is available"""
        return self._undo_stack.canUndo()
    
    def can_redo(self) -> bool:
        """Check if redo is available"""
        return self._undo_stack.canRedo()
    
    def perform_color_switch(self, x: int, y: int) -> bool:
        """
        Perform flood fill color switch at given coordinates.
        Red (artery) -> Blue (vein), Blue -> Red, Green/other -> Red
        
        Args:
            x, y: Pixel coordinates in the overlay
            
        Returns:
            True if a switch was performed, False if out of bounds or no color to switch
        """
        # Convert QImage to numpy array for flood fill
        ptr = self._qimage.bits()
        h, w = self._qimage.height(), self._qimage.width()
        byte_count = self._qimage.bytesPerLine() * h
        
        # Create numpy view of QImage (ARGB32)
        pixels = np.ndarray((byte_count,), dtype=np.uint8, buffer=ptr)
        pixels = pixels.reshape((h, w, 4))
        
        # Bounds check
        if x < 0 or x >= w or y < 0 or y >= h:
            return False
        
        # Get color at click position (BGRA format in memory)
        clicked_color = pixels[int(y), int(x), :3].copy() # BGR
        
        # If clicked on transparent (black), no action
        if np.all(clicked_color == 0):
            return False
        
        # Determine target color based on clicked color
        # BGR format: Red=(0,0,255), Blue=(255,0,0), Green=(0,255,0)
        target_color = self.get_switch_target_color(clicked_color)
        
        # Create a copy of pixels for flood fill mask
        # Use only the B and R channels to identify the color region
        mask = np.zeros((h, w), dtype=np.uint8)
        self._flood_fill_mask(pixels, mask, int(x), int(y), clicked_color, tolerance=5)
        
        # Apply color switch to all masked pixels
        pixels[mask > 0, :3] = target_color
        
        return True
    
    ############ PRIVATE METHODS ############
    
    def _array_to_qimage(self, rgb_array: np.ndarray) -> QImage:
        """Convert RGB numpy array to ARGB32 QImage"""
        h, w = rgb_array.shape[:2]
        
        # Create RGBA with alpha channel
        rgba = np.empty((h, w, 4), dtype=np.uint8)
        rgba[:, :, 0:3] = rgb_array[:, :, 0:3] # RGB channels

        # Alpha: transparent where all channels are 0
        rgba[:, :, 3] = np.bitwise_or(np.bitwise_or(rgba[:, :, 0], rgba[:, :, 1]), rgba[:, :, 2])
        
        # Swap to ABGR for QImage
        abgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)

        bytes_per_line = 4 * w
        q_image = QImage(abgr.tobytes(), w, h, bytes_per_line, QImage.Format.Format_ARGB32)
        
        return q_image.copy()  # Detach from numpy buffer
    
    def _update_display(self):
        """Update pixmap based on current channel selection"""
        # Create a copy to apply channel filtering
        if self._channel == "none":
            transparent = QPixmap(self._qimage.width(), self._qimage.height())
            transparent.fill(Qt.GlobalColor.transparent)
            self._pixmap = transparent
            return

        # Create pixmap and apply filtering in one step
        self._pixmap = self._qimage_to_filtered_pixmap()

    def _qimage_to_filtered_pixmap(self) -> QPixmap:
        """Convert QImage to filtered QPixmap - where filtering actually happens"""
        display_image = self._qimage.copy()
        
        if self._channel == "arteries":
            self._filter_channel(display_image, keep_red=True, keep_blue=False)
        elif self._channel == "veins":
            self._filter_channel(display_image, keep_red=False, keep_blue=True)
        # else: "both" - keep as is
        
        self._apply_opacity(display_image)
        
        return QPixmap.fromImage(display_image)
    
    def _apply_opacity(self, qimage: QImage):
        """Modify QImage alpha channel based on opacity setting"""
        ptr = qimage.bits()
        h, w = qimage.height(), qimage.width()
        byte_count = qimage.bytesPerLine() * h
        
        # Create numpy view of pixel data (ARGB32)
        pixels = np.ndarray((byte_count,), dtype=np.uint8, buffer=ptr)
        pixels = pixels.reshape((h, w, 4))
        
        # Scale alpha channel by opacity
        pixels[:, :, 3] = (pixels[:, :, 3] * self.opacity).astype(np.uint8)
    
    def _filter_channel(self, qimage: QImage, keep_red: bool, keep_blue: bool):
        """Modify QImage to show only selected channels"""
        # Get writable buffer
        ptr = qimage.bits()
        
        h, w = qimage.height(), qimage.width()
        byte_count = qimage.bytesPerLine() * h
        
        # Create a writable numpy view without copying
        pixels = np.ndarray((byte_count,), dtype=np.uint8, buffer=ptr)
        pixels = pixels.reshape((h, w, 4))
        
        # Zero out channels we don't want to keep
        if not keep_red:
            pixels[:, :, 2] = 0 # Red channel
        if not keep_blue:
            pixels[:, :, 0] = 0 # Blue channel
        
        # Update alpha based on remaining visible channels
        pixels[:, :, 3] = np.bitwise_or(np.bitwise_or(pixels[:, :, 0], pixels[:, :, 1]), pixels[:, :, 2])

    def _draw_brush_segment(self, points, radius: int):
        """Draw a single line segment on the overlay"""
        painter = QPainter(self._qimage)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = painter.pen()
        pen.setWidth(2 * radius)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        
        if self.channel == 'arteries':
            pen.setColor(Qt.GlobalColor.red)
        elif self.channel == 'veins':
            pen.setColor(Qt.GlobalColor.blue)
        elif self.channel == 'both':
            pen.setColor(QColor(255, 0, 255))
            
        painter.setPen(pen)
        painter.drawLine(int(points[0][0]), int(points[0][1]), int(points[1][0]), int(points[1][1]))
        painter.end()

    def _draw_erase_segment(self, points, radius: int):
        """Erase a single line segment"""
        painter = QPainter(self._qimage)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
        pen = painter.pen()
        pen.setWidth(2 * radius)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.drawLine(int(points[0][0]), int(points[0][1]), int(points[1][0]), int(points[1][1]))
        painter.end()
    
    def _draw_smart_paint_segment(self, points, radius: int):
        """
        Draw a smart paint segment: paint over existing vessels only.
        Only non-black pixels under the brush stroke are painted with the target color.
        """
        # Determine target color based on channel (avoid repeated if checks)
        color_map = {
            'both': (255, 255),      # Magenta: B=255, R=255
            'arteries': (0, 255),    # Red: B=0, R=255
            'veins': (255, 0)        # Blue: B=255, R=0
        }
        
        if self.channel not in color_map:
            return
        
        b_val, r_val = color_map[self.channel]
        
        # Get overlay pixels directly (avoid creating temp QImage)
        overlay_ptr = self._qimage.bits()
        h, w = self._qimage.height(), self._qimage.width()
        overlay_pixels = np.ndarray((h, w, 4), dtype=np.uint8, buffer=overlay_ptr)
        
        # Use Qt's built-in rasterization instead of creating a separate QImage
        # Draw stroke directly to a numpy mask in one pass
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Vectorized line rasterization using Bresenham-like approach
        x0, y0 = int(points[0][0]), int(points[0][1])
        x1, y1 = int(points[1][0]), int(points[1][1])
        
        # Use cv2 for efficient line drawing (already imported)
        cv2.line(mask, (x0, y0), (x1, y1), 255, int(2 * radius), cv2.LINE_AA)
        
        # Create combined mask: stroke AND non-black pixels
        non_black = np.bitwise_or(np.bitwise_or(overlay_pixels[:, :, 0] > 0, overlay_pixels[:, :, 1] > 0),
                                  overlay_pixels[:, :, 2] > 0)
        paint_area = (mask > 0) & non_black
        
        # Apply color in one vectorized operation
        overlay_pixels[paint_area, 0] = b_val
        overlay_pixels[paint_area, 2] = r_val
            
        
        # Update alpha channel only for modified pixels (batch operation)
        overlay_pixels[paint_area, 3] = 255
    
    def _sync_qimage_to_array(self):
        """Convert QImage back to RGB numpy array for data persistence"""
        ptr = self._qimage.bits()
        buf = bytes(ptr)
        
        # Reshape from ARGB32 to (height, width, 4) [R G B A]
        h, w = self._qimage.height(), self._qimage.width()
        argb = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))
        
        # Extract RGB channels and update numpy array
        self._overlay_array[:, :, 0:3] = argb[:, :, 0:3]
          
    def _flood_fill_mask(self, pixels: np.ndarray, mask: np.ndarray, 
                         x: int, y: int, target_color: np.ndarray, tolerance: int = 5):
        """
        BFS flood fill to mark all connected pixels of similar color.
        
        Args:
            pixels: ARGB32 pixel array (H, W, 4)
            mask: Output mask to mark filled pixels
            x, y: Starting coordinates
            target_color: BGR color to match
            tolerance: Color tolerance for matching
        """
        h, w = pixels.shape[:2]
        visited = np.zeros((h, w), dtype=bool)
        queue = [(x, y)]
        visited[y, x] = True
        
        while queue:
            cx, cy = queue.pop(0)
            
            # Check if current pixel matches target color (within tolerance)
            if self._color_matches(pixels[cy, cx, :3], target_color, tolerance):
                mask[cy, cx] = 1
                
                # Add neighbors to queue
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((nx, ny))
                        
    def _color_matches(self, color1_bgr: np.ndarray, color2_bgr: np.ndarray, tolerance: int = 0) -> np.bool | bool:
        """
        Check if two colors are similar within tolerance.
        
        Args:
            color1_bgr, color2_bgr: Colors in BGR format
            tolerance: Maximum difference per channel
        """
        # Skip fully transparent pixels
        if np.all(color1_bgr == 0):
            return False
        
        diff = np.abs(color1_bgr.astype(int) - color2_bgr.astype(int))
        return np.all(diff <= tolerance)
    
    def _update_qimage_after_edit(self):
        """Update internal state after direct pixel manipulation"""
        # Force QImage to recognize changes
        self._qimage = self._array_to_qimage(self.get_array())
        self._update_display()
    

class Canvas(QGraphicsView):
    """Main canvas for displaying and editing image layers"""
    
    signal_zoom_changed = Signal()
    
    def __init__(self, image_layer: ImageLayer, overlay_layer: OverlayLayer):
        super().__init__()
        self.image_layer = image_layer
        self.overlay_layer = overlay_layer
        
        # Create scene
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        
        # Create pixmap items for each layer
        self.image_item = QGraphicsPixmapItem(image_layer.pixmap)
        self.overlay_item = QGraphicsPixmapItem(overlay_layer.pixmap)
        
        # Add items to scene (image first, overlay on top)
        self.scene.addItem(self.image_item)
        self.scene.addItem(self.overlay_item)
        
        # Position overlay on top of image
        self.overlay_item.setPos(0, 0)
        
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
        self._temp_overlay_qimage = None  # Temporary QImage for undo commands
    
    ################ GETTER/SETTER ################
    
    def get_image_channel(self) -> str:
        """Get the current image layer channel"""
        return self.image_layer.channel

    def get_overlay_channel(self) -> str:
        """Get the current overlay layer channel"""
        return self.overlay_layer.channel
    
    def set_tool(self, tool: str):
        """Set the current tool: 'brush', 'smart_paint', 'eraser' or 'change'"""
        self.current_tool = tool
        if tool and tool.lower() in ['brush', 'smart_paint']:
            self.set_brush_channel(self.get_overlay_channel())

    def set_brush_channel(self, channel: str):
        """Set the current brush channel: 'arteries' or 'veins'"""
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
        self.image_item.setPixmap(self.image_layer.pixmap)
    
    def update_overlay_display(self):
        """Update overlay layer display"""
        self.overlay_layer._update_display()
        self.overlay_item.setPixmap(self.overlay_layer.pixmap)

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
        self.image_item.setPixmap(image_layer.pixmap)
        self.overlay_item.setPixmap(overlay_layer.pixmap)
        
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
        if self.current_tool is None:
            # Let parent class handle panning/default behavior
            super().mousePressEvent(event)
            return

        # Store temporary qimage prior to stroke for undo
        self._temp_overlay_qimage = self.overlay_layer._qimage.copy()
        pos = self.mapToScene(event.position().toPoint())
        
        # Handle color switch tool (single click, no stroke)
        if self.current_tool == "change":
            command = ColorSwitchCommand(self.overlay_layer, self._temp_overlay_qimage, 
                                        int(pos.x()), int(pos.y()))
            self.overlay_layer._undo_stack.push(command)
            self.update_overlay_display()
            return
        
        self.stroke_points = [(pos.x(), pos.y())]

    def mouseMoveEvent(self, event):
        """Add points to current stroke"""
        if self.current_tool is None or not self.stroke_points:
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
            elif self.current_tool == "eraser" and self.overlay_layer.channel == 'both':
                self.overlay_layer._draw_erase_segment(segment, self.brush_radius)
            self.update_overlay_display()

    def mouseReleaseEvent(self, event):
        """Finalize the stroke by pushing to undo stack"""
        if self.current_tool is None or len(self.stroke_points) < 2:
            self.stroke_points = []
            return
        
        # Push entire stroke to undo stack once
        if self.current_tool == "brush" and self.overlay_layer.channel != 'none':
            command = BrushStrokeCommand(self.overlay_layer, self._temp_overlay_qimage, 
                                         self.stroke_points, self.overlay_layer.channel, 
                                         self.brush_radius)
            self.overlay_layer._undo_stack.push(command)
        elif self.current_tool == "smart_paint" and self.overlay_layer.channel != 'none':
            command = SmartPaintCommand(self.overlay_layer, self._temp_overlay_qimage, 
                                         self.stroke_points, self.overlay_layer.channel, 
                                         self.brush_radius)
            self.overlay_layer._undo_stack.push(command)
        elif self.current_tool == "eraser" and self.overlay_layer.channel == 'both':
            command = EraseCommand(self.overlay_layer, self._temp_overlay_qimage, 
                                   self.stroke_points, self.brush_radius)
            self.overlay_layer._undo_stack.push(command)
        
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
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._is_panning = True
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            event.accept()
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        """Handle keyboard release"""
        # Exit pan mode when spacebar is released
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._is_panning = False
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
        else:
            super().keyReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        """Reset zoom to fit view on double click"""
        self.fitInView(self.scene.itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self._zoom_level = 1.0
        event.accept()

