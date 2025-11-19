from PySide6.QtGui import QUndoCommand
from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    from frontend.widgets.canvas import ChannelBuffer
    from frontend.widgets.canvas import OverlayLayer


class BrushStrokeCommand(QUndoCommand):
    """Undo command for brush stroke operations"""
    
    def __init__(self, overlay_layer: 'OverlayLayer', channels_before: Optional[Dict[str, 'ChannelBuffer']], 
                 points: list, radius: float, description: str = "Brush Stroke"):
        super().__init__(description)
        self.overlay_layer = overlay_layer
        self.channels_before = channels_before
        self.points = points
        self.radius = radius
        self._already_applied = True
    
    def redo(self):
        """Reapply the brush stroke"""
        if self._already_applied:
            self._already_applied = False
            return
        
        # Redraw the stroke
        for i in range(len(self.points) - 1):
            segment = self.points[i:i+2]
            self.overlay_layer._draw_brush_segment(segment, self.radius)
        
        # Mark display as dirty
        self.overlay_layer._pixmap_dirty = True
    
    def undo(self):
        """Restore channels to state before brush stroke"""
        if self.channels_before is not None:
            self.overlay_layer._channels = {
                k: v.copy() for k, v in self.channels_before.items()
            }
            self.overlay_layer._pixmap_dirty = True


class EraseCommand(QUndoCommand):
    """Undo command for erase operations"""
    
    def __init__(self, overlay_layer: 'OverlayLayer', channels_before: Optional[Dict[str, 'ChannelBuffer']], 
                 points: list, radius: float, description: str = "Erase"):
        super().__init__(description)
        self.overlay_layer = overlay_layer
        self.channels_before = channels_before
        self.points = points
        self.radius = radius
        self._already_applied = True
    
    def redo(self):
        """Reapply the erase operation"""
        if self._already_applied:
            self._already_applied = False
            return
        
        # Redraw the erase
        for i in range(len(self.points) - 1):
            segment = self.points[i:i+2]
            self.overlay_layer._draw_erase_segment(segment, self.radius)
        
        self.overlay_layer._pixmap_dirty = True
    
    def undo(self):
        """Restore channels to state before erase"""
        if self.channels_before is not None:
            self.overlay_layer._channels = {
                k: v.copy() for k, v in self.channels_before.items()
            }
            self.overlay_layer._pixmap_dirty = True


class ColorSwitchCommand(QUndoCommand):
    """Undo command for color switch operations"""
    
    def __init__(self, overlay_layer: 'OverlayLayer', channels_before: Optional[Dict[str, 'ChannelBuffer']], 
                 x: int, y: float, description: str = "Color Switch"):
        super().__init__(description)
        self.overlay_layer = overlay_layer
        self.channels_before = channels_before
        self.x = x
        self.y = y
    
    def redo(self):
        """Reapply the color switch"""
        self.overlay_layer.perform_color_switch(int(self.x), int(self.y))
        self.overlay_layer._pixmap_dirty = True
    
    def undo(self):
        """Restore channels to state before color switch"""
        if self.channels_before is not None:
            self.overlay_layer._channels = {
                k: v.copy() for k, v in self.channels_before.items()
            }
            self.overlay_layer._pixmap_dirty = True


class SmartPaintCommand(QUndoCommand):
    """Undo command for smart paint operations"""
    
    def __init__(self, overlay_layer: 'OverlayLayer', channels_before: Optional[Dict[str, 'ChannelBuffer']], 
                 points: list, radius: float, description: str = "Smart Paint"):
        super().__init__(description)
        self.overlay_layer = overlay_layer
        self.channels_before = channels_before
        self.points = points
        self.radius = radius
        self._already_applied = True
    
    def redo(self):
        """Reapply the smart paint stroke"""
        if self._already_applied:
            self._already_applied = False
            return
        
        # Redraw all segments of the stroke
        for i in range(len(self.points) - 1):
            segment = self.points[i:i+2]
            self.overlay_layer._draw_smart_paint_segment(segment, self.radius)
        
        self.overlay_layer._pixmap_dirty = True
    
    def undo(self):
        """Restore channels to state before smart paint"""
        if self.channels_before is not None:
            self.overlay_layer._channels = {
                k: v.copy() for k, v in self.channels_before.items()
            }
            self.overlay_layer._pixmap_dirty = True