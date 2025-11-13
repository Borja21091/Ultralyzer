from PySide6.QtGui import QImage, QPainter, QColor, Qt, QUndoCommand
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from frontend.widgets.canvas import OverlayLayer

class BrushStrokeCommand(QUndoCommand):
    """Undo command for brush stroke operations"""
    
    def __init__(self, overlay_layer: 'OverlayLayer', temp_qimage: QImage, points: list, channel: str, radius: int, description: str = "Brush Stroke"):
        super().__init__(description)
        self.overlay_layer = overlay_layer  # Store reference to overlay layer
        self.points = points
        self.channel = channel
        self.radius = radius
        self.image_before = temp_qimage
        self._already_applied = True
    
    def redo(self):
        """Reapply the brush stroke"""
        # Only redraw if this wasn't already drawn incrementally
        if self._already_applied:
            self._already_applied = False
            return
        
        painter = QPainter(self.overlay_layer._qimage)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        pen = painter.pen()
        pen.setWidth(2 * self.radius)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        if self.channel == "arteries":
            pen.setColor(Qt.GlobalColor.red)
        elif self.channel == "veins":
            pen.setColor(Qt.GlobalColor.blue)
        elif self.channel == "both":
            pen.setColor(QColor(255, 0, 255))
        painter.setPen(pen)
        
        for i in range(len(self.points) - 1):
            painter.drawLine(int(self.points[i][0]), int(self.points[i][1]), 
                           int(self.points[i+1][0]), int(self.points[i+1][1]))
        painter.end()
    
    def undo(self):
        """Restore image to state before brush stroke"""
        self.overlay_layer._qimage = self.image_before.copy()


class EraseCommand(QUndoCommand):
    """Undo command for erase operations"""
    
    def __init__(self, overlay_layer: 'OverlayLayer', temp_qimage: QImage, points: list, radius: int, description: str = "Erase"):
        super().__init__(description)
        self.overlay_layer = overlay_layer  # Store reference to overlay layer
        self.points = points
        self.radius = radius
        self.image_before = temp_qimage
        self._already_applied = True
    
    def redo(self):
        """Reapply the erase operation"""
        # Only redraw if this wasn't already drawn incrementally
        if self._already_applied:
            self._already_applied = False
            return
        
        painter = QPainter(self.overlay_layer._qimage)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
        
        pen = painter.pen()
        pen.setWidth(2 * self.radius)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        
        for i in range(len(self.points) - 1):
            painter.drawLine(int(self.points[i][0]), int(self.points[i][1]), 
                           int(self.points[i+1][0]), int(self.points[i+1][1]))
        painter.end()
    
    def undo(self):
        """Restore image to state before erase"""
        self.overlay_layer._qimage = self.image_before.copy()


class ColorSwitchCommand(QUndoCommand):
    """Undo command for color switch operations"""
    
    def __init__(self, overlay_layer: 'OverlayLayer', temp_qimage: QImage, x: int, y: int, description: str = "Color Switch"):
        super().__init__(description)
        self.overlay_layer = overlay_layer
        self.x = x
        self.y = y
        self.image_before = temp_qimage
    
    def redo(self):
        """Reapply the color switch"""
        self.overlay_layer.perform_color_switch(self.x, self.y)
    
    def undo(self):
        """Restore image to state before color switch"""
        self.overlay_layer._qimage = self.image_before.copy()


class SmartPaintCommand(QUndoCommand):
    """Undo command for smart paint operations"""
    
    def __init__(self, overlay_layer: 'OverlayLayer', temp_qimage: QImage, points: list, channel: str, radius: int, description: str = "Smart Paint"):
        super().__init__(description)
        self.overlay_layer = overlay_layer
        self.points = points
        self.channel = channel
        self.radius = radius
        self.image_before = temp_qimage
        self._already_applied = True
    
    def redo(self):
        """Reapply the smart paint stroke"""
        # Only redraw if this wasn't already drawn incrementally
        if self._already_applied:
            self._already_applied = False
            return
        
        # Redraw all segments of the stroke
        for i in range(len(self.points) - 1):
            segment = self.points[i:i+2]
            self.overlay_layer._draw_smart_paint_segment(segment, self.radius)
    
    def undo(self):
        """Restore image to state before smart paint"""
        self.overlay_layer._qimage = self.image_before.copy()
