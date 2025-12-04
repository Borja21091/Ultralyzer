from pathlib import Path
from typing import List
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel
)
from PySide6.QtGui import QFont
from PySide6.QtCore import Qt, Signal
from definitions import IMAGE_FORMATS, CANVAS_BORDER_COLORS
from frontend.widgets.canvas import Canvas
from backend.models.database import DatabaseManager
from backend.steps.segmentation import SegmentationStep


class BaseWidget(QWidget):
    """Base widget with common functionality for processing steps"""

    ############ SIGNALS ############
    
    index_changed = Signal(int) # index of current image
    
    def __init__(self, db_manager: DatabaseManager = None):
        super().__init__()
        self.db_manager = db_manager or DatabaseManager()
        self.image_paths: List[Path] = []
        self._index = 0
        self.canvas = None
        self.step_seg: SegmentationStep = None  # To be defined in subclasses
    
    ############ PROPERTIES ############
    
    @property
    def index(self) -> int:
        """Get current image index"""
        return self._index
    
    @index.setter
    def index(self, value: int):
        self._index = value

    @property
    def image_count(self) -> int:
        """Get total number of images"""
        return len(self.image_paths)

    @property
    def image_path(self) -> Path:
        """Get path of current image"""
        if 0 <= self.index < len(self.image_paths):
            return self.image_paths[self.index]
        return None

    @property
    def canvas_color(self) -> None:
        pass
    
    @canvas_color.setter
    def canvas_color(self, decision: str) -> None:
        if decision not in CANVAS_BORDER_COLORS.keys():
            decision = "default"
        color = CANVAS_BORDER_COLORS[decision]
        self.canvas.setStyleSheet(f"border: 4px solid {color}; border-radius: 4px;")

    @property
    def button_styles(self) -> dict:
        """
        Get predefined button style configurations.
        
        Returns:
            Dictionary of button style configs
        """
        return {
            "pass": {
                "normal": self._create_button_stylesheet(
                    "#10b981", "white", "#059669", "#047857"
                ),
                "highlighted": self._create_button_stylesheet(
                    "#059669", "white", "#047857", "#047857", "#047857"
                )
            },
            "borderline": {
                "normal": self._create_button_stylesheet(
                    "#f59e0b", "black", "#d97706", "#b45309"
                ),
                "highlighted": self._create_button_stylesheet(
                    "#d97706", "black", "#b45309", "#b45309", "#b45309"
                )
            },
            "reject": {
                "normal": self._create_button_stylesheet(
                    "#f43f5e", "white", "#e11d48", "#be123c"
                ),
                "highlighted": self._create_button_stylesheet(
                    "#e11d48", "white", "#be123c", "#be123c", "#be123c"
                )
            },
            "segment": {
                "normal": self._create_button_stylesheet(
                    "#f59e0b", "black", "#d97706", "#b45309"
                ),
                "highlighted": self._create_button_stylesheet(
                    "#d97706", "black", "#b45309", "#b45309", "#b45309"
                ),
                "finished": self._create_button_stylesheet(
                    "#10b981", "white", "#059669", "#047857", "#047857"
                )
            },
            "nav": {
                "normal": self._create_button_stylesheet(
                    "#64748b", "white", "#475569", "#334155", font_size="10px"
                ),
                "disabled": self._create_button_stylesheet(
                    "#cbd5e1", "white", "#cbd5e1", "#cbd5e1", font_size="10px"
                )
            }
        }

    ############ PLACEHOLDER METHODS ############

    def _save_state(self):
        """
        Save current widget state to database.
        
        To be implemented in subclasses.
        """
        pass
    
    def display_image(self):
        """
        Display the current image in the canvas.

        To be implemented in subclasses.
        """
        pass
    
    ############ PUBLIC METHODS ############
    
    def load_images(self, image_folder: Path) -> bool:
        """
        Load all images from folder.
        
        Args:
            image_folder: Path to image folder
            
        Returns:
            True if images loaded, False otherwise
        """
        self.image_paths = sorted([
            f for f in Path(image_folder).iterdir()
            if f.suffix.lower() in IMAGE_FORMATS
        ])
        
        # Save image metadata if needed
        self.db_manager.save_folder_metadata(image_folder)
        
        return len(self.image_paths) > 0
    
    def next_image(self):
        """Move to next image"""
        if self.index < len(self.image_paths) - 1:
            self.index += 1
            self.index_changed.emit(self.index)
            return True
        return False
    
    def previous_image(self):
        """Move to previous image"""
        if self.index > 0:
            self.index -= 1
            self.index_changed.emit(self.index)
            return True
        return False

    def get_progress_percentage(self) -> int:
        """Get current progress as percentage"""
        if self.image_count == 0:
            return 0
        return int(((self.index + 1) / self.image_count) * 100)
    
    ########### ACTIONS ############
        
    def _on_canvas_clicked(self, x: int, y: int, screen_pos):
        """Handle canvas click"""
        r, g, b = self.canvas.get_pixel_value(x, y)
    
    def _on_image_channel_changed(self, channel_text: str):
        """Handle image channel change"""
        self.canvas.set_image_channel(channel_text.lower())
        
    def _on_overlay_channel_changed(self, channel_text: str):
        """Handle overlay channel change"""
        self.canvas.set_overlay_channel(channel_text.lower())

    ############ PRIVATE METHODS ############
    
    def _create_top_info_layout(self) -> tuple:
        """
        Create top information layout with counter and image name.
        
        Returns:
            Tuple of (layout, image_counter_label, image_name_label)
        """
        info_layout = QHBoxLayout()
        
        # Image counter
        image_counter_label = QLabel("0 / 0")
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        image_counter_label.setFont(font)
        info_layout.addWidget(image_counter_label)
        info_layout.addSpacing(10)
        
        # Image name
        image_name_label = QLabel("No image loaded")
        image_name_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        info_layout.addWidget(image_name_label)
        
        return info_layout, image_counter_label, image_name_label

    def _create_canvas(self) -> Canvas:
        """
        Create and configure the image canvas.
        
        Returns:
            Configured ImageCanvas instance
        """
        self.canvas = Canvas()
        self.canvas.setMinimumHeight(400)
        self.canvas.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.canvas_color = "default" # Set default border color

    def _create_main_panel(self) -> tuple[QWidget, Canvas]:
        """
        Create main content panel.
        
        Returns:
            Tuple of (widget, canvas)
        """
        panel = QWidget()
        
        window_layout = QVBoxLayout()
        window_layout.setContentsMargins(5, 5, 5, 5)
        
        # Canvas
        self._create_canvas()
        window_layout.addWidget(self.canvas)

        panel.setLayout(window_layout)

        return panel, self.canvas
        
    def _create_button_stylesheet(
        self,
        bg_color: str,
        text_color: str,
        hover_color: str,
        pressed_color: str,
        border_color: str = None,
        font_size: str = "11px"
    ) -> str:
        """
        Create a button stylesheet.
        
        Args:
            bg_color: Background color hex
            text_color: Text color
            hover_color: Hover state color
            pressed_color: Pressed state color
            border_color: Optional border color for highlighted state
            font_size: Font size
            
        Returns:
            Stylesheet string
        """
        border_style = f"border: 3px solid {border_color};" if border_color else ""
        
        return f"""
            QPushButton {{
                background-color: {bg_color};
                color: {text_color};
                font-weight: bold;
                padding: 8px;
                border-radius: 5px;
                font-size: {font_size};
                {border_style}
            }}
            QPushButton:hover {{
                background-color: {hover_color};
            }}
            QPushButton:pressed {{
                background-color: {pressed_color};
            }}
        """
    
    