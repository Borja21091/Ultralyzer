import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from PySide6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QPushButton, QGraphicsView, 
    QProgressBar, QComboBox, QSlider, QSplitter, 
    QLabel, QMessageBox, QWidget, QTextEdit, QCheckBox
)
from frontend.widgets.canvas import Canvas
from backend.steps.metrics import MetricsStep
from PySide6.QtCore import Qt, Signal
from backend.models.segmentor import Segmentor
from frontend.widgets.widget_base import BaseWidget
from backend.models.database import DatabaseManager
from backend.steps.segmentation import SegmentationStep
from frontend.widgets.canvas import ImageLayer, OverlayLayer
from definitions import IMAGE_CHANNEL_MAP, OVERLAY_MAP, BLANK_STATE
from PySide6.QtGui import QShortcut, QKeySequence, QCursor, QPixmap, QPainter
from backend.models.segmentor import UWFFoveaSegmentor, UWFDiscSegmentor, UWFAVSegmentor

from backend.utils.threads import SingleMetricsWorker, BatchMetricsWorker
from backend.utils.threads import SingleSegmentationWorker, BatchSegmentationWorker



class SegmentationWidget(BaseWidget):
    """Interactive segmentation, visualization and correction widget"""
    
    status_text = Signal(str)
    decision_made = Signal(str, str) # filename, decision
    
    def __init__(self, 
                 db_manager: DatabaseManager = None, 
                 av_segmentor: Segmentor = UWFAVSegmentor(), 
                 disc_segmentor: Segmentor = UWFDiscSegmentor(), 
                 fovea_segmentor: Segmentor = UWFFoveaSegmentor()):
        super().__init__(db_manager)
        self.step_seg = SegmentationStep(av_segmentor=av_segmentor, 
                                         disc_segmentor=disc_segmentor, 
                                         fovea_segmentor=fovea_segmentor,
                                         db_manager=self.db_manager)
        self.step_metrics = MetricsStep(db_manager=self.db_manager, 
                                        micron_mex=None)
        
        # Track segmentation worker thread to prevent garbage collection
        self._worker_thread = None
        
        # Overlay opacity
        self._overlay_opacity = 0.75
        
        # QC state
        self.state = BLANK_STATE.copy()
        
        # Edit mode state
        self._edit_mode = False
        self._overlay_layer: OverlayLayer = None
        self._active_tool = None
        self._brush_size = 5
        self._stroke_started = False
        self._has_unsaved_changes = False
        
        # Track which keys are currently pressed
        self._keys_pressed = set()
        
        self._init_ui()
    
    ############ PROPERTIES ############
    
    @property
    def state(self) -> dict:
        """Get current widget state"""
        return self._state

    @state.setter
    def state(self, value: dict):
        """Set current widget state"""
        self._state = value
    
    @property
    def edit_mode(self) -> bool:
        """Get edit mode state"""
        return self._edit_mode
    
    @edit_mode.setter
    def edit_mode(self, value: bool):
        """Set edit mode state"""
        self._edit_mode = value
    
    @property
    def active_tool(self):
        """Get active editing tool"""
        return self._active_tool
    
    @active_tool.setter
    def active_tool(self, tool: str):
        """Set active editing tool"""
        if self._active_tool == tool:
            # Deselect tool
            self._active_tool = None
            self.btn_brush.setChecked(False)
            self.btn_smart_paint.setChecked(False)
            self.btn_eraser.setChecked(False)
            self.btn_change.setChecked(False)
            self.btn_fovea_location.setChecked(False)
            self.canvas.set_tool(None)
            self.canvas.setCursor(Qt.CursorShape.ArrowCursor)
        else:
            # Select tool
            self._active_tool = tool
            self.btn_brush.setChecked(tool == "brush")
            self.btn_smart_paint.setChecked(tool == "smart_paint")
            self.btn_eraser.setChecked(tool == "eraser")
            self.btn_change.setChecked(tool == "change")
            self.btn_fovea_location.setChecked(tool == "fovea_location")
            self.canvas.set_tool(tool)
            if tool in ["brush", "smart_paint", "eraser"]:
                cursor = self._create_brush_cursor(self.brush_size // 2)
                self.canvas.setCursor(cursor)
            else:
                self.canvas.setCursor(Qt.CursorShape.CrossCursor)

    @property
    def channel(self) -> str:
        """Get current display channel"""
        return self.channel_combo.currentText().lower()
    
    @channel.setter
    def channel(self, value: str):
        """Set current display channel"""
        if value.lower() in IMAGE_CHANNEL_MAP.keys():
            index = list(IMAGE_CHANNEL_MAP.keys()).index(value.lower())
            self.channel_combo.setCurrentIndex(index)
    
    @property
    def overlay(self) -> str:
        """Get current segmentation overlay option"""
        return self.overlay_combo.currentText().lower()
    
    @overlay.setter
    def overlay(self, value: str):
        """Set current segmentation overlay option"""
        if value.lower() in OVERLAY_MAP.keys():
            index = list(OVERLAY_MAP.keys()).index(value.lower())
            self.overlay_combo.setCurrentIndex(index)

    @property
    def segmentation_mask_path(self) -> Path:
        """Get segmentation mask path for current image"""
        name = str(self.image_path.stem)
        return self.db_manager.get_segmentation_mask_path(name)

    @property
    def brush_size(self) -> int:
        """Get brush size"""
        return self._brush_size
    
    @brush_size.setter
    def brush_size(self, size: int):
        """Set brush size"""
        self._brush_size = min(max(1, size), 500)

    @property
    def has_unsaved_changes(self) -> bool:
        """Get unsaved changes state"""
        return self._overlay_layer._is_dirty

    ############ UI ############
    
    def _init_ui(self):
        """Initialize User Interface"""
        layout = QVBoxLayout(self)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        
        # Top section: Image info
        info_layout = QHBoxLayout()
        
        # Use base widget's top layout
        top_info, self.image_counter_label, self.image_name_label = self._create_top_info_layout()
        info_layout.addLayout(top_info)
        info_layout.addSpacing(10)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumHeight(12)
        info_layout.addWidget(self.progress_bar)
        
        info_layout.addSpacing(10)
        layout.addLayout(info_layout)
        
        # Main content area: Splitter with canvas (left) and edit toolbar (right)
        self.main_splitter = QSplitter(Qt.Horizontal)
        
        # Left side: Canvas container
        self.canvas_container = QWidget()
        canvas_layout = QVBoxLayout(self.canvas_container)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        
        # Initialize Canvas immediately
        self.canvas = Canvas()
        self.canvas.signal_zoom_changed.connect(self._update_brush_cursor)
        self.canvas.signal_fovea_selected.connect(self._on_fovea_location_selected)
        self.canvas.signal_opacity_changed.connect(lambda value: self._on_opacity_changed(self._overlay_opacity + value))
        self.canvas.signal_opacity_changed.connect(lambda value: self.opacity_slider.setValue(int(min(max(self._overlay_opacity + value, 0.0), 1.0) * 100)))
        self.canvas.signal_brush_radius_changed.connect(lambda delta: self._on_brush_size_changed(max(self.brush_size + delta, 1)))
        canvas_layout.addWidget(self.canvas)
        self.main_splitter.addWidget(self.canvas_container)
        
        # Right side: Edit toolbar (hidden by default)
        self.edit_toolbar_widget = QWidget()
        edit_toolbar_layout = self._create_edit_toolbar()
        self.edit_toolbar_widget.setLayout(edit_toolbar_layout)
        self.edit_toolbar_widget.setVisible(False)
        self.edit_toolbar_widget.setMaximumWidth(120)
        self.edit_toolbar_widget.setMinimumWidth(110)
        self.main_splitter.addWidget(self.edit_toolbar_widget)
        
        # Set splitter proportions (canvas takes most space)
        self.main_splitter.setSizes([1000, 0])
        self.main_splitter.setCollapsible(1, True)
        
        layout.addWidget(self.main_splitter, 1)
        
        # Bottom section: Controls
        bottom_layout = QHBoxLayout()
        
        # Create all button sections (now includes display controls)
        buttons_layout = self._create_bottom_section()
        bottom_layout.addLayout(buttons_layout)
        
        layout.addLayout(bottom_layout)
        
        # Shortcuts
        self._setup_shortcuts()

    def _create_bottom_section(self) -> QHBoxLayout:
        """Create segmentation button layout with three sections: Display | Current Image | Segment All + Nav"""
        buttons_layout = QHBoxLayout()
        button_styles = self.button_styles

        # LEFT SECTION: Display Controls (vertical channel/overlay + vertical opacity slider)
        display_layout = QHBoxLayout()
        
        # Combobox section (vertical stacking)
        combo_layout = QVBoxLayout()
        
        # Refresh display button
        self.btn_refresh_display = QPushButton("🔄 Refresh Display")
        self.btn_refresh_display.setMinimumHeight(30)
        self.btn_refresh_display.clicked.connect(self._on_refresh_display)
        combo_layout.addWidget(self.btn_refresh_display)
        
        # Channel selector with label
        channel_label = QLabel("Channel:")
        channel_label.setMaximumWidth(70)
        self.channel_combo = QComboBox()
        self.channel_combo.setMinimumWidth(100)
        self.channel_combo.setMaximumWidth(200)
        self.channel_combo.setToolTip("Display Channel")
        self.channel_combo.addItems(["Color", "Red", "Green", "Blue"])
        self.channel_combo.setCurrentText("Color")
        self.channel_combo.currentTextChanged.connect(self._on_image_channel_changed)
        combo_layout.addWidget(channel_label)
        combo_layout.addWidget(self.channel_combo)
        
        # Segmentation overlay selector with label
        overlay_label = QLabel("Overlay:")
        overlay_label.setMaximumWidth(70)
        self.overlay_combo = QComboBox()
        self.overlay_combo.setMinimumWidth(100)
        self.overlay_combo.setMaximumWidth(200)
        self.overlay_combo.setToolTip("Segmentation Overlay")
        self.overlay_combo.addItems(["Red", "Green", "Blue", "Vessels", "All", "None"])
        self.overlay_combo.setCurrentText("All")
        self.overlay_combo.currentTextChanged.connect(self._on_overlay_channel_changed)
        combo_layout.addWidget(overlay_label)
        combo_layout.addWidget(self.overlay_combo)
        
        # Add Fovea Checkbox
        self.chk_fovea = QCheckBox("Show Fovea")
        self.chk_fovea.setChecked(False)
        self.chk_fovea.toggled.connect(self._on_toggle_fovea)
        combo_layout.addWidget(self.chk_fovea)
        
        display_layout.addLayout(combo_layout)
        display_layout.addSpacing(15)
        
        # Opacity slider section (vertical)
        opacity_layout = QVBoxLayout()
        opacity_label = QLabel("Opacity:")
        opacity_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        opacity_layout.addWidget(opacity_label)
        
        self.opacity_slider = QSlider(Qt.Vertical)
        self.opacity_slider.setMinimum(0)
        self.opacity_slider.setMaximum(100)
        self.opacity_slider.setValue(75)
        self.opacity_slider.setMaximumHeight(100)
        self.opacity_slider.setToolTip("Adjust overlay opacity")
        self.opacity_slider.valueChanged.connect(lambda value: self._on_opacity_changed(value / 100.0))
        opacity_layout.addWidget(self.opacity_slider, alignment=Qt.AlignmentFlag.AlignHCenter)
        
        self.opacity_label = QLabel("75%")
        self.opacity_label.setMaximumWidth(35)
        self.opacity_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        opacity_layout.addWidget(self.opacity_label)
        
        display_layout.addLayout(opacity_layout)
        
        # MIDDLE_LEFT SECTION: Image Quality Controls (vertical decision + vertical notes)
        quality_layout = QHBoxLayout()
        quality_layout.setContentsMargins(0, 0, 0, 0)
        quality_layout.setSpacing(5)
        
        # Decision buttons (vertical stacking)
        decision_buttons_layout = QVBoxLayout()
        decision_buttons_layout.setContentsMargins(0, 0, 0, 0)
        decision_buttons_layout.setSpacing(5)
        
        # Pass button
        self.btn_pass = QPushButton("✅ PASS")
        self.btn_pass.setMaximumWidth(200)
        self.btn_pass.setMinimumHeight(40)
        self.btn_pass.setStyleSheet(button_styles["pass"]["normal"])
        self.btn_pass.clicked.connect(lambda: self._on_qc_decision("pass"))
        decision_buttons_layout.addWidget(self.btn_pass)
        
        # Borderline button
        self.btn_borderline = QPushButton("⚠️ BORDERLINE")
        self.btn_borderline.setMaximumWidth(200)
        self.btn_borderline.setMinimumHeight(40)
        self.btn_borderline.setStyleSheet(button_styles["borderline"]["normal"])
        self.btn_borderline.clicked.connect(lambda: self._on_qc_decision("borderline"))
        decision_buttons_layout.addWidget(self.btn_borderline)
        
        # Reject button
        self.btn_reject = QPushButton("❌ REJECT")
        self.btn_reject.setMaximumWidth(200)
        self.btn_reject.setMinimumHeight(40)
        self.btn_reject.setStyleSheet(button_styles["reject"]["normal"])
        self.btn_reject.clicked.connect(lambda: self._on_qc_decision("reject"))
        decision_buttons_layout.addWidget(self.btn_reject)
        
        quality_layout.addLayout(decision_buttons_layout)
        
        # Notes section (vertical)
        notes_layout = QVBoxLayout()
        notes_layout.setContentsMargins(0, 0, 0, 0)
        notes_layout.setSpacing(5)
        
        notes_layout.addWidget(QLabel("Notes (optional):"))
        self.notes_text = QTextEdit()
        self.notes_text.setPlaceholderText("Add any observations about the image...")
        notes_layout.addWidget(self.notes_text)
        quality_layout.addLayout(notes_layout, 9)
        
        # MIDDLE_RIGHT SECTION: Current Image Actions (vertical stacking)
        current_image_layout = QVBoxLayout()
        current_image_layout.setContentsMargins(0, 0, 0, 0)
        current_image_layout.setSpacing(5)
        
        self.btn_segment_current = QPushButton("⏩ Segment Current")
        self.btn_segment_current.setMinimumHeight(40)
        self.btn_segment_current.setStyleSheet(button_styles["segment"]["normal"])
        self.btn_segment_current.clicked.connect(self._on_segment_current_image)
        current_image_layout.addWidget(self.btn_segment_current)
        
        self.btn_metrics_current = QPushButton("📊 Metrics")
        self.btn_metrics_current.setMinimumHeight(40)
        self.btn_metrics_current.setStyleSheet(button_styles["segment"]["normal"])
        self.btn_metrics_current.clicked.connect(self._on_metrics_current_image)
        current_image_layout.addWidget(self.btn_metrics_current)
        
        self.btn_edit = QPushButton("✏️ Edit Mask")
        self.btn_edit.setMinimumHeight(40)
        self.btn_edit.setStyleSheet(button_styles["segment"]["normal"])
        self.btn_edit.clicked.connect(self._on_edit_mode_toggle)
        current_image_layout.addWidget(self.btn_edit)
        
        # RIGHT SECTION: Segment All (top) + Metrics All (middle) + Navigation (bottom)
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(5)
        
        self.btn_segment_all = QPushButton("⏩ Segment All")
        self.btn_segment_all.setMinimumHeight(40)
        self.btn_segment_all.setStyleSheet(button_styles["segment"]["normal"])
        self.btn_segment_all.clicked.connect(self._on_segment_all)
        right_layout.addWidget(self.btn_segment_all)
        
        self.btn_metrics_all = QPushButton("📊 Metrics All")
        self.btn_metrics_all.setMinimumHeight(40)
        self.btn_metrics_all.setStyleSheet(button_styles["segment"]["normal"])
        self.btn_metrics_all.clicked.connect(self._on_calculate_metrics_all)
        right_layout.addWidget(self.btn_metrics_all)
        
        # Navigation buttons (horizontal - same width, dynamic sizing)
        nav_layout = QHBoxLayout()
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.setSpacing(5)
        
        self.btn_prev = QPushButton("◀ Previous")
        self.btn_prev.setMinimumHeight(40)
        self.btn_prev.setStyleSheet(button_styles["nav"]["normal"])
        self.btn_prev.clicked.connect(self._on_prev)
        nav_layout.addWidget(self.btn_prev, 1)
        
        self.btn_next = QPushButton("Next ▶")
        self.btn_next.setMinimumHeight(40)
        self.btn_next.setStyleSheet(button_styles["nav"]["normal"])
        self.btn_next.clicked.connect(self._on_next)
        nav_layout.addWidget(self.btn_next, 1)
        
        right_layout.addLayout(nav_layout)
        
        
        # Wrap sections into containers to control spacing
        left_container = QWidget()
        left_container.setLayout(display_layout)
        middle_left_container = QWidget()
        middle_left_container.setLayout(quality_layout)
        middle_right_container = QWidget()
        middle_right_container.setLayout(current_image_layout)
        middle_right_container.setMaximumWidth(250)
        right_container = QWidget()
        right_container.setLayout(right_layout)
        right_container.setMaximumWidth(250)
        
        # Assemble final layout
        buttons_layout.addWidget(left_container, 0)
        buttons_layout.addStretch()
        buttons_layout.addWidget(middle_left_container, 5)
        buttons_layout.addSpacing(20)
        buttons_layout.addWidget(middle_right_container, 1)
        buttons_layout.addSpacing(20)
        buttons_layout.addWidget(right_container, 1)

        return buttons_layout
    
    def _create_edit_toolbar(self) -> QVBoxLayout:
        """Create edit toolbar with tools and controls"""
        toolbar_layout = QVBoxLayout()
        toolbar_layout.setContentsMargins(8, 8, 8, 8)
        toolbar_layout.setSpacing(8)
        
        # Brush tool
        self.btn_brush = QPushButton("🖌️")
        self.btn_brush.setToolTip("Brush - Add to mask (Ctrl/Cmd+B)")
        self.btn_brush.setMaximumWidth(80)
        self.btn_brush.setCheckable(True)
        self.btn_brush.clicked.connect(lambda: self._set_active_tool("brush"))
        toolbar_layout.addWidget(self.btn_brush)
        
        # Smart Paint tool
        self.btn_smart_paint = QPushButton("✨")
        self.btn_smart_paint.setToolTip("Smart Paint - Paint over existing vessels (Ctrl/Cmd+Shift+B)")
        self.btn_smart_paint.setMaximumWidth(80)
        self.btn_smart_paint.setCheckable(True)
        self.btn_smart_paint.clicked.connect(lambda: self._set_active_tool("smart_paint"))
        toolbar_layout.addWidget(self.btn_smart_paint)
        
        # Eraser tool
        self.btn_eraser = QPushButton("🧹")
        self.btn_eraser.setToolTip("Eraser - Remove from mask (Ctrl/Cmd+E)")
        self.btn_eraser.setMaximumWidth(80)
        self.btn_eraser.setCheckable(True)
        self.btn_eraser.clicked.connect(lambda: self._set_active_tool("eraser"))
        toolbar_layout.addWidget(self.btn_eraser)
        
        # Change color tool
        self.btn_change = QPushButton("⇄")
        self.btn_change.setToolTip("Change Color - Switch artery/vein (Ctrl/Cmd+C)")
        self.btn_change.setMaximumWidth(80)
        self.btn_change.setCheckable(True)
        self.btn_change.clicked.connect(lambda: self._set_active_tool("change"))
        toolbar_layout.addWidget(self.btn_change)
        
        # Fovea location button
        self.btn_fovea_location = QPushButton("🎯")
        self.btn_fovea_location.setToolTip("Edit Fovea Location")
        self.btn_fovea_location.setMaximumWidth(80)
        self.btn_fovea_location.setCheckable(True)
        self.btn_fovea_location.clicked.connect(lambda: self._set_active_tool("fovea_location"))
        toolbar_layout.addWidget(self.btn_fovea_location)
        
        toolbar_layout.addSpacing(10)
        
        # Brush size slider
        toolbar_layout.addWidget(QLabel("Size:"))
        self.brush_size_slider = QSlider(Qt.Vertical)
        self.brush_size_slider.setMinimum(1)
        self.brush_size_slider.setMaximum(50)
        self.brush_size_slider.setValue(5)
        self.brush_size_slider.setMaximumWidth(100)
        self.MAX_BRUSH_SIZE = 500
        self.MIN_BRUSH_SIZE = 1
        self.brush_size_slider.valueChanged.connect(self._on_brush_size_changed)
        toolbar_layout.addWidget(self.brush_size_slider, alignment=Qt.AlignmentFlag.AlignHCenter)
        
        self.brush_size_label = QLabel("5px")
        self.brush_size_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        toolbar_layout.addWidget(self.brush_size_label)
        
        toolbar_layout.addSpacing(4)
        
        # Undo button
        self.btn_undo = QPushButton("↶")
        self.btn_undo.setToolTip("Undo (Ctrl/Cmd + Z)")
        self.btn_undo.setMaximumWidth(70)
        self.btn_undo.clicked.connect(self._on_undo)
        toolbar_layout.addWidget(self.btn_undo)
        
        # Redo button
        self.btn_redo = QPushButton("↷")
        self.btn_redo.setToolTip("Redo (Ctrl/Cmd + Shift + Z)")
        self.btn_redo.setMaximumWidth(70)
        self.btn_redo.clicked.connect(self._on_redo)
        toolbar_layout.addWidget(self.btn_redo)
        
        # Reset button
        self.btn_reset = QPushButton("🔄")
        self.btn_reset.setToolTip("Reset all edits")
        self.btn_reset.setMaximumWidth(70)
        self.btn_reset.clicked.connect(self._on_reset_edits)
        toolbar_layout.addWidget(self.btn_reset)
        
        toolbar_layout.addSpacing(10)
        
        # Exit edit mode button
        self.btn_exit_edit = QPushButton("✕")
        self.btn_exit_edit.setToolTip("Exit Edit Mode")
        self.btn_exit_edit.setMaximumWidth(90)
        self.btn_exit_edit.clicked.connect(self._on_edit_mode_toggle)
        toolbar_layout.addWidget(self.btn_exit_edit)
        
        toolbar_layout.addStretch()
        
        return toolbar_layout

    def _setup_shortcuts(self):
        """Create keyboard shortcuts"""
        sc_next = QShortcut(QKeySequence(Qt.Key_Right), self)
        sc_next.activated.connect(self._on_next)
        sc_prev = QShortcut(QKeySequence(Qt.Key_Left), self)
        sc_prev.activated.connect(self._on_prev)
        
        # Channel shortcuts
        sc_c = QShortcut(QKeySequence("C"), self)
        sc_c.activated.connect(lambda: self._set_channel_by_key("color"))
        sc_g = QShortcut(QKeySequence("G"), self)
        sc_g.activated.connect(lambda: self._set_channel_by_key("green"))
        sc_r = QShortcut(QKeySequence("R"), self)
        sc_r.activated.connect(lambda: self._set_channel_by_key("red"))
        sc_b = QShortcut(QKeySequence("B"), self)
        sc_b.activated.connect(lambda: self._set_channel_by_key("blue"))
        
        # Overlay shortcuts
        sc_a = QShortcut(QKeySequence("1"), self)
        sc_a.activated.connect(lambda: self._set_overlay_by_key("red"))
        sc_v = QShortcut(QKeySequence("2"), self)
        sc_v.activated.connect(lambda: self._set_overlay_by_key("green"))
        sc_both = QShortcut(QKeySequence("3"), self)
        sc_both.activated.connect(lambda: self._set_overlay_by_key("blue"))
        sc_none = QShortcut(QKeySequence("4"), self)
        sc_none.activated.connect(lambda: self._set_overlay_by_key("vessels"))
        sc_all = QShortcut(QKeySequence("5"), self)
        sc_all.activated.connect(lambda: self._set_overlay_by_key("all"))
        sc_no_overlay = QShortcut(QKeySequence("6"), self)
        sc_no_overlay.activated.connect(lambda: self._set_overlay_by_key("none"))
        
        # Edit mode shortcuts
        sc_save = QShortcut(QKeySequence.StandardKey.Save, self)
        sc_save.activated.connect(self._on_save)
        sc_undo = QShortcut(QKeySequence.StandardKey.Undo, self)
        sc_undo.activated.connect(self._on_undo)
        sc_redo = QShortcut(QKeySequence.StandardKey.Redo, self)
        sc_redo.activated.connect(self._on_redo)
        sc_brush = QShortcut(QKeySequence("Ctrl+B"), self)
        sc_brush.activated.connect(lambda: self._set_active_tool("brush"))
        sc_smart_paint = QShortcut(QKeySequence("Ctrl+Shift+B"), self)
        sc_smart_paint.activated.connect(lambda: self._set_active_tool("smart_paint"))
        sc_eraser = QShortcut(QKeySequence("Ctrl+E"), self)
        sc_eraser.activated.connect(lambda: self._set_active_tool("eraser"))
        sc_change = QShortcut(QKeySequence("Ctrl+C"), self)
        sc_change.activated.connect(lambda: self._set_active_tool("change"))
        
        # Brush size shortcuts
        sc_plus = QShortcut(QKeySequence(Qt.Key_Plus), self)
        sc_plus.activated.connect(self._on_increase_brush_size)
        sc_equals = QShortcut(QKeySequence(Qt.Key_Equal), self)
        sc_equals.activated.connect(self._on_increase_brush_size)
        sc_minus = QShortcut(QKeySequence(Qt.Key_Minus), self)
        sc_minus.activated.connect(self._on_decrease_brush_size)

    ############ WRAPPER METHODS ############
    
    def display_image(self):
        """Wrapper method to display new image"""
        self._display_new_image()
    
    def _set_channel_by_key(self, channel: str):
        """Wrapper for channel property setter"""
        self.channel = channel

    def _set_overlay_by_key(self, overlay: str):
        """Wrapper for overlay property setter"""
        self.overlay = overlay

    def _set_active_tool(self, tool: str):
        """Wrapper for active tool property setter"""
        self.active_tool = tool
        if tool in ["brush", "smart_paint", "eraser"]:
            self._update_brush_cursor()
        elif tool in ["change", "fovea_location"]:
            self.canvas.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.canvas.setCursor(Qt.CursorShape.ArrowCursor)

    ############ PRIVATE METHODS ############
    
    def _display_new_image(self):
        """Display new image with selected channel and segmentation overlay"""
        if not self.image_paths:
            return
        
        # Get image and segmentation paths
        image_path = self.image_path
        name = image_path.stem
        
        # Load notes if any
        qc_result = self.db_manager.get_qc_result(name)
        if qc_result:
            self._load_state(qc_result)
            self.notes_text.setPlainText(qc_result.notes)
            self._highlight_decision(qc_result.decision.value)
        
        # Load image
        image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        image_layer = ImageLayer(image)
        
        # Load or create overlay - default to empty
        overlay_array = np.zeros_like(image)
        seg_result = self.db_manager.get_segmentation_by_filename(name)
        if seg_result:
            mask_path = Path(seg_result.seg_folder) / Path(str(seg_result.name) + str(seg_result.extension))
            if mask_path.is_file():
                overlay_array = np.array(Image.open(mask_path))
        
        overlay_layer = OverlayLayer(overlay_array)
        overlay_layer.opacity = int(min(max(self._overlay_opacity * 100, 0), 100))
        self._overlay_layer = overlay_layer
        
        # Create canvas only once, or update existing canvas
        if self.canvas is None:
            self.canvas = Canvas(image_layer, overlay_layer)
            layout = self.canvas_container.layout()
            layout.takeAt(0) # Remove placeholder
            layout.addWidget(self.canvas)
        else:
            # Update canvas contents instead of recreating
            self.canvas.reset_layers(image_layer, overlay_layer)
        self.canvas.signal_zoom_changed.connect(self._update_brush_cursor)
        self.canvas.signal_fovea_selected.connect(self._on_fovea_location_selected)
        
        # Update fovea marker
        fovea_coords = self.db_manager.get_fovea_by_filename(name)
        if fovea_coords and fovea_coords[0] is not None:
            self.canvas.update_fovea(fovea_coords[0], fovea_coords[1])
            self.canvas.set_fovea_visibility(self.chk_fovea.isChecked())
        else:
            self.canvas.set_fovea_visibility(False)
        
        # Set canvas edge color
        try:
            qc_decision = self.db_manager.get_qc_result(name).decision.value
            self.canvas_color = qc_decision
        except Exception:
            self.canvas_color = "default"
        
        # Center the image on the canvas
        self.canvas.centerOn(self.canvas.scene().itemsBoundingRect().center()) # Move to center
        # self.canvas.fitInView(self.canvas.scene.itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio) # Move to center + scale to fit
        
        # Set channel and overlay
        self.canvas.set_image_channel(self.channel)
        self.canvas.set_overlay_channel(self.overlay)
        
        # Restore active tool if in edit mode
        if self.edit_mode and self.active_tool:
            self.canvas.set_tool(self.active_tool)
            self._update_brush_cursor()

    def _highlight_decision(self, decision: str):
        """Highlight decision button"""
        self._clear_decision_highlight()
        button_styles = self.button_styles
        
        if decision == "pass":
            self.btn_pass.setStyleSheet(button_styles["pass"]["highlighted"])
        elif decision == "borderline":
            self.btn_borderline.setStyleSheet(button_styles["borderline"]["highlighted"])
        elif decision == "reject":
            self.btn_reject.setStyleSheet(button_styles["reject"]["highlighted"])
    
    def _clear_decision_highlight(self):
        """Clear all decision highlights"""
        button_styles = self.button_styles
        self.btn_pass.setStyleSheet(button_styles["pass"]["normal"])
        self.btn_borderline.setStyleSheet(button_styles["borderline"]["normal"])
        self.btn_reject.setStyleSheet(button_styles["reject"]["normal"])
    
    def _prompt_save_changes(self):
        """Prompt user to save unsaved edits"""
        reply = QMessageBox.question(
            self,
            "Unsaved Changes",
            "You have unsaved edits. Do you want to save them?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self._save_state()
            self._save_edits()
        elif reply == QMessageBox.StandardButton.No:
            return False
        
        return True
    
    def _create_brush_cursor(self, radius: float) -> QCursor:
        """Create a circular brush cursor with transparent fill"""
        # Create pixmap for cursor
        size = int(2 * radius)
        cursor_pixmap = QPixmap(size, size)
        cursor_pixmap.fill(Qt.GlobalColor.transparent)
        
        # Draw white circle outline
        painter = QPainter(cursor_pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.GlobalColor.white)
        painter.drawEllipse(0, 0, size - 1, size - 1)
        painter.end()
        
        # Create cursor from pixmap
        cursor = QCursor(cursor_pixmap)
        return cursor
    
    def _reset_buttons_state(self):
        """Reset button styles to default"""
        # QC decision buttons
        self._clear_decision_highlight()
        self.notes_text.clear()
        
        # Segmentation buttons
        self.btn_segment_all.setStyleSheet(self.button_styles["segment"]["normal"])
        self.btn_segment_current.setStyleSheet(self.button_styles["segment"]["normal"])
        
        # Metrics buttons
        self.btn_metrics_all.setStyleSheet(self.button_styles["segment"]["normal"])
        self.btn_metrics_current.setStyleSheet(self.button_styles["segment"]["normal"])
    
    def _update_brush_cursor(self):
        """Update brush cursor based on current zoom and brush size"""
        if self._active_tool in ["brush", "smart_paint", "eraser"] and self.canvas:
            # Scale brush radius by canvas zoom level
            scaled_radius = (self.brush_size / 2) * self.canvas._zoom_level
            cursor = self._create_brush_cursor(scaled_radius)
            self.canvas.setCursor(cursor)
    
    def _load_state(self, qc_result):
        """Load QC data into state"""
        if not self.image_paths:
            return
        
        self.state['filename'] = qc_result.name
        self.state['decision'] = qc_result.decision.value
        self.state['notes'] = qc_result.notes
    
    def _save_state(self):
        """Save current image's QC data"""
        if not self.image_paths:
            return

        self.state['notes'] = self.notes_text.toPlainText()

        try:
            state = self.state
            self.db_manager.save_qc_result(
                state['filename'],
                state['decision'],
                state['notes']
                )
        except Exception as e:
            print(f"Error saving QC result: {e}")
            return False
        return True
        
    def _save_edits(self):
        """Save current overlay edits"""
        
        # Get the RGB array from overlay
        overlay_array = self.canvas.overlay_layer.get_array()
        
        # Get id of current image
        name = str(self.image_path.stem)
        qc_result = self.db_manager.get_qc_result(name)
        id = qc_result.id
        
        # Get segmentation results
        seg_result = self.db_manager.get_segmentation_result_by_id(id)
        if seg_result is None:
            self.status_text.emit("No segmentation result found in database")
            return
        
        extension = str(seg_result.extension)
        seg_path = Path(seg_result.seg_folder)
        
        # Save to file
        seg_path.mkdir(parents=True, exist_ok=True)
        
        mask_path = seg_path / Path(name + extension)
        Image.fromarray(overlay_array).save(mask_path)
        
        # Mark overlay as saved
        self._overlay_layer.mark_saved()
        
        self.status_text.emit(f"Edits saved to {mask_path}")
        
    ############ ACTIONS ############
    
    def _on_qc_decision(self, decision: str):
        """Handle decision"""
        if not self.image_paths:
            return
        
        notes = self.notes_text.toPlainText()
        
        self._highlight_decision(decision)
        self.canvas_color = decision

        # Update state
        self.state['filename'] = self.image_path.stem
        self.state['decision'] = decision
        self.state['notes'] = notes

        # Save to database & emit signal
        if self._save_state():
            self.decision_made.emit(self.state['filename'], decision)
    
    def _on_next(self):
        """Move to next image"""
        if self.has_unsaved_changes:
            self._prompt_save_changes()
        
        if self.next_image():
            self._save_state()
            self._reset_buttons_state()
            self.display_image()
    
    def _on_prev(self):
        """Move to previous image"""
        if self.has_unsaved_changes:
            self._prompt_save_changes()
        
        if self.previous_image():
            self._save_state()
            self._reset_buttons_state()
            self.display_image()
    
    def _on_refresh_display(self):
        """Refresh image display"""
        self._display_new_image()
    
    def _on_segment_all(self):
        """Start segmentation"""
        pending = self.step_seg.get_pending_images()
        
        if not pending:
            self.status_text.emit(("No images to segment"))
            return
        
        self.status_text.emit(f"Segmenting {len(pending)} images...")
        self.btn_segment_all.setEnabled(False)
        self.btn_segment_all.setStyleSheet(self.button_styles["segment"]["highlighted"])
        
        try:
            self.worker_thread = BatchSegmentationWorker(self.step_seg, pending)
            self.worker_thread.progress.connect(self._on_progress)
            self.worker_thread.finished.connect(self._on_batch_segment_finished)
            self.worker_thread.finished.connect(lambda x: self._on_worker_finished(self.worker_thread))
            self.worker_thread.start()
        except Exception as e:
            self.status_text.emit(f"Error: {str(e)}")
            self.btn_segment_all.setStyleSheet(self.button_styles["segment"]["normal"])
        finally:
            self.btn_segment_all.setEnabled(True)
    
    def _on_segment_current_image(self):
        """Segment only the currently displayed image"""
        if not self.image_path:
            self.status_text.emit("No image loaded")
            return
        
        self.status_text.emit(f"Segmenting {self.image_path.name}...")
        self.btn_segment_current.setEnabled(False)
        self.btn_segment_current.setStyleSheet(self.button_styles["segment"]["highlighted"])
        
        try:
            metadata = self.db_manager.get_metadata_by_filename(self.image_path.stem)
            self.worker_thread = SingleSegmentationWorker(self.step_seg, self.image_path, metadata.id)
            self.worker_thread.finished.connect(self._on_single_segment_finished)
            self.worker_thread.finished.connect(lambda x: self._on_worker_finished(self.worker_thread))
            self.worker_thread.start()
            
            self.btn_segment_current.setStyleSheet(self.button_styles["segment"]["highlighted"])
        
        except Exception as e:
            self.status_text.emit(f"Error: {str(e)}")
            self.btn_segment_current.setStyleSheet(self.button_styles["segment"]["normal"])
        
        finally:
            self.btn_segment_current.setEnabled(True)
            # Reload image to show updated segmentation
            self._display_new_image()
    
    def _on_batch_segment_finished(self, success: bool):
        """Handle completion"""
        self.btn_segment_all.setStyleSheet(self.button_styles["segment"]["finished"] if success else self.button_styles["segment"]["normal"])
        self.btn_segment_all.setEnabled(True)
        self.status_text.emit("Complete!" if success else "Failed!")
        self.progress_bar.setValue(100 if success else 0)
        
    def _on_single_segment_finished(self, success: bool):
        """Handle single image segmentation completion"""
        if success:
            self.btn_segment_current.setStyleSheet(self.button_styles["segment"]["finished"])
            self.status_text.emit(f"{self.image_path.name}: ✓")
        else:
            self.btn_segment_current.setStyleSheet(self.button_styles["segment"]["normal"])
            self.status_text.emit(f"{self.image_path.name}: ✗")
    
    def _on_edit_mode_toggle(self):
        """Toggle edit mode on/off"""
        if self.edit_mode:
            # Exit edit mode
            self.edit_mode = False
            self.canvas.set_edit_mode(False)
            self.edit_toolbar_widget.setVisible(False)
            self.main_splitter.setSizes([1000, 0])
            self.btn_edit.setStyleSheet(self.button_styles["segment"]["normal"])
            self.btn_next.setEnabled(True)
            self.btn_prev.setEnabled(True)
            self.canvas.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.canvas.set_tool(None)
            self._active_tool = None
        else:
            # Enter edit mode
            if not self._overlay_layer:
                self.status_text.emit("No overlay to edit")
                return
            
            self.edit_mode = True
            self.canvas.set_edit_mode(True)
            self.edit_toolbar_widget.setVisible(True)
            self.main_splitter.setSizes([800, 110])
            self.btn_edit.setStyleSheet(self.button_styles["segment"]["highlighted"])
            self.btn_next.setEnabled(False)
            self.btn_prev.setEnabled(False)
            self.canvas.setDragMode(QGraphicsView.DragMode.NoDrag)
    
    def _on_brush_size_changed(self, size: int):
        """Handle brush size slider change"""
        self._brush_size = size
        self.brush_size_label.setText(f"{size}px")
        self.canvas.set_brush_radius(size / 2)
        
        # Update cursor if a tool is active
        if self._active_tool:
            self._update_brush_cursor()
    
    def _on_increase_brush_size(self):
        """Increase brush size"""
        new_size = min(self.MAX_BRUSH_SIZE, self._brush_size + 1)
        self.brush_size_slider.setValue(new_size)
        self._on_brush_size_changed(new_size)
    
    def _on_decrease_brush_size(self):
        """Decrease brush size"""
        new_size = max(self.MIN_BRUSH_SIZE, self._brush_size - 1)
        self.brush_size_slider.setValue(new_size)
        self._on_brush_size_changed(new_size)
    
    def _on_undo(self):
        """Undo last operation"""
        if self._overlay_layer:
            self.canvas.undo()
    
    def _on_redo(self):
        """Redo last undone operation"""
        if self._overlay_layer:
            self.canvas.redo()

    def _on_reset_edits(self):
        """Reset all edits"""
        reply = QMessageBox.question(
            self,
            "Reset Edits",
            "Are you sure you want to discard all edits?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes and self._overlay_layer:
            self._overlay_layer.reset()
            self.canvas.update_overlay_display()

    def _on_save(self):
        """Handle save action"""
        self._save_edits()
        self._save_state()

    def _on_opacity_changed(self, value: float):
        """Handle opacity slider changes"""
        # Prepare value
        value = min(max(value, 0), 1.0)
        
        # Store opacity for future images
        self._overlay_opacity = value
        
        # Update label with percentage
        self.opacity_label.setText(f"{int(value * 100)}%")
        
        # Convert to 0.0-1.0 range and update canvas
        opacity_normalized = value
        if self.canvas is not None:
            self.canvas.set_overlay_opacity(opacity_normalized)

    @staticmethod
    def _on_worker_finished(worker):
        """Clean up worker thread"""
        if worker:
            worker.quit()
            worker.wait()
            worker = None
    
    def _on_metrics_current_image(self):
        """Calculate metrics for currently displayed image"""
        if not self.image_path:
            self.status_text.emit("No image loaded")
            return
        
        if self.canvas.overlay_layer.has_changes():
            QMessageBox.warning(
                self,
                "Unsaved Edits",
                "Please save your edits (Ctrl/Command + S) before calculating metrics."
            )
            return
        
        self.status_text.emit(f"Calculating metrics for {self.image_path.name}...")
        self.btn_metrics_current.setEnabled(False)
        self.btn_metrics_current.setStyleSheet(self.button_styles["segment"]["highlighted"])
        
        try:
            metadata = self.db_manager.get_metadata_by_filename(self.image_path.stem)
            self.worker_thread = SingleMetricsWorker(self.step_metrics, self.image_path, metadata.id)
            self.worker_thread.finished.connect(self._on_single_metrics_finished)
            self.worker_thread.finished.connect(lambda x: self._on_worker_finished(self.worker_thread))
            self.worker_thread.start()
        except Exception as e:
            self.status_text.emit(f"Error: {str(e)}")
            self.btn_metrics_current.setStyleSheet(self.button_styles["segment"]["normal"])
        finally:
            self.btn_metrics_current.setEnabled(True)
    
    def _on_single_metrics_finished(self, success: bool):
        """Handle single image metrics calculation completion"""
        if success:
            self.btn_metrics_current.setStyleSheet(self.button_styles["segment"]["finished"])
            self.status_text.emit(f"{self.image_path.name}: Metrics calculated ✓")
        else:
            self.btn_metrics_current.setStyleSheet(self.button_styles["segment"]["normal"])
            self.status_text.emit(f"{self.image_path.name}: Metrics calculation failed ✗")
    
    def _on_calculate_metrics_all(self):
        """Calculate metrics for pending images"""
        pending = self.step_metrics.get_pending_images()
        
        if not pending:
            self.status_text.emit(("No images to calculate metrics for"))
            return
        
        if self.canvas.overlay_layer.has_changes():
            QMessageBox.warning(
                self,
                "Unsaved Edits",
                "Please save your edits (Ctrl/Command + S) before re-segmenting the image."
            )
            return
        
        self.status_text.emit(f"Calculating metrics for {len(pending)} images...")
        self.btn_metrics_all.setEnabled(False)
        self.btn_metrics_all.setStyleSheet(self.button_styles["segment"]["highlighted"])
        
        try:
            self.worker_thread = BatchMetricsWorker(self.step_metrics, pending)
            self.worker_thread.progress.connect(self._on_progress)
            self.worker_thread.finished.connect(self._on_batch_metrics_finished)
            self.worker_thread.finished.connect(lambda x: self._on_worker_finished(self.worker_thread))
            self.worker_thread.start()
        except Exception as e:
            self.status_text.emit(f"Error: {str(e)}")
            self.btn_metrics_all.setStyleSheet(self.button_styles["segment"]["normal"])
        finally:
            self.btn_metrics_all.setEnabled(True)
    
    def _on_progress(self, progress: int, message: str):
        """Handle metrics calculation progress"""
        self.progress_bar.setValue(progress)
        self.status_text.emit(message)
    
    def _on_batch_metrics_finished(self, success: bool):
        """Handle completion of batch metrics calculation"""
        self.btn_metrics_all.setStyleSheet(self.button_styles["segment"]["finished"] if success else self.button_styles["segment"]["normal"])
        self.btn_metrics_all.setEnabled(True)
        self.status_text.emit("Metrics calculation complete!" if success else "Metrics calculation failed!")
        self.progress_bar.setValue(100 if success else 0)
    
    def _on_toggle_fovea(self, checked: bool):
        """Toggle fovea visibility on canvas"""
        if self.canvas:
            self.canvas.set_fovea_visibility(checked)
    
    def _on_fovea_location_selected(self, x: int, y: int):
        """Handle fovea location selection"""
        if not self.image_paths:
            return
        
        name = self.image_path.stem
        
        try:
            self.db_manager.save_metrics_fovea_by_name(name, x, y)
            self.status_text.emit(f"Fovea location saved at ({x}, {y}) for {name}")
        except Exception as e:
            self.status_text.emit(f"Error saving fovea location: {e}")
    