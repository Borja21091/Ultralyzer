from pathlib import Path
from PySide6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QTextEdit, QProgressBar, QComboBox
)
from PySide6.QtCore import Qt, Signal
from definitions import IMAGE_CHANNEL_MAP
from PySide6.QtGui import QShortcut, QKeySequence
from frontend.widgets.widget_base import BaseWidget
from backend.models.database import DatabaseManager

BLANK_STATE = {
    'filename': '', 
    'decision': None, 
    'notes': ''
}

class QualityControlWidget(BaseWidget):
    """Interactive quality control interface"""
    
    # Signals
    decision_made = Signal(str, str)  # filename, decision
    all_images_reviewed = Signal()
    
    def __init__(self, db_manager: DatabaseManager = None):
        super().__init__(db_manager)
        self._init_ui()
        
        self.state = BLANK_STATE.copy()
    
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
    def channel(self) -> str:
        """Get current display channel"""
        return self.channel_combo.currentText().lower()
    
    @channel.setter
    def channel(self, value: str):
        """Set current display channel"""
        if value.lower() in IMAGE_CHANNEL_MAP.keys():
            index = list(IMAGE_CHANNEL_MAP.keys()).index(value.lower())
            self.channel_combo.setCurrentIndex(index)
    
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
        
        # Main content area: Canvas
        panel, self.canvas = self._create_main_panel()
        self.canvas.image_clicked.connect(self._on_canvas_clicked)

        layout.addWidget(panel, 1)

        # Bottom section: Controls
        bottom_layout = QHBoxLayout()
        
        # Left: Channel selector & Notes
        controls_layout = QHBoxLayout()
        
        # Channel selector
        channel_layout = QVBoxLayout()
        self.channel_combo = QComboBox()
        self.channel_combo.setPlaceholderText("Display Channel")
        self.channel_combo.addItems(["Color", "Red", "Green", "Blue"])
        self.channel_combo.currentTextChanged.connect(self._on_image_channel_changed)
        channel_layout.addWidget(self.channel_combo)
        channel_layout.addStretch()
        controls_layout.addLayout(channel_layout, 1)

        # Notes
        notes_layout = QVBoxLayout()
        notes_layout.addWidget(QLabel("Notes (optional):"))
        
        self.notes_text = QTextEdit()
        self.notes_text.setPlaceholderText("Add any observations about the image...")
        notes_layout.addWidget(self.notes_text)
        controls_layout.addLayout(notes_layout, 9)

        bottom_layout.addLayout(controls_layout, 3)

        # Right: Decision & navigation buttons
        buttons_layout = self._create_decision_buttons()
        bottom_layout.addLayout(buttons_layout, 1)
        
        layout.addLayout(bottom_layout)
        
        # Shortcuts
        self._setup_shortcuts()
        self.canvas.setFocus()
        
    def _create_decision_buttons(self) -> QVBoxLayout:
        """Create decision button layout"""
        buttons_layout = QVBoxLayout()
        button_styles = self.button_styles
        
        # Reject button
        self.btn_reject = QPushButton("❌ REJECT")
        self.btn_reject.setMaximumWidth(200)
        self.btn_reject.setMinimumHeight(40)
        self.btn_reject.setStyleSheet(button_styles["reject"]["normal"])
        self.btn_reject.clicked.connect(lambda: self._on_decision("reject"))
        buttons_layout.addWidget(self.btn_reject)
        
        # Borderline button
        self.btn_borderline = QPushButton("⚠️ BORDERLINE")
        self.btn_borderline.setMaximumWidth(200)
        self.btn_borderline.setMinimumHeight(40)
        self.btn_borderline.setStyleSheet(button_styles["borderline"]["normal"])
        self.btn_borderline.clicked.connect(lambda: self._on_decision("borderline"))
        buttons_layout.addWidget(self.btn_borderline)
        
        # Pass button
        self.btn_pass = QPushButton("✅ PASS")
        self.btn_pass.setMaximumWidth(200)
        self.btn_pass.setMinimumHeight(40)
        self.btn_pass.setStyleSheet(button_styles["pass"]["normal"])
        self.btn_pass.clicked.connect(lambda: self._on_decision("pass"))
        buttons_layout.addWidget(self.btn_pass)
        
        buttons_layout.addStretch()
        
        # Navigation buttons
        nav_layout = QHBoxLayout()
        
        self.btn_prev = QPushButton("◀ Previous")
        self.btn_prev.setMaximumWidth(90)
        self.btn_prev.setMinimumHeight(35)
        self.btn_prev.setStyleSheet(button_styles["nav"]["normal"])
        self.btn_prev.clicked.connect(self._on_prev)
        nav_layout.addWidget(self.btn_prev)
        
        self.btn_next = QPushButton("Next ▶")
        self.btn_next.setMaximumWidth(90)
        self.btn_next.setMinimumHeight(35)
        self.btn_next.setStyleSheet(button_styles["nav"]["normal"])
        self.btn_next.clicked.connect(self._on_next)
        nav_layout.addWidget(self.btn_next)
        
        buttons_layout.addLayout(nav_layout)
        
        return buttons_layout
        
    def _setup_shortcuts(self):
        """Create keyboard shortcuts"""
        sc_next = QShortcut(QKeySequence(Qt.Key_Right), self)
        sc_next.activated.connect(self._on_next)
        sc_prev = QShortcut(QKeySequence(Qt.Key_Left), self)
        sc_prev.activated.connect(self._on_prev)
        
        sc_c = QShortcut(QKeySequence("C"), self)
        sc_c.activated.connect(lambda: self._set_channel_by_key("color"))
        sc_g = QShortcut(QKeySequence("G"), self)
        sc_g.activated.connect(lambda: self._set_channel_by_key("green"))
        sc_r = QShortcut(QKeySequence("R"), self)
        sc_r.activated.connect(lambda: self._set_channel_by_key("red"))
        sc_b = QShortcut(QKeySequence("B"), self)
        sc_b.activated.connect(lambda: self._set_channel_by_key("blue"))
    
    def load_images(self, image_folder: Path):
        """Load all images from folder"""
        if not super().load_images(image_folder):
            self.image_name_label.setText("No images found")
            return False
        
        self._display_new_image()
        return True
    
    ############ WRAPPER METHODS ############
    
    def display_image(self):
        """Wrapper method to display new image"""
        self._display_new_image()
    
    def _set_channel_by_key(self, channel: str):
        """Wrapper for channel property setter"""
        self.channel = channel
    
    ############ SUBCLASS METHODS ############
    
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

    def reset_state(self):
        self.state = BLANK_STATE.copy()
        self.notes_text.clear()
        self.canvas_color = "default"
    
    def _display_new_image(self):
        """Save old state and display new image"""
        if not self.image_paths:
            return
        
        # Save previous data before switching
        self._save_state()
        self.reset_state()
        
        current_path = self.image_path
        
        # Load image
        self.canvas.load_image(str(current_path))
        
        # Update UI
        self.image_name_label.setText(current_path.name)
        self.image_counter_label.setText(
            f"{self.index + 1} / {len(self.image_paths)}"
        )
        self.progress_bar.setValue(self.get_progress_percentage())
        self.channel_combo.setCurrentIndex(0)
        
        # Load existing QC result if any
        self.notes_text.clear()
        existing_result = self.db_manager.get_qc_result(str(current_path.name))
        if existing_result:
            notes = existing_result.notes
            self.notes_text.setText(notes)
            decision = existing_result.decision.value
            self._highlight_decision(decision)
            self.canvas_color = decision
        else:
            self._clear_decision_highlight()
            self.canvas_color = "default"
        
        # Set widget state
        state = {
            'filename': str(current_path.name), 
            'decision': decision if existing_result else None, 
            'notes': notes if existing_result else ""
        }
        self.state = state

        # Update button states
        self.btn_prev.setEnabled(self.index > 0)
        self.btn_next.setEnabled(self.index < len(self.image_paths) - 1)

        self.canvas.setFocus()
    
    ############ ACTIONS ############
    
    def _on_decision(self, decision: str):
        """Handle decision"""
        if not self.image_paths:
            return
        
        notes = self.notes_text.toPlainText()
        
        self._highlight_decision(decision)
        self.canvas_color = decision

        # Update state
        self.state['decision'] = decision
        self.state['notes'] = notes

        # Save to database & emit signal
        if self._save_state():
            self.decision_made.emit(self.state['filename'], decision)

    def _on_next(self):
        """Move to next image"""
        if self.next_image():
            self._display_new_image()
        else:
            self.all_images_reviewed.emit()
    
    def _on_prev(self):
        """Move to previous image"""
        if self.previous_image():
            self._display_new_image()
    
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