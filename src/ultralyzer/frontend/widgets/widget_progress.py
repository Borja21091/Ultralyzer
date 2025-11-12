from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QProgressBar, QLabel, QTextEdit
)
from PySide6.QtCore import Qt

class PipelineProgressWidget(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        
        # Step progress
        layout.addWidget(QLabel("Current Step:"))
        self.step_label = QLabel("Ready")
        layout.addWidget(self.step_label)
        
        self.step_progress = QProgressBar()
        layout.addWidget(self.step_progress)
        
        # Image progress
        layout.addWidget(QLabel("Image Progress:"))
        self.image_label = QLabel("0 / 0")
        layout.addWidget(self.image_label)
        
        self.image_progress = QProgressBar()
        layout.addWidget(self.image_progress)
        
        # Log output
        layout.addWidget(QLabel("Log:"))
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        layout.addWidget(self.log_text)
    
    def update_progress(self, progress: dict):
        """Update UI with progress information"""
        current = progress["current_step"]
        total = progress["total_steps"]
        self.step_progress.setValue(int((current / total) * 100))
        self.step_label.setText(
            f"Step {current}/{total} - Processing images..."
        )
        
        img_current = progress["images_processed"]
        img_total = progress["total_images"]
        self.image_progress.setValue(int((img_current / img_total) * 100))
        self.image_label.setText(f"{img_current} / {img_total}")
        
        # Append log
        msg = f"[Step {current}] {progress['current_image']} - {progress['status']}"
        self.log_text.append(msg)