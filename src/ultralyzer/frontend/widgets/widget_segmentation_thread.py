from pathlib import Path
from typing import List
import numpy as np
from PIL import Image

from PySide6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QProgressBar, QComboBox, QListWidget, QListWidgetItem
)
from PySide6.QtGui import QFont, QColor
from PySide6.QtCore import QThread

from frontend.widgets.widget_base import BaseWidget
from backend.models.database import DatabaseManager
from backend.steps.s2_segmentation import SegmentationStep
from backend.models.segmentor import Segmentor


class SegmentationWorker(QThread):
    """Worker thread for segmentation"""
    
    from PySide6.QtCore import Signal
    progress = Signal(int, str)
    finished = Signal(bool)
    
    def __init__(self, seg_step: SegmentationStep, qc_results: List):
        super().__init__()
        self.seg_step = seg_step
        self.qc_results = qc_results
    
    def run(self):
        """Run segmentation batch"""
        total = len(self.qc_results)
        
        for idx, qc_result in enumerate(self.qc_results):
            try:
                success = self.seg_step.process_and_save_to_db(
                    qc_result.image_path,
                    qc_result.id
                )
                
                progress_pct = int((idx + 1) / total * 100)
                msg = f"{Path(qc_result.image_path).name}: {'✓' if success else '✗'}"
                self.progress.emit(progress_pct, msg)
            
            except Exception as e:
                progress_pct = int((idx + 1) / total * 100)
                self.progress.emit(progress_pct, f"Error: {str(e)}")
        
        self.finished.emit(True)


class SegmentationWidget(BaseWidget):
    """UI for segmentation step"""
    
    def __init__(self, segmentor: Segmentor, db_manager: DatabaseManager = None):
        super().__init__(db_manager)
        self.seg_step = SegmentationStep(segmentor, self.db_manager)
        self.worker_thread = None
        self._init_ui()
    
    def _init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout(self)
        
        # Top: Title and controls
        top_layout = QHBoxLayout()
        
        title = QLabel("Step 2: Segmentation (Arteries & Veins)")
        font = QFont()
        font.setPointSize(14)
        font.setBold(True)
        title.setFont(font)
        top_layout.addWidget(title)
        
        top_layout.addStretch()
        
        self.btn_segment = QPushButton("▶ Start Segmentation")
        self.btn_segment.clicked.connect(self.on_start_segmentation)
        self.btn_segment.setMaximumWidth(200)
        top_layout.addWidget(self.btn_segment)
        
        layout.addLayout(top_layout)
        
        # Main content: Canvas with right panel
        splitter, self.canvas, right_panel = self._create_main_splitter()
        
        # Configure right panel
        right_layout = right_panel.layout()
        right_layout.addWidget(QLabel("Display Options:"))
        
        self.combo_overlay = QComboBox()
        self.combo_overlay.addItems(["Combined", "Arteries Only", "Veins Only", "Original"])
        self.combo_overlay.currentTextChanged.connect(self._on_overlay_changed)
        right_layout.addWidget(self.combo_overlay)
        
        right_layout.addWidget(QLabel("Segmented Images:"))
        self.results_list = QListWidget()
        right_layout.addWidget(self.results_list)
        
        layout.addWidget(splitter, 1)
        
        # Bottom: Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumHeight(30)
        layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)
    
    def on_start_segmentation(self):
        """Start segmentation"""
        pending = self.seg_step.get_pending_images()
        
        if not pending:
            self.status_label.setText("No images to segment")
            return
        
        self.status_label.setText(f"Segmenting {len(pending)} images...")
        self.btn_segment.setEnabled(False)
        
        self.worker_thread = SegmentationWorker(self.seg_step, pending)
        self.worker_thread.progress.connect(self._on_progress)
        self.worker_thread.finished.connect(self._on_finished)
        self.worker_thread.start()
    
    def _on_progress(self, progress: int, message: str):
        """Handle progress"""
        self.progress_bar.setValue(progress)
        self.status_label.setText(message)
    
    def _on_finished(self, success: bool):
        """Handle completion"""
        self.btn_segment.setEnabled(True)
        self.status_label.setText("Complete!" if success else "Failed!")
        self.progress_bar.setValue(100 if success else 0)
        self._refresh_results_list()
    
    def _refresh_results_list(self):
        """Refresh results list"""
        self.results_list.clear()
        
        seg_results = self.db_manager.get_all_segmentation_results()
        
        for seg_result in seg_results:
            qc_result = self.db_manager.get_qc_result(seg_result.qc_result.image_path)
            item = QListWidgetItem(Path(qc_result.image_path).name)
            item.setForeground(QColor("#2f9e44"))
            self.results_list.addItem(item)
    
    def _on_overlay_changed(self, option: str):
        """Handle overlay change"""
        pass
    
    def load_segmentation_result(self, seg_result_id: int):
        """Load segmentation result"""
        seg_result = self.db_manager.get_segmentation_result(seg_result_id)
        
        if seg_result:
            combined_img = np.array(Image.open(seg_result.combined_mask_path))
            self.canvas.load_image_array(combined_img)