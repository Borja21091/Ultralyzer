import cv2
import numpy as np
from pathlib import Path
from definitions import IMAGE_FORMATS, METRIC_DICTIONARY, SEG_DIR
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QFileDialog, QMessageBox, QComboBox, QTextEdit
)
from PySide6.QtGui import QAction
from frontend.widgets.widget_base import BaseWidget
from frontend.widgets.widget_s2 import SegmentationWidget
from backend.models.database import DatabaseManager
from backend.models.segmentor import UnetSegmentor

from backend.utils.threads import AVSegmentationWorker, DiscSegmentationWorker

class MainWindow(QMainWindow):
    """
    Main application window - displays selected processing step
    """
    
    def __init__(self):
        super().__init__()
        
        self._image_folder = None
        self._mask_folder = None
        self._db_manager = DatabaseManager()
        self._image_list = []
        self.worker = None
        
        self._init_ui()
    
    ############ PROPERTIES ############
    
    @property
    def image_folder(self):
        """Get current image folder"""
        return self._image_folder
    
    @property
    def db_manager(self):
        """Get database manager"""
        return self._db_manager
    
    @property
    def image_list(self):
        return self._image_list
    
    @image_list.setter
    def image_list(self, files: list):
        self._image_list = files
    
    ############ UI ############
    
    def _init_ui(self):
        """Initialize main window UI"""
        
        # Create menu bar
        self._create_menu_bar()
        
        # Initialize window
        self.setWindowTitle("Ultralyzer - Retinal Image Processing Pipeline")
        self.showMaximized()
        
        # Create central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Top: Folder label & image dropdown
        top_layout = QHBoxLayout()
        
        # Folder label
        self.folder_label = QLabel("No image folder loaded")
        self.folder_label.setWordWrap(True)
        top_layout.addWidget(self.folder_label, 1)
        
        # Dropdown with image names in folder
        self.image_dropdown = QComboBox()
        self.image_dropdown.setPlaceholderText("Select an image")
        self.image_dropdown.activated.connect(self._on_select_image)
        top_layout.addWidget(self.image_dropdown)

        main_layout.addLayout(top_layout)
        
        # Create step-specific widget
        self.widget = self._create_widget()
        self.widget.index_changed.connect(self.image_dropdown.setCurrentIndex)
        main_layout.addWidget(self.widget, 1)
        
        # Status bar
        self.statusBar().showMessage("Ready")
    
    def _create_menu_bar(self):
        """Create the application menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        action_open_image_folder = QAction("Open Image Folder", self)
        action_open_image_folder.triggered.connect(self._on_select_image_folder)
        file_menu.addAction(action_open_image_folder)
        
        action_load_mask_folder = QAction("Load Mask Folder", self)
        action_load_mask_folder.triggered.connect(self._on_select_mask_folder)
        file_menu.addAction(action_load_mask_folder)
        
        # file_menu.addSeparator()
        
        # Segmentation menu
        segmentation_menu = menubar.addMenu("Segmentation")
        
        action_av_segment = QAction("A/V Segment", self)
        action_av_segment.triggered.connect(self._on_av_segment)
        segmentation_menu.addAction(action_av_segment)
        
        action_disc_segment = QAction("Disc Segment", self)
        action_disc_segment.triggered.connect(self._on_disc_segment)
        segmentation_menu.addAction(action_disc_segment)
        
        # Database menu
        db_menu = menubar.addMenu("Database")
        
        action_export_QC = QAction("Export QC Results", self)
        action_export_QC.triggered.connect(self._on_export_qc_results)
        db_menu.addAction(action_export_QC)
        
        action_export_metrics = QAction("Export Metrics", self)
        action_export_metrics.triggered.connect(self._on_export_metrics)
        db_menu.addAction(action_export_metrics)
        
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        action_metric_definitions = QAction("Metric Definitions", self)
        action_metric_definitions.triggered.connect(self._on_metric_definitions)
        help_menu.addAction(action_metric_definitions)
        
        action_about = QAction("About", self)
        action_about.triggered.connect(self._on_about)
        help_menu.addAction(action_about)
    
    ############ PUBLIC METHODS ############
    
    ############ PRIVATE METHODS ############
    
    def _create_widget(self) -> BaseWidget:
        """Create the appropriate widget for the selected step"""
        segmentor = UnetSegmentor()
        widget = SegmentationWidget(segmentor, self._db_manager)
        widget.decision_made.connect(self._on_qc_decision)
        widget.status_text.connect(self.statusBar().showMessage)
        return widget
    
    def _load_images(self, folder: Path):
        """
        Load images from the specified folder into database & combobox widget
        """
        if not folder.is_dir():
            raise ValueError(f"Invalid folder: {folder}")
        
        self.widget.load_images(folder)
        image_files = self.widget.image_paths
        image_files = [p.name for p in image_files]
        self.image_list = image_files
        
        if not self.image_list:
            return False
        
        self.image_dropdown.clear()
        self.image_dropdown.addItems(image_files)
        
        return True
    
    def _load_mask_info_to_db(self, mask_folder: Path):
        """Load mask information from folder into database"""
        mask_files = list(mask_folder.glob("*"))
        mask_files = [f for f in mask_files if f.suffix.lower() in IMAGE_FORMATS]
        for mask_file in mask_files:
            mask_folder = mask_file.parent
            mask_name = mask_file.name
            mask_suffix = mask_file.suffix.lower()
            if mask_suffix not in IMAGE_FORMATS:
                continue
            meta = self._db_manager.get_metadata_by_filename(mask_name)
            if not meta:
                continue
            self._db_manager.set_mask_info(meta.id, mask_folder, mask_suffix)
    
    def _save_empty_mask(self, image_path: Path, seg_path: Path):
        """Save an empty segmentation mask to the specified path"""
        image = cv2.imread(str(image_path))
        empty_mask = np.zeros(image.shape[:2] + (3,), dtype=np.uint8)
        cv2.imwrite(str(seg_path), empty_mask)
    
    ############ ACTIONS ############
    
    def _on_about(self):
        """Show about dialog"""
        QMessageBox.information(
            self,
            "About Ultralyzer",
            "Ultralyzer - Retinal Image Processing Pipeline\n\nVersion 1.0"
        )
    
    def _on_metric_definitions(self):
        """Show metric definitions dialog"""
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Metric Definitions")
        dialog.setText("Metric Definitions")
        
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setMarkdown(''.join(f"<p><b>{key}</b>: {value}</p>" for key, value in METRIC_DICTIONARY.items()))
        text_edit.setMinimumWidth(800)
        text_edit.setMinimumHeight(600)
        
        dialog.layout().addWidget(text_edit, 0, 1)
        dialog.exec()
    
    def _on_qc_decision(self, filename: str, decision: str):
        """Handle quality control decision"""
        status = f"Decided: {filename} → {decision.upper()}"
        self.statusBar().showMessage(status)
    
    def _on_select_image(self, img_idx: int):
        """Handle image selection from dropdown"""        
        if not self._image_folder:
            QMessageBox.warning(self, "No Folder Selected", "Please select an image folder first.")
            return
        
        self.widget.index = img_idx
        self.widget.display_image()
        
        image_name = self.image_dropdown.itemText(img_idx)
        image_path = self._image_folder / image_name
        if not image_path.exists():
            QMessageBox.warning(self, "Image Not Found", f"The selected image does not exist: {image_path}")
            return
    
    def _on_select_image_folder(self):
        """Select image folder"""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Image Folder"
        )
        
        if not folder:
            return
        
        self._image_folder = Path(folder)
        self.folder_label.setText(f"📂 {self._image_folder.name}")
        self.statusBar().showMessage(f"Loaded folder: {self._image_folder}")
        
        # Load images in the appropriate widget
        self._load_images(self._image_folder)
    
    def _on_select_mask_folder(self):
        """Select mask folder"""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Mask Folder"
        )
        
        if not folder:
            return
        self._mask_folder = Path(folder)
        self._load_mask_info_to_db(self._mask_folder)
        self.statusBar().showMessage(f"Loaded masks from: {self._mask_folder}")
    
    def _on_av_segment(self):
        """Handle A/V segmentation action"""
        
        # Get current image
        img_name = self.image_dropdown.currentText()
        if not img_name:
            QMessageBox.warning(self, "No Image Selected", "Please select an image to segment.")
            return
        
        # Find in database
        meta = self._db_manager.get_metadata_by_filename(img_name)
        if not meta:
            QMessageBox.warning(self, "Image Not in Database", f"The selected image is not in the database: {img_name}")
            return
        image_path = Path(meta.folder) / Path(meta.name + meta.extension)
        
        # Find segmentation mask details
        seg_meta = self._db_manager.get_segmentation_result_by_id(meta.id)
        if not seg_meta:
            # Save new segmentation entry
            seg_meta = {
                "id": meta.id,
                "extension": ".png",
                "seg_folder": SEG_DIR,
                "model_name": self.widget.step_seg.segmentor.model_name,
                "model_version": self.widget.step_seg.segmentor.model_version
            }
            self._db_manager.save_segmentation_result(**seg_meta)
            # Save empty segmentation mask
            seg_path = Path(SEG_DIR) / Path(meta.name + seg_meta["extension"])
            self._save_empty_mask(image_path, seg_path)
        else:
            seg_path = Path(seg_meta.seg_folder) / Path(meta.name + seg_meta.extension)
        
        # Perform segmentation
        try:
            self.statusBar().showMessage(f"Segmenting A/V for image: {img_name}")
            self.worker = AVSegmentationWorker(self.widget.step_seg, image_path, seg_path)
            self.worker.finished.connect(lambda success: self.statusBar().showMessage(
                f"A/V Segmentation {'succeeded' if success else 'failed'} for image: {img_name}"
            ))
            self.worker.finished.connect(self.widget.display_image)
            self.worker.start()
            
        except Exception as e:
            self.statusBar().showMessage(f"Error during A/V segmentation: {str(e)}")
            
        # Update display
        self.widget.display_image()
    
    def _on_disc_segment(self):
        """Handle Disc segmentation action"""
        
        # Get current image
        img_name = self.image_dropdown.currentText()
        if not img_name:
            QMessageBox.warning(self, "No Image Selected", "Please select an image to segment.")
            return
        
        # Find in database
        meta = self._db_manager.get_metadata_by_filename(img_name)
        if not meta:
            QMessageBox.warning(self, "Image Not in Database", f"The selected image is not in the database: {img_name}")
            return
        image_path = Path(meta.folder) / Path(meta.name + meta.extension)
        
        # Find segmentation mask details
        seg_meta = self._db_manager.get_segmentation_result_by_id(meta.id)
        if not seg_meta:
            # Save new segmentation entry
            seg_meta = {
                "id": meta.id,
                "extension": ".png",
                "seg_folder": SEG_DIR,
                "model_name": self.widget.step_seg.segmentor.model_name,
                "model_version": self.widget.step_seg.segmentor.model_version
            }
            self._db_manager.save_segmentation_result(**seg_meta)
            # Save empty segmentation mask
            seg_path = Path(SEG_DIR) / Path(meta.name + seg_meta["extension"])
            self._save_empty_mask(image_path, seg_path)
        else:
            seg_path = Path(seg_meta.seg_folder) / Path(meta.name + seg_meta.extension)
        
        # Perform segmentation
        try:
            self.statusBar().showMessage(f"Segmenting Disc for image: {img_name}")
            self.worker = DiscSegmentationWorker(self.widget.step_seg, image_path, seg_path)
            self.worker.finished.connect(lambda success: self.statusBar().showMessage(
                f"Disc Segmentation {'succeeded' if success else 'failed'} for image: {img_name}"
            ))
            self.worker.finished.connect(self.widget.display_image)
            self.worker.start()
            
        except Exception as e:
            self.statusBar().showMessage(f"Error during disc segmentation: {str(e)}")
    
    def _on_export_qc_results(self):
        """Export QC results from database"""
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save QC Results",
            "qc_data.csv",
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if not save_path:
            return
        
        try:
            self._db_manager.export_qc_results(Path(save_path))
            self.statusBar().showMessage(f"QC results exported to: {save_path}")
        except Exception as e:
            self.statusBar().showMessage(f"Error exporting QC results: {str(e)}")
    
    def _on_export_metrics(self):
        """Export metrics from database"""
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Metrics",
            "metrics_data.csv",
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if not save_path:
            return
        
        try:
            self._db_manager.export_metrics_results(Path(save_path))
            self.statusBar().showMessage(f"Metrics exported to: {save_path}")
        except Exception as e:
            self.statusBar().showMessage(f"Error exporting metrics: {str(e)}")
            
        