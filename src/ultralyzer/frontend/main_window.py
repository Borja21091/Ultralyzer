from pathlib import Path
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox, QComboBox
)
from frontend.widgets.widget_base import BaseWidget
from frontend.widgets.widget_s1 import QualityControlWidget
from frontend.widgets.widget_s2 import SegmentationWidget
from backend.models.database import DatabaseManager
from backend.models.segmentor import UnetSegmentor

class MainWindow(QMainWindow):
    """
    Main application window - displays selected processing step
    """
    
    STEPS = {
        "qc": {"name": "Quality Control", "order": 1},
        "seg": {"name": "Segmentation", "order": 2},
        "properties": {"name": "Properties", "order": 3}
    }
    
    def __init__(self, step: str = "qc"):
        super().__init__()
        
        # Validate step
        if step not in self.STEPS:
            raise ValueError(f"Invalid step: {step}. Must be one of {list(self.STEPS.keys())}")
        
        self.current_step = step
        self.image_folder = None
        self.db_manager = DatabaseManager()
        
        # Initialize window
        self.setWindowTitle("Ultralyzer - Retinal Image Processing Pipeline")
        self.showMaximized()
        
        # Create central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Top: Folder selection and step info
        top_layout = QHBoxLayout()
        
        # Folder selection
        btn_select_folder = QPushButton("📁 Select Image Folder")
        btn_select_folder.clicked.connect(self.on_select_folder)
        top_layout.addWidget(btn_select_folder)
        
        self.folder_label = QLabel("No folder selected")
        self.folder_label.setWordWrap(True)
        top_layout.addWidget(self.folder_label, 1)
        
        # Dropdown with image names in folder
        self.image_dropdown = QComboBox()
        self.image_dropdown.setPlaceholderText("Select an image")
        self.image_dropdown.activated.connect(self.on_select_image)
        top_layout.addWidget(self.image_dropdown)

        main_layout.addLayout(top_layout)
        
        # Create step-specific widget
        self.step_widget = self._create_step_widget()
        self.step_widget.index_changed.connect(self.image_dropdown.setCurrentIndex)
        main_layout.addWidget(self.step_widget, 1)
        
        # Status bar
        self.statusBar().showMessage("Ready")
        
    def _create_step_widget(self) -> BaseWidget:
        """Create the appropriate widget for the selected step"""

        if self.current_step == "qc":
            widget = QualityControlWidget(self.db_manager)
            widget.decision_made.connect(self.on_qc_decision)
            widget.all_images_reviewed.connect(self.on_all_images_reviewed)
            return widget
        
        elif self.current_step == "seg":
            segmentor = UnetSegmentor()
            widget = SegmentationWidget(segmentor, self.db_manager)
            widget.status_text.connect(self.statusBar().showMessage)
            return widget
        
        else:
            raise ValueError(f"Unknown step: {self.current_step}")
    
    def on_select_image(self, img_idx: int):
        """Handle image selection from dropdown"""        
        if not self.image_folder:
            QMessageBox.warning(self, "No Folder Selected", "Please select an image folder first.")
            return
        
        self.step_widget.index = img_idx
        self.step_widget.display_image()
        
        image_name = self.image_dropdown.itemText(img_idx)
        image_path = self.image_folder / image_name
        if not image_path.exists():
            QMessageBox.warning(self, "Image Not Found", f"The selected image does not exist: {image_path}")
            return
    
    def on_select_folder(self):
        """Select image folder"""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Image Folder"
        )
        
        if not folder:
            return
        
        self.image_folder = Path(folder)
        self.folder_label.setText(f"📂 {self.image_folder.name}")
        self.statusBar().showMessage(f"Loaded folder: {self.image_folder.name}")
        
        # Load images in the appropriate widget
        self.load_images(self.image_folder)
    
    def load_images(self, folder: Path):
        """
        Load images from the specified folder into combobox widget
        """
        if not folder.is_dir():
            raise ValueError(f"Invalid folder: {folder}")
        
        self.step_widget.load_images(folder)
        image_files = self.step_widget.image_paths
        image_files = [p.name for p in image_files]
        
        if not image_files:
            return False
        
        self.image_dropdown.clear()
        self.image_dropdown.addItems(image_files)
        
        return True
    
    def on_qc_decision(self, filename: str, decision: str):
        """Handle quality control decision"""
        status = f"Decided: {filename} → {decision.upper()}"
        self.statusBar().showMessage(status)
    
    def on_all_images_reviewed(self):
        """Handle completion of QC review"""
        stats = self.db_manager.get_statistics()
        
        summary = (
            f"Quality Control Review Complete!\n\n"
            f"✅ PASS: {stats['pass']}\n"
            f"⚠️ BORDERLINE: {stats['borderline']}\n"
            f"❌ REJECT: {stats['reject']}\n\n"
            f"Total: {stats['total']} images"
        )
        
        QMessageBox.information(self, "Review Complete", summary)
        self.statusBar().showMessage("QC review complete")
        
