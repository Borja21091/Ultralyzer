from pathlib import Path
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox, QComboBox
)
from frontend.widgets.widget_base import BaseWidget
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
    
    def __init__(self):
        super().__init__()
        
        self._image_folder = None
        self._db_manager = DatabaseManager()
        self._image_list = []
        
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
        btn_select_folder.clicked.connect(self._on_select_folder)
        top_layout.addWidget(btn_select_folder)
        
        self.folder_label = QLabel("No folder selected")
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
    
    def _create_widget(self) -> BaseWidget:
        """Create the appropriate widget for the selected step"""
        segmentor = UnetSegmentor()
        widget = SegmentationWidget(segmentor, self._db_manager)
        widget.decision_made.connect(self._on_qc_decision)
        widget.status_text.connect(self.statusBar().showMessage)
        return widget
    
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
    
    def _on_select_folder(self):
        """Select image folder"""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Image Folder"
        )
        
        if not folder:
            return
        
        self._image_folder = Path(folder)
        self.folder_label.setText(f"📂 {self._image_folder.name}")
        self.statusBar().showMessage(f"Loaded folder: {self._image_folder.name}")
        
        # Load images in the appropriate widget
        self.load_images(self._image_folder)
    
    def load_images(self, folder: Path):
        """
        Load images from the specified folder into combobox widget
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
    
    def _on_qc_decision(self, filename: str, decision: str):
        """Handle quality control decision"""
        status = f"Decided: {filename} → {decision.upper()}"
        self.statusBar().showMessage(status)
    