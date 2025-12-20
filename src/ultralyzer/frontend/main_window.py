import cv2
import numpy as np
from pathlib import Path
from definitions import IMAGE_FORMATS, METRIC_DICTIONARY, SEG_DIR
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QFileDialog, QMessageBox, QComboBox, QTextEdit, 
    QProgressDialog, QApplication
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from frontend.widgets.widget_base import BaseWidget
from frontend.widgets.widget_s2 import SegmentationWidget
from backend.models.database import DatabaseManager

from backend.utils.threads import AVSegmentationWorker, DiscSegmentationWorker
import photoshopapi as psapi

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
        self.widget: BaseWidget = None
        
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
        
        # Create top menu bar
        self._create_menu_bar()
    
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
        
        # Segmentation menu
        segmentation_menu = menubar.addMenu("Segmentation")
        
        action_save_segmentation = QAction("Save", self)
        action_save_segmentation.triggered.connect(self.widget._save_edits)
        segmentation_menu.addAction(action_save_segmentation)
        
        segmentation_menu.addSeparator()
        
        export_psd_menu = segmentation_menu.addMenu("Export to PSD...")
        
        action_export_psd_all = QAction("All", self)
        action_export_psd_all.triggered.connect(self._on_export_to_psd_all)
        export_psd_menu.addAction(action_export_psd_all)
        
        action_export_psd_current = QAction("Current", self)
        action_export_psd_current.triggered.connect(self._on_export_to_psd_current)
        export_psd_menu.addAction(action_export_psd_current)
        
        segmentation_menu.addSeparator()
        
        action_av_segment = QAction("A/V Segment", self)
        action_av_segment.triggered.connect(self._on_av_segment)
        segmentation_menu.addAction(action_av_segment)
        
        action_disc_segment = QAction("Disc Segment", self)
        action_disc_segment.triggered.connect(self._on_disc_segment)
        segmentation_menu.addAction(action_disc_segment)
        
        # Database menu
        db_menu = menubar.addMenu("Database")
        
        # Assign QC All submenu
        qc_assign_all_menu = db_menu.addMenu("Assign QC All")
        
        action_qc_pass_all = QAction("Pass", self)
        action_qc_pass_all.triggered.connect(lambda: self._on_assign_qc_all("pass"))
        qc_assign_all_menu.addAction(action_qc_pass_all)
        
        action_qc_borderline_all = QAction("Borderline", self)
        action_qc_borderline_all.triggered.connect(lambda: self._on_assign_qc_all("borderline"))
        qc_assign_all_menu.addAction(action_qc_borderline_all)
        
        action_qc_reject_all = QAction("Reject", self)
        action_qc_reject_all.triggered.connect(lambda: self._on_assign_qc_all("reject"))
        qc_assign_all_menu.addAction(action_qc_reject_all)
        
        db_menu.addSeparator()
        
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
        widget = SegmentationWidget(self._db_manager)
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
    
    def _on_assign_qc_all(self, decision: str):
        """Assign QC decision to all images in the current list"""
        if not self.image_list:
            QMessageBox.warning(self, "No Images", "No images loaded to assign QC.")
            return
            
        # Create message box explicitly to better control focus/default
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Confirm Batch Assignment")
        msg_box.setText(f"Are you sure you want to assign '{decision.upper()}' to all {len(self.image_list)} images?")
        msg_box.setIcon(QMessageBox.Icon.Question)
        
        yes_btn = msg_box.addButton(QMessageBox.StandardButton.Yes)
        no_btn = msg_box.addButton(QMessageBox.StandardButton.No)
        
        msg_box.setDefaultButton(no_btn)
        
        msg_box.exec()
        
        if msg_box.clickedButton() == yes_btn:
            count = 0
            for image_name in self.image_list:
                name = Path(image_name).stem
                if self._db_manager.save_qc_result(name, decision):
                    count += 1
            
            self.statusBar().showMessage(f"Assigned {decision.upper()} to {count} / {len(self.image_list)} images.")
            
            # Refresh current image display to show new status
            self.widget.display_image()

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
            
    def _on_export_to_psd_all(self):
        """Export all segmentations to PSD"""
        # Get image list from dropdown
        image_names = self.image_list
        
        if not image_names:
            QMessageBox.warning(self, "No Images", "No images loaded to export.")
            return
        
        save_folder = QFileDialog.getExistingDirectory(self, "Select Folder to Save PSD Files")
        
        # Setup Progress Dialog
        progress = QProgressDialog("Initializing export...", "Cancel", 0, len(image_names), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        
        k = 0
        for i, img_name in enumerate(image_names):
            
            # Check if user clicked cancel
            if progress.wasCanceled():
                break
                
            progress.setLabelText(f"Exporting {img_name}...")
            
            name = Path(img_name).stem
            
            # Find mask in database
            meta = self._db_manager.get_metadata_by_filename(img_name)
            if not meta:
                QMessageBox.warning(self, "Image Not in Database", f"The selected image is not in the database: {name}")
                return
            seg_meta = self._db_manager.get_segmentation_result_by_id(meta.id)
            if not seg_meta:
                QMessageBox.warning(self, "No Segmentation Found", f"No segmentation found for image: {name}")
                return
            img_path = Path(meta.folder) / Path(meta.name + meta.extension)
            seg_path = Path(seg_meta.seg_folder) / Path(meta.name + seg_meta.extension)
            
            # Check if files exist
            if not img_path.exists() or not seg_path.exists():
                QMessageBox.warning(self, "Files Not Found", f"Image or segmentation file not found for: {name}")
                continue
            
            # Read image & mask
            image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            green = cv2.cvtColor(image[:,:,1], cv2.COLOR_GRAY2RGB)
            mask = cv2.imread(str(seg_path), cv2.IMREAD_COLOR)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            w, h = image.shape[1], image.shape[0]

            arteries = np.zeros((h, w, 4), dtype=np.uint8)
            arteries[mask[:,:,0] > 0] = [255, 0, 0, 255]
            veins = np.zeros((h, w, 4), dtype=np.uint8)
            veins[mask[:,:,2] > 0] = [0, 0, 255, 255]
            disc = np.zeros((h, w, 4), dtype=np.uint8)
            disc[mask[:,:,1] > 0] = [0, 255, 0, 255]
            
            # Transpose to (C, H, W) for psapi
            image = np.ascontiguousarray(np.transpose(image, (2, 0, 1)))
            green = np.ascontiguousarray(np.transpose(green, (2, 0, 1)))
            arteries = np.ascontiguousarray(np.transpose(arteries, (2, 0, 1)))
            veins = np.ascontiguousarray(np.transpose(veins, (2, 0, 1)))
            disc = np.ascontiguousarray(np.transpose(disc, (2, 0, 1)))
            
            # Prepare PSD file
            color_mode = psapi.enum.ColorMode.rgb
            psd = psapi.LayeredFile_8bit(color_mode, w, h)
            
            # Center the layers
            cx, cy = w / 2, h / 2
            
            psd.add_layer(psapi.ImageLayer_8bit(arteries, "Arteries", width=w, height=h, opacity=0.5, pos_x=cx, pos_y=cy))
            psd.add_layer(psapi.ImageLayer_8bit(veins, "Veins", width=w, height=h, opacity=0.5, pos_x=cx, pos_y=cy))
            psd.add_layer(psapi.ImageLayer_8bit(disc, "Optic Disc", width=w, height=h, opacity=0.5, pos_x=cx, pos_y=cy))
            psd.add_layer(psapi.ImageLayer_8bit(green, "Green Channel", width=w, height=h, is_visible=True, is_locked=True, pos_x=cx, pos_y=cy))
            psd.add_layer(psapi.ImageLayer_8bit(image, "Color Image", width=w, height=h, is_visible=False, is_locked=True, pos_x=cx, pos_y=cy))
            
            
            # Save PSD file
            psd.write(Path(save_folder) / Path(name + ".psd"))
            k += 1
            
            # Update progress
            progress.setValue(i + 1)
            QApplication.processEvents()
            
        progress.setValue(len(image_names))        
        self.statusBar().showMessage(f"Exported {k} / {len(image_names)} PSD files to: {save_folder}")            
    
    def _on_export_to_psd_current(self):
        """Export current segmentation to PSD"""
        img_name = self.image_dropdown.currentText()
        name = Path(img_name).stem
        
        if not img_name:
            QMessageBox.warning(self, "No Image Selected", "Please select an image to export.")
            return
        
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save PSD File",
            name + ".psd",
            "PSD Files (*.psd);;All Files (*)"
        )
        
        if not save_path:
            return
        
        # Find mask in database
        meta = self._db_manager.get_metadata_by_filename(name)
        if not meta:
            QMessageBox.warning(self, "Image Not in Database", f"The selected image is not in the database: {name}")
            return
        seg_meta = self._db_manager.get_segmentation_result_by_id(meta.id)
        if not seg_meta:
            QMessageBox.warning(self, "No Segmentation Found", f"No segmentation found for image: {name}")
            return
        img_path = Path(meta.folder) / Path(meta.name + meta.extension)
        seg_path = Path(seg_meta.seg_folder) / Path(meta.name + seg_meta.extension)
        
        # Check if files exist
        if not img_path.exists() or not seg_path.exists():
            QMessageBox.warning(self, "Files Not Found", f"Image or segmentation file not found for: {name}")
            return
        
        # Read image & mask
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        green = cv2.cvtColor(image[:,:,1], cv2.COLOR_GRAY2RGB)
        mask = cv2.imread(str(seg_path), cv2.IMREAD_COLOR)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        w, h = image.shape[1], image.shape[0]

        arteries = np.zeros((h, w, 4), dtype=np.uint8)
        arteries[mask[:,:,0] > 0] = [255, 0, 0, 255]
        veins = np.zeros((h, w, 4), dtype=np.uint8)
        veins[mask[:,:,2] > 0] = [0, 0, 255, 255]
        disc = np.zeros((h, w, 4), dtype=np.uint8)
        disc[mask[:,:,1] > 0] = [0, 255, 0, 255]
        
        # Transpose to (C, H, W) for psapi
        image = np.ascontiguousarray(np.transpose(image, (2, 0, 1)))
        green = np.ascontiguousarray(np.transpose(green, (2, 0, 1)))
        arteries = np.ascontiguousarray(np.transpose(arteries, (2, 0, 1)))
        veins = np.ascontiguousarray(np.transpose(veins, (2, 0, 1)))
        disc = np.ascontiguousarray(np.transpose(disc, (2, 0, 1)))
        
        # Prepare PSD file
        w, h = image.shape[2], image.shape[1]
        color_mode = psapi.enum.ColorMode.rgb
        psd = psapi.LayeredFile_8bit(color_mode, w, h)
        
        # Center the layers
        cx, cy = w / 2, h / 2
        
        psd.add_layer(psapi.ImageLayer_8bit(arteries, "Arteries", width=w, height=h, opacity=0.5, pos_x=cx, pos_y=cy))
        psd.add_layer(psapi.ImageLayer_8bit(veins, "Veins", width=w, height=h, opacity=0.5, pos_x=cx, pos_y=cy))
        psd.add_layer(psapi.ImageLayer_8bit(disc, "Optic Disc", width=w, height=h, opacity=0.5, pos_x=cx, pos_y=cy))
        psd.add_layer(psapi.ImageLayer_8bit(green, "Green Channel", width=w, height=h, is_visible=True, is_locked=True, pos_x=cx, pos_y=cy))
        psd.add_layer(psapi.ImageLayer_8bit(image, "Color Image", width=w, height=h, is_visible=False, is_locked=True, pos_x=cx, pos_y=cy))
        
        # Save PSD file
        print(save_path)
        psd.write(save_path)
        
        self.statusBar().showMessage(f"Exported PSD file to: {save_path}")

