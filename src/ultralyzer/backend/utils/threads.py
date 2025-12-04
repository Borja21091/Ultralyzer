import cv2
from pathlib import Path
from PySide6.QtCore import Signal, QThread
from backend.steps.metrics import MetricsStep
from backend.steps.segmentation import SegmentationStep


class BatchSegmentationWorker(QThread):
    """Worker thread for batch segmentation"""
    
    progress = Signal(int, str)
    finished = Signal(bool)
    
    def __init__(self, step_seg: SegmentationStep, metadata: list):
        super().__init__()
        self.step_seg = step_seg
        self.metadata = metadata
    
    def run(self):
        """Run segmentation batch"""
        total = len(self.metadata)
        
        for idx, meta in enumerate(self.metadata):
            try:
                image_path = Path(meta.folder) / Path(meta.name + meta.extension)
                success = self.step_seg.process_and_save_to_db(
                    str(image_path),
                    meta.id,
                    ".png"
                )
                
                progress_pct = int((idx + 1) / total * 100)
                msg = f"{meta.name}: {'✓' if success else '✗'}"
                self.progress.emit(progress_pct, msg)
            
            except Exception as e:
                progress_pct = int((idx + 1) / total * 100)
                self.progress.emit(progress_pct, f"Error: {str(e)}")
        
        self.finished.emit(True)


class SingleSegmentationWorker(QThread):
    """Worker thread for single image segmentation"""
    
    finished = Signal(bool)
    
    def __init__(self, step_seg: SegmentationStep, image_path: Path, id: int):
        super().__init__()
        self.step_seg = step_seg
        self.image_path = image_path
        self.id = id
        
    def run(self):
        """Run segmentation for a single image"""
        try:
            success = self.step_seg.process_and_save_to_db(
                str(self.image_path),
                self.id, 
                ".png"
            )
            self.finished.emit(success)
        except Exception as e:
            self.finished.emit(False)


class AVSegmentationWorker(QThread):
    """Worker thread for A/V segmentation"""
    
    finished = Signal(bool)
    
    def __init__(self, step_seg: SegmentationStep, image_path: Path, seg_path: Path):
        super().__init__()
        self.step_seg = step_seg
        self.image_path = image_path
        self.seg_path = seg_path
        
    def run(self):
        """Run A/V segmentation for a single image"""
        try:
            # Load image
            image = cv2.imread(str(self.image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            av_mask, success = self.step_seg.segment_av(image)
            
            if success:
                # Load mask
                seg_mask = cv2.imread(str(self.seg_path))
                # Replace vessel channels (BGR format)
                seg_mask[:, :, 2] = av_mask[:, :, 0] # Arteries
                seg_mask[:, :, 0] = av_mask[:, :, 2] # Veins
                # Save updated segmentation mask
                cv2.imwrite(str(self.seg_path), seg_mask)
            
            self.finished.emit(success)
        except Exception as e:
            print(f'Error during A/V segmentation: {str(e)}')
            self.finished.emit(False)
            

class DiscSegmentationWorker(QThread):
    """Worker thread for disc segmentation"""
    
    finished = Signal(bool)
    
    def __init__(self, step_seg: SegmentationStep, image_path: Path, seg_path: Path):
        super().__init__()
        self.step_seg = step_seg
        self.image_path = image_path
        self.seg_path = seg_path
        
    def run(self):
        """Run disc segmentation for a single image"""
        try:
            # Load image
            image = cv2.imread(str(self.image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            disc_mask = self.step_seg.segment_disc(image)
            
            # Load mask
            seg_mask = cv2.imread(str(self.seg_path))
            # Replace disc channel (G channel)
            seg_mask[:, :, 1] = disc_mask
            # Save updated segmentation mask
            cv2.imwrite(str(self.seg_path), seg_mask)
            
            self.finished.emit(True)
        except Exception as e:
            print(f'Error during disc segmentation: {str(e)}')
            self.finished.emit(False)


class BatchMetricsWorker(QThread):
    """Worker thread for batch metrics calculation"""
    
    progress = Signal(int, str)
    finished = Signal(bool)
    
    def __init__(self, step_metrics: MetricsStep, metadata: list):
        super().__init__()
        self.step_metrics = step_metrics
        self.metadata = metadata
    
    def run(self):
        """Run metrics calculation batch"""
        total = len(self.metadata)
        
        for idx, meta in enumerate(self.metadata):
            try:
                image_path = Path(meta.folder) / Path(meta.name + meta.extension)
                success = self.step_metrics.process_and_save_to_db(
                    str(image_path),
                    meta.id
                )
                
                progress_pct = int((idx + 1) / total * 100)
                msg = f"{meta.name}: {'✓' if success else '✗'}"
                self.progress.emit(progress_pct, msg)
            
            except Exception as e:
                progress_pct = int((idx + 1) / total * 100)
                self.progress.emit(progress_pct, f"Error: {str(e)}")
        
        self.finished.emit(True)


class SingleMetricsWorker(QThread):
    """Worker thread for single image metrics calculation"""
    
    finished = Signal(bool)
    
    def __init__(self, step_metrics, image_path: Path, id: int):
        super().__init__()
        self.step_metrics = step_metrics
        self.image_path = image_path
        self.id = id
        
    def run(self):
        """Run metrics calculation for a single image"""
        try:
            success = self.step_metrics.process_and_save_to_db(
                str(self.image_path),
                self.id
            )
            self.finished.emit(success)
        except Exception as e:
            self.finished.emit(False)

