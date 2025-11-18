from definitions import SEG_DIR
from typing import Dict, Any
from pathlib import Path
from PIL import Image
import numpy as np
import logging

from backend.models.database import DatabaseManager
from backend.steps.base_step import ProcessingStep
from backend.models.segmentor import Segmentor

class SegmentationStep(ProcessingStep):
    
    def __init__(
        self,
        segmentor: Segmentor,
        db_manager: DatabaseManager = None,
        output_dir: Path = None,
        disc_segmentor: Segmentor = None,
        fovea_segmentor: Segmentor = None,
        vessel_segmentor: Segmentor = None
    ):
        super().__init__("Segmentation", 2)
        self.segmentor = segmentor
        self.disc_segmentor = disc_segmentor
        self.fovea_segmentor = fovea_segmentor
        self.vessel_segmentor = vessel_segmentor
        self.db_manager = db_manager or DatabaseManager()
        
        # Set output directory
        if output_dir is None:
            output_dir = Path(SEG_DIR)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def process(self, image_path: str) -> Dict[str, Any]:
        """
        Segment a single image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with segmentation results
        """
        if not self.validate_input(Path(image_path)):
            return {"success": False, "error": "Invalid input"}
        
        try:
            # Load image
            image = np.array(Image.open(image_path).convert("RGB"))
            
            # Segment
            self.logger.info(f"Segmenting {Path(image_path).name}...")
            av_mask, vessel_mask = self.segmentor.segment(image)

            # TODO: Improve segmentation by using vessel binary segmentation
            if self.vessel_segmentor:
                _, v_mask = self.vessel_segmentor.segment(image)
                (self.output_dir / "binary").mkdir(parents=True, exist_ok=True)
                Image.fromarray(v_mask * 255).save(str(self.output_dir / "binary" / f"{base_name}.png"))

            if self.disc_segmentor:
                av_mask[:, :, 1] = self.disc_segmentor.segment(image)
            
            if self.fovea_segmentor:
                mask, loc = self.fovea_segmentor.segment(image)
                av_mask[:, :, 1] = av_mask[:, :, 1] | mask
            
            # Save masks
            base_name = Path(image_path).stem
            mask_path = self.output_dir / "mask"
            mask_path.mkdir(parents=True, exist_ok=True)
            av_folder = self.output_dir / "av"
            av_folder.mkdir(parents=True, exist_ok=True)
            Image.fromarray(vessel_mask).save(str(mask_path / f"{base_name}.png"))
            Image.fromarray(av_mask).save(str(av_folder / f"{base_name}.png"))

            self.logger.info(f"Segmentation complete: {Path(image_path).name}")
            
            return {
                "success": True,
                "image_name": str(base_name),
                "vessel_folder": str(mask_path),
                "av_folder": str(av_folder)
            }
        
        except Exception as e:
            self.logger.error(f"Error segmenting {image_path}: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def process_and_save_to_db(self, image_path: str, id: int, extension: str) -> bool:
        """
        Process image and save results to database.
        
        Args:
            image_path: Path to the image file
            qc_result_id: ID of the QC result
            
        Returns:
            True if successful, False otherwise
        """
        result = self.process(image_path)
        
        if not result["success"]:
            self.logger.error(f"Processing failed for {image_path}")
            return False
        
        # Save to database
        success = self.db_manager.save_segmentation_result(
            id=id,
            extension=extension,
            vessel_folder=result["vessel_folder"],
            av_folder=result["av_folder"],
            model_name=self.segmentor.model_name,
            model_version=self.segmentor.model_version
        )
        
        return success
    
    def process_batch(self, image_paths: list, qc_result_ids: list = None) -> Dict[str, Any]:
        """
        Process multiple images.
        
        Args:
            image_paths: List of image file paths
            qc_result_ids: Optional list of QC result IDs (for DB saving)
            
        Returns:
            Dictionary with batch results
        """
        results = {
            "total": len(image_paths),
            "successful": 0,
            "failed": 0,
            "results": []
        }
        
        for idx, image_path in enumerate(image_paths):
            qc_id = qc_result_ids[idx] if qc_result_ids else None
            
            if qc_id:
                success = self.process_and_save_to_db(image_path, qc_id)
            else:
                result = self.process(image_path)
                success = result["success"]
                results["results"].append(result)
            
            if success:
                results["successful"] += 1
            else:
                results["failed"] += 1
            
            self.logger.info(
                f"Batch progress: {idx + 1}/{len(image_paths)} "
                f"({results['successful']} successful, {results['failed']} failed)"
            )
        
        return results
    
    def get_pending_images(self):
        """Get all images that need segmentation"""
        metadata = self.db_manager.get_pending_segmentations()
        return sorted(metadata, key=lambda x: x.name)