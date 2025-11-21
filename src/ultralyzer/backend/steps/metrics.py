
from skimage.measure import regionprops, label
from definitions import METRIC_DICTIONARY
from typing import Dict, Any
from pathlib import Path
from PIL import Image
import numpy as np
import logging
import ctypes

from backend.models.database import DatabaseManager
from backend.steps.base_step import ProcessingStep


class MetricsStep(ProcessingStep):
    
    def __init__(self, 
                 db_manager: DatabaseManager = None,
                 micron_mex: Path = None):
        
        super().__init__("Metrics Calculation", 3)
        self.db_manager = db_manager or DatabaseManager()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize masks
        self.a_mask = None  # Artery mask
        self.v_mask = None  # Vein mask
        self.od_mask = None # Optic disc mask
        self.vessel_mask = None # Vessel mask
        
        # MEX file for pixel to micron conversion
        self.mex_flag = micron_mex and micron_mex.is_file()
    
    def process(self, image_path: str, extension: str = ".png") -> Dict[str, Any]:
        """
        Calculate metrics for a single image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with metrics results
        """
        if not self.validate_input(Path(image_path)):
            return {"success": False, "error": "Invalid input"}
        
        try:
            # Image segmentation metadata
            name = Path(image_path).stem
            seg_metadata = self.db_manager.get_segmentation_by_filename(name)
            
            # Return if no segmentation data found
            if not seg_metadata:
                self.logger.error(f"No segmentation data found for {name}. Cannot calculate metrics.")
                return {"success": False, "error": "No segmentation data"}
            
            # Load segmentation mask
            seg_folder = Path(seg_metadata.seg_folder)
            extension = seg_metadata.extension
            seg_mask_path = seg_folder / Path(name + extension)
            
            if not seg_mask_path.is_file():
                self.logger.error(f"Segmentation mask file not found: {seg_mask_path}")
                return {"success": False, "error": "Segmentation mask file not found"}
            
            # Prepare output dictionary
            metrics = {key: float('nan') for key in METRIC_DICTIONARY.keys()}
            
            # Load segmentation masks
            mask = np.array(Image.open(seg_mask_path))
            self.a_mask = mask[:, :, 0] > 0  # Artery mask (red channel)
            self.od_mask = mask[:, :, 1] > 0  # Optic disc mask (green channel)
            self.v_mask = mask[:, :, 2] > 0  # Vein mask (blue channel)
            self.vessel_mask = self.a_mask | self.v_mask # Vessel mask
            
            # Calculate Optic Disc metrics
            if not self.od_mask.any():
                self.logger.warning(f"No optic disc detected in {name}. Skipping OD metrics.")
            else:
                od_indices = np.argwhere(self.od_mask)
                od_center_y, od_center_x = np.mean(od_indices, axis=0)
                metrics["disc_center_x"] = float(od_center_x)
                metrics["disc_center_y"] = float(od_center_y)
                metrics["disc_area_px"] = float(np.sum(self.od_mask))
                metrics["disc_diameter_px"] = float(2 * np.sqrt(metrics["disc_area_px"] / np.pi))
                
                # Circularity and Eccentricity
                labeled_od = label(self.od_mask)
                props = regionprops(labeled_od)[0]
                metrics["disc_eccentricity"] = float(props.eccentricity)
                metrics["disc_circularity"] = float((4 * np.pi * props.area) / (props.perimeter ** 2)) * float((1 - 0.5 / ((props.perimeter / (2*np.pi)) + 0.5))**2) if props.perimeter > 0 else 0.0

                # Conversion to microns
                if self.mex_flag:
                    # TODO: Call MEX function to convert pixels to microns
                    pass
                else:
                    self.logger.warning("Micron conversion MEX file not provided or not found. Skipping optic disc micron metrics.")
            
            # Get fovea center from database
            fovea_center_x, fovea_center_y = self.db_manager.get_fovea_by_filename(name)
            
            # Optic Disc - Fovea metrics
            if fovea_center_x and fovea_center_y:
                metrics["fovea_center_x"] = float(fovea_center_x)
                metrics["fovea_center_y"] = float(fovea_center_y)
                if (not np.isnan(metrics["disc_center_x"]) or metrics["disc_center_x"] is not None) and (not np.isnan(metrics["disc_center_y"]) or metrics["disc_center_y"] is not None):
                    od_fovea_distance = np.sqrt(
                        (metrics["disc_center_x"] - fovea_center_x) ** 2 +
                        (metrics["disc_center_y"] - fovea_center_y) ** 2
                    )
                    metrics["od_fovea_distance_px"] = float(od_fovea_distance)
                    metrics["od_fovea_angle_deg"] = float(np.degrees(np.arctan(
                        (fovea_center_y - metrics["disc_center_y"]) /
                        (fovea_center_x - metrics["disc_center_x"] + 1e-6)
                    )))
                    # Conversion to microns
                    if self.mex_flag:
                        pass
                    else:
                        self.logger.warning("Micron conversion MEX file not provided or not found. Skipping OD-Fovea micron metrics.")
                        
            # Vessel / Artery / Vein metrics
            masks = [self.vessel_mask, self.a_mask, self.v_mask]
            for mask, prefix in zip(masks, ["vessel", "a", "v"]):
                if not mask.any():
                    type = "vessels" if prefix == "vessel" else ("arteries" if prefix == "a" else "veins")
                    self.logger.warning(f"No {type} detected in {name}. Skipping {type} metrics.")
                    continue
                metrics[f"{prefix}_density"] = float(np.sum(mask) / mask.size)
            
            return {"success": True, "metrics": metrics}
        
        except Exception as e:
            self.logger.error(f"Error calculating metrics for {Path(image_path).name}: {str(e)}")
            return {"success": False, "error": str(e)}
        
    def process_and_save_to_db(self, image_path: str, id: int) -> bool:
        """
        Process image and save results to database.
        
        Args:
            image_path: Path to the image file
            id: ID of the image file in the database
        Returns:
            True if successful, False otherwise
        """
        seg_meta = self.db_manager.get_segmentation_result_by_id(id)
        if not seg_meta:
            self.logger.error(f"No segmentation data found for ID {id}. Cannot calculate metrics.")
            return False
        extension = seg_meta.extension
        result = self.process(image_path, extension)
        
        if not result["success"]:
            self.logger.error(f"Processing failed for {image_path}")
            return False
        
        metrics = result["metrics"]
        
        # Save metrics to database
        save_success = self.db_manager.save_metrics_by_id(id, metrics)
        
        if not save_success:
            self.logger.error(f"Failed to save metrics to database for {image_path}")
            return False
        
        return True