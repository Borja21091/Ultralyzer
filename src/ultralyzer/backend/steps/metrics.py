
from skimage.morphology import skeletonize
from skimage.measure import regionprops, label
from typing import Dict, Any
from pathlib import Path
import uwf_eye_geometry
from PIL import Image
import numpy as np
import logging

from backend.utils.preprocessing import localise_centre_mass
from backend.models.database import DatabaseManager
from backend.steps.base_step import ProcessingStep
from backend.utils.arcades import ArcadeRANSAC

from backend.utils.feature_measurement import fractal_dimension_boxcount, fractal_dimension_sandbox
from backend.utils.feature_measurement import curve_length, chord_length, tortuosity_density
from backend.utils.feature_measurement import generate_vessel_skeleton, compute_edges
from backend.utils.feature_measurement import calculate_vessel_widths_px, calculate_vessel_widths_mm

from matplotlib import pyplot as plt


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
            metrics = {}
            
            # Load segmentation masks
            mask = np.array(Image.open(seg_mask_path))
            self.a_mask = mask[:, :, 0] > 0  # Artery mask (red channel)
            self.od_mask = mask[:, :, 1] > 0  # Optic disc mask (green channel)
            self.v_mask = mask[:, :, 2] > 0  # Vein mask (blue channel)
            self.vessel_mask = self.a_mask | self.v_mask # Vessel mask
            
            # Calculate Optic Disc metrics
            disc_flag = self.od_mask.any()
            if not disc_flag:
                self.logger.warning(f"No optic disc detected in {name}. Skipping OD metrics.")
            else:
                od_center_y, od_center_x = localise_centre_mass((self.od_mask * 255).astype(np.uint8))
                metrics["disc_center_x"] = float(od_center_x)
                metrics["disc_center_y"] = float(od_center_y)
                metrics["disc_area_px"] = float(np.sum(self.od_mask))
                metrics["disc_diameter_px"] = float(2 * np.sqrt(metrics["disc_area_px"] / np.pi))
                
                # Fitted ellipse properties
                labeled_od = label(self.od_mask)
                props = regionprops(labeled_od)[0]
                metrics["disc_major_axis_px"] = float(props.major_axis_length)
                metrics["disc_minor_axis_px"] = float(props.minor_axis_length)
                metrics["disc_orientation_deg"] = float(props.orientation * (180.0 / np.pi))
                
                # Circularity and Eccentricity
                metrics["disc_eccentricity"] = float(props.eccentricity)
                metrics["disc_circularity"] = float((4 * np.pi * props.area) / (props.perimeter ** 2)) * float((1 - 0.5 / ((props.perimeter / (2*np.pi)) + 0.5))**2) if props.perimeter > 0 else 0.0

                # Conversion to microns
                disc_um_metrics = self._compute_disc_um_metrics(metrics)
                metrics.update(disc_um_metrics)
                
                # Save to DB
                self.db_manager.save_metrics_by_id(seg_metadata.id, metrics)
            
            # Get fovea center from database
            fovea_center_x, fovea_center_y = self.db_manager.get_fovea_by_filename(name) # (col, row) = (x, y)
            fovea_flag = fovea_center_x is not None and fovea_center_y is not None
            
            # Transform (col, row) to Cartesian (x, y)
            fovX = fovea_center_x
            fovY = mask.shape[0] - fovea_center_y
            discX = od_center_x
            discY = mask.shape[0] - od_center_y
            
            # Optic Disc - Fovea metrics
            if fovea_flag:
                metrics["fovea_center_x"] = float(fovea_center_x)
                metrics["fovea_center_y"] = float(fovea_center_y)
                if disc_flag:
                    od_fovea_distance = np.sqrt(
                        (metrics["disc_center_x"] - fovea_center_x) ** 2 +
                        (metrics["disc_center_y"] - fovea_center_y) ** 2
                    )
                    metrics["disc_fovea_distance_px"] = float(od_fovea_distance)
                    metrics["disc_fovea_angle_deg"] = float(np.degrees(np.arctan(
                        (fovY - discY) / np.abs(fovX - discX + 1e-6))))
                    
                    # Conversion to microns
                    metrics["disc_fovea_distance_um"] = float(
                        uwf_eye_geometry.calculate_pair_distances(
                            np.array([[metrics["disc_center_x"], metrics["disc_center_y"]]]).astype(np.float64),
                            np.array([[fovea_center_x, fovea_center_y]]).astype(np.float64)
                        ) * 1e3
                    )
                    
                    # Save to DB
                    self.db_manager.save_metrics_by_id(seg_metadata.id, metrics)
            else:
                self.logger.warning(f"Fovea location not found in database for {name}. Please identify the Fovea using the 'Edit mask' tool and re-run metric calculation. Skipping OD-Fovea metrics for now.")
            
            # Laterality
            if fovea_flag and disc_flag:
                laterality = "right" if fovea_center_x < metrics["disc_center_x"] else "left"
                metrics["laterality"] = laterality
                self.db_manager.save_metrics_by_id(seg_metadata.id, metrics)
            
            # Vessel / Artery / Vein metrics
            masks = [self.vessel_mask, self.a_mask, self.v_mask]
            for mask, prefix in zip(masks, ["vessel", "a", "v"]):
                if not mask.any():
                    type = "vessels" if prefix == "vessel" else ("arteries" if prefix == "a" else "veins")
                    self.logger.warning(f"No {type} detected in {name}. Skipping {type} metrics.")
                    continue
                
                # Density
                metrics[f"{prefix}_density"] = float(np.sum(mask) / mask.size)
                
                # Fractal Dimension (Sandbox and Boxcount)
                metrics[f"{prefix}_fractal_dimension_sandbox"] = float(fractal_dimension_sandbox(mask.astype(int)))
                metrics[f"{prefix}_fractal_dimension_boxcount"] = float(fractal_dimension_boxcount(mask.astype(int)))
                
                # Generate ordered skeleton coordinates
                vcoords = generate_vessel_skeleton(mask.astype(np.uint8), self.od_mask, (od_center_y, od_center_x)) # [(row, col), ...]
                
                # Initialise vessel widths and count lists
                tcurve = 0
                tcc = 0
                td = 0
                vessel_count = 0
                zonal_vessels = []
                for vessel in vcoords:
                    vessel_count += 1
                    zonal_vessels.append(vessel)
                    
                    # Work out length of current vessel
                    vessel = vessel.T
                    v_length = curve_length(vessel[1], vessel[0])
                    c_length = chord_length(vessel[1], vessel[0])
                    tcc += v_length / c_length
                            
                    # tcurve is simply the pixel length of the vessel
                    tcurve += v_length
                    
                    # td measures curve_chord_ratio for subvessel segments per inflection point 
                    # and cumulatively add them, and scale by number of inflections and overall curve length
                    # formula comes from https://ieeexplore.ieee.org/document/1279902
                    td += tortuosity_density(vessel[1], vessel[0], v_length)
                
                # Normalise tortuosity density and tortuosity distance by vessel_count
                metrics[f"{prefix}_tortuosity_density"] = td/vessel_count
                metrics[f"{prefix}_tortuosity_distance"] = tcc/vessel_count
            
                # This is measuring the same thing as average_width computed in global_metrics, but should be smaller as 
                # individual vessel segments exclude branching points in their calculation
                edges1, edges2, centerline_coords = compute_edges(mask.astype(np.uint8), zonal_vessels)
                all_vessel_widths, avg_width = calculate_vessel_widths_px(edges1, edges2)
                all_vessel_widths_mm, avg_width_mm = calculate_vessel_widths_mm(edges1, edges2)
                
                # Average Width
                metrics[f"{prefix}_width_px"] = np.mean(avg_width)
                metrics[f"{prefix}_width_um"] = np.mean(avg_width_mm) * 1e3
                
                # Width Gradient as the slope of the linear fit to vessel width vs distance from OD center
                # dist = [np.sqrt((c[0] - od_center_x) ** 2 + (c[1] - od_center_y) ** 2) for c in coords]
                dist = np.linalg.norm(centerline_coords - np.array([[od_center_y, od_center_x]]), axis=1)
                dist_mm = uwf_eye_geometry.calculate_pair_distances(
                                    centerline_coords[:, ::-1].astype(np.float64),
                                    np.tile(np.array([[od_center_x, od_center_y]]), (centerline_coords.shape[0], 1)).astype(np.float64),
                                )
                
                if len(dist) >= 2:
                    p = np.polyfit(dist, np.concatenate(all_vessel_widths), 1)
                    metrics[f"{prefix}_width_gradient_px"] = float(p[0])
                    metrics[f"{prefix}_width_intercept_px"] = float(p[1])
                    
                if len(dist_mm) >= 2:
                    p_mm = np.polyfit(dist_mm, np.concatenate(all_vessel_widths_mm), 1)
                    metrics[f"{prefix}_width_gradient_um"] = float(p_mm[0]) * 1e3
                    metrics[f"{prefix}_width_intercept_um"] = float(p_mm[1]) * 1e3
                
                # Tortuosity FFT
                
            # Artery/Vein CRAE/CRVE, groups, branching points, branches
            for mask, prefix in zip([self.a_mask, self.v_mask], ["a", "v"]):
                if not mask.any():
                    type = "arteries" if prefix == "a" else "veins"
                    self.logger.warning(f"No {type} detected in {name}. Skipping {type} CRAE/CRVE and branching metrics.")
                    continue
                    
                # CRAE/CRVE
                
                # Groups
                
                # Branching Points
                
                # Branches
                
            # Artery - Vein Relationship metrics
            if not self.a_mask.any() or not self.v_mask.any():
                self.logger.warning(f"Insufficient artery/vein data in {name}. Skipping artery-vein relationship metrics.")
            else:
                # AV Ratio
                metrics["av_ratio"] = metrics["a_density"] / metrics["v_density"] if metrics["v_density"] > 0 else float('nan')
                
                # AV Crossings
                artery_skeleton = skeletonize(self.a_mask)
                vein_skeleton = skeletonize(self.v_mask)
                metrics["av_crossings"] = float(np.sum(artery_skeleton & vein_skeleton))
                
                # AV Arcade Concavity
                arcader = ArcadeRANSAC(name=name, 
                                       mask=self.vessel_mask, 
                                       db_manager=self.db_manager)
                arcader()
                metrics["av_arcade_concavity"] = np.abs(arcader.concavity) if arcader.concavity is not None else float('nan')
            
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
    
    def get_pending_images(self):
        """Get all images that need segmentation"""
        metadata = self.db_manager.get_pending_metrics()
        return sorted(metadata, key=lambda x: x.name)
    
    ################ PRIVATE METHODS ################
    
    def _compute_disc_um_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Compute optic disc metrics in microns"""
        disc_um_metrics = {}
        try:
            
            # Convert center to 3D coordinates
            center_px = np.array([[metrics["disc_center_x"], metrics["disc_center_y"]]])
            radius_px = metrics["disc_diameter_px"] / 2.0
            edge_points_px = np.array([np.array([metrics["disc_center_x"], metrics["disc_center_y"]]) + radius_px * np.array([np.cos(theta), -np.sin(theta)])
                                       for theta in np.linspace(0, 2*np.pi, num=360, endpoint=False)])
            points_3d = uwf_eye_geometry.pixels_to_3d(np.vstack((center_px, edge_points_px)))
            center_3d = points_3d[0, :]
            edge_points_3d = points_3d[1:, :]
            N = edge_points_3d.shape[0]
            
            center_vector = center_3d / np.linalg.norm(center_3d)
            
            # Project edge points directly onto center vector to get distance of base plane from origin
            # projections = R * cos(theta)
            projections = np.dot(edge_points_3d, center_vector)
            
            # Spherical cap height: h = R - R * cos(theta)
            h = np.mean(np.linalg.norm(center_3d) - projections) * 1e3  # in micrometers
            
            R = 12000 # Eye radius in microns
            disc_area_um = 2 * np.pi * R * h  # in micrometers squared
            
            # Geodesic Diameter (Arc length)
            disc_diameter_um = uwf_eye_geometry.calculate_pair_distances(
                edge_points_px[0:N//2 - 1, :], edge_points_px[N//2:N - 1, :]
            ) * 1e3 # in micrometers
            disc_diameter_um = np.mean(disc_diameter_um)
            
            # Convert major/minor axes to microns
            # 1. Get 2D axis endpoints in pixel space
            orientation_rad = np.radians(metrics["disc_orientation_deg"])
            major_axis_coords_px = np.array([
                [metrics["disc_center_x"] - (metrics["disc_major_axis_px"] / 2) * np.sin(orientation_rad),
                 metrics["disc_center_y"] + (metrics["disc_major_axis_px"] / 2) * np.cos(orientation_rad)],
                [metrics["disc_center_x"] + (metrics["disc_major_axis_px"] / 2) * np.sin(orientation_rad),
                 metrics["disc_center_y"] - (metrics["disc_major_axis_px"] / 2) * np.cos(orientation_rad)]
            ])
            minor_axis_coords_px = np.array([
                [metrics["disc_center_x"] - (metrics["disc_minor_axis_px"] / 2) * np.cos(orientation_rad),
                 metrics["disc_center_y"] - (metrics["disc_minor_axis_px"] / 2) * np.sin(orientation_rad)],
                [metrics["disc_center_x"] + (metrics["disc_minor_axis_px"] / 2) * np.cos(orientation_rad), 
                 metrics["disc_center_y"] + (metrics["disc_minor_axis_px"] / 2) * np.sin(orientation_rad)]
            ])
            # 2. Calculate distance in between endpoints in 3D space
            major_axis_um = uwf_eye_geometry.calculate_pair_distances(
                major_axis_coords_px[0,:].astype(np.float64).reshape(1, -1),
                major_axis_coords_px[1,:].astype(np.float64).reshape(1, -1)
            ) * 1e3  # in micrometers
            minor_axis_um = uwf_eye_geometry.calculate_pair_distances(
                minor_axis_coords_px[0,:].astype(np.float64).reshape(1, -1),
                minor_axis_coords_px[1,:].astype(np.float64).reshape(1, -1)
            ) * 1e3  # in micrometers
            
            disc_um_metrics = {
                "disc_area_um": float(disc_area_um),
                "disc_diameter_um": float(disc_diameter_um),
                "disc_major_axis_um": float(major_axis_um),
                "disc_minor_axis_um": float(minor_axis_um),
            }
            
            # Update main metrics dictionary
            metrics.update(disc_um_metrics)
            
        except Exception as e:
            self.logger.error(f"Error converting optic disc metrics to microns: {str(e)}")
            
        return disc_um_metrics
    
    