from sqlalchemy import create_engine, Column, String, DateTime, Integer, Enum, ForeignKey, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from definitions import DB_DIR, IMAGE_FORMATS
from datetime import datetime
from pathlib import Path
from typing import Any
import datetime as dt
import enum
import os

Base = declarative_base()

class QCDecisionEnum(str, enum.Enum):
    """Enum for QC decisions"""
    PASS = "pass"
    BORDERLINE = "borderline"
    REJECT = "reject"


class MetaData(Base):
    """Metadata of images and processing"""
    __tablename__ = "metadata"
    
    id = Column(Integer, primary_key=True)
    extension = Column(String, nullable=False)
    name = Column(String, unique=True, nullable=False)
    folder = Column(String, nullable=False)


class QCResult(Base):
    """Quality Control result for an image"""
    __tablename__ = "QC"

    id = Column(Integer, ForeignKey("metadata.id"), primary_key=True, unique=True)
    name = Column(String, ForeignKey("metadata.name"), unique=True, nullable=False)
    decision = Column(Enum(QCDecisionEnum), nullable=False)
    notes = Column(String, default="")
    
    timestamp = Column(DateTime, default=datetime.now(dt.timezone.utc))
    
    # Relationship
    meta = relationship("MetaData", foreign_keys=[id])

    def __repr__(self):
        return f"<QCResult(name='{self.name}', decision='{self.decision}')>"


class SegmentationResult(Base):
    """Segmentation result for an image"""
    __tablename__ = "segmentation"

    id = Column(Integer, ForeignKey("metadata.id"), primary_key=True)
    extension = Column(String, nullable=False)
    name = Column(String, ForeignKey("metadata.name"),
                  unique=True, nullable=False)
    
    # Mask path
    seg_folder = Column(String, nullable=False)
    
    # Metadata
    model_name = Column(String, default="dummy")
    model_version = Column(String, default="1.0")
    
    timestamp = Column(DateTime, default=datetime.now(dt.timezone.utc))

    # Relationship
    meta = relationship("MetaData", foreign_keys=[id])
    
    def __repr__(self):
        return f"<SegmentationResult(name={self.name + self.extension}, seg_folder={self.seg_folder})>"
    

class MetricsResult(Base):
    """Metrics result for an image"""
    __tablename__ = "metrics"

    id = Column(Integer, ForeignKey("metadata.id"), primary_key=True)
    name = Column(String, ForeignKey("metadata.name"),
                  unique=True, nullable=False)
    
    # Metrics fields
    # GENERAL
    laterality = Column(String, nullable=True)
    # OPTIC DISC
    disc_center_x = Column(Float, nullable=True)
    disc_center_y = Column(Float, nullable=True)
    disc_diameter_px = Column(Float, nullable=True)
    disc_diameter_um = Column(Float, nullable=True)
    disc_area_px = Column(Float, nullable=True)
    disc_area_um = Column(Float, nullable=True)
    disc_major_axis_px = Column(Float, nullable=True)
    disc_major_axis_um = Column(Float, nullable=True)
    disc_minor_axis_px = Column(Float, nullable=True)
    disc_minor_axis_um = Column(Float, nullable=True)
    disc_orientation_deg = Column(Float, nullable=True)
    disc_circularity = Column(Float, nullable=True)
    disc_eccentricity = Column(Float, nullable=True)
    # FOVEA
    fovea_center_x = Column(Float, nullable=True)
    fovea_center_y = Column(Float, nullable=True)
    # OPTIC DISC - FOVEA RELATIONSHIP
    disc_fovea_distance_px = Column(Float, nullable=True)
    disc_fovea_distance_um = Column(Float, nullable=True)
    disc_fovea_angle_deg = Column(Float, nullable=True)
    # VESSELS
    vessel_density = Column(Float, nullable=True)
    vessel_tortuosity_density = Column(Float, nullable=True)
    vessel_tortuosity_fft = Column(Float, nullable=True)
    vessel_fractal_dimension_sandbox = Column(Float, nullable=True)
    vessel_fractal_dimension_boxcount = Column(Float, nullable=True)
    vessel_width_px = Column(Float, nullable=True)
    vessel_width_um = Column(Float, nullable=True)
    vessel_width_gradient = Column(Float, nullable=True)
    vessel_width_intercept_px = Column(Float, nullable=True)
    # ARTERIES
    crae = Column(Float, nullable=True)
    a_density = Column(Float, nullable=True)
    a_tortuosity_density = Column(Float, nullable=True)
    a_tortuosity_fft = Column(Float, nullable=True)
    a_fractal_dimension_sandbox = Column(Float, nullable=True)
    a_fractal_dimension_boxcount = Column(Float, nullable=True)
    a_width_px = Column(Float, nullable=True)
    a_width_um = Column(Float, nullable=True)
    a_width_gradient = Column(Float, nullable=True)
    a_width_intercept_px = Column(Float, nullable=True)
    a_groups = Column(Float, nullable=True)
    a_branching_points = Column(Float, nullable=True)
    a_branches = Column(Float, nullable=True)
    # VEINS
    crve = Column(Float, nullable=True)
    v_density = Column(Float, nullable=True)
    v_tortuosity_density = Column(Float, nullable=True)
    v_tortuosity_fft = Column(Float, nullable=True)
    v_fractal_dimension_sandbox = Column(Float, nullable=True)
    v_fractal_dimension_boxcount = Column(Float, nullable=True)
    v_width_px = Column(Float, nullable=True)
    v_width_um = Column(Float, nullable=True)
    v_width_gradient = Column(Float, nullable=True)
    v_width_intercept_px = Column(Float, nullable=True)
    v_groups = Column(Float, nullable=True)
    v_branching_points = Column(Float, nullable=True)
    v_branches = Column(Float, nullable=True)
    # ARTERIES - VEINS RELATIONSHIP
    av_ratio = Column(Float, nullable=True)
    av_crossings = Column(Float, nullable=True)
    av_arcade_concavity = Column(Float, nullable=True)
    
    timestamp = Column(DateTime, default=datetime.now(dt.timezone.utc))

    # Relationship
    meta = relationship("MetaData", foreign_keys=[id])
    
    def __repr__(self):
        return f"<MetricsResult(name={self.name})>"


class DatabaseManager:
    """Manages database connection and operations"""
    
    def __init__(self, db_path: Path = None):
        """
        Initialize database manager.
        
        Args:
            db_path: Path to database file. If None, uses ':memory:'
        """
        if db_path is None:
            self.db_path = Path(os.path.join(DB_DIR, "ultralyzer.db"))
        else:
            self.db_path = Path(db_path)
        
        # Create directory if it doesn't exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create engine
        db_url = f"sqlite:///{self.db_path}"
        self.engine = create_engine(db_url, echo=False)
        
        # Create tables
        Base.metadata.create_all(self.engine)
        
        # Create session factory
        self.SessionLocal = sessionmaker(bind=self.engine)
    
    def get_image_path(self, name: str) -> Path:
        """
        Get full image path from metadata.
        
        Args:
            name: Name of the image file
        Returns:
            Full path to the image file
        """
        session = self.session
        try:
            meta = session.query(MetaData).filter_by(name=name).first()
            if meta:
                return Path(meta.folder) / name
            else:
                return None
        finally:
            session.close()
    
    ############ PROPERTIES ############
    
    @property
    def session(self):
        """Get a new database session"""
        return self.SessionLocal()
    
    ############ METADATA GET METHODS ############
    
    def get_metadata_by_filename(self, name: str) -> MetaData:
        """Get metadata for a specific image by filename"""
        session = self.session
        if name.endswith(tuple(IMAGE_FORMATS)):
            name = name.rsplit('.', 1)[0]
        try:
            meta = session.query(MetaData).filter_by(name=name).first()
            return meta
        finally:
            session.close()
    
    ############ METADATA SET METHODS ############
    
    def save_folder_metadata(self, folder: Path) -> bool:
        """
        Save metadata for all images in a folder.
        
        Args:
            folder: Path to image folder
            
        Returns:
            True if successful, False otherwise
        """
        session = self.session
        try:
            image_files = [
                f for f in folder.iterdir()
                if f.suffix.lower() in IMAGE_FORMATS
            ]
            image_files.sort()
            
            for img_path in image_files:
                name = str(img_path.stem)
                extension = str(img_path.suffix.lower())
                folder_str = str(folder)
                
                # Check if metadata already exists
                existing = session.query(MetaData).filter_by(name=name).first()
                if not existing:
                    meta = MetaData(
                        extension=extension,
                        name=name,
                        folder=folder_str
                    )
                    session.add(meta)
                elif str(existing.folder) != folder_str:
                    existing.folder = folder_str
                    
            session.commit()
            return True
        
        except Exception as e:
            session.rollback()
            print(f"Error saving folder metadata: {str(e)}")
            return False
        
        finally:
            session.close()
    
    ############ QC GET METHODS ############
    
    def get_qc_result(self, name: str) -> QCResult:
        """Get QC result for a specific image"""
        session = self.session
        try:
            result = session.query(QCResult).filter_by(name=name).first()
            return result
        finally:
            session.close()
    
    def get_all_qc_results(self) -> list:
        """Get all QC results"""
        session = self.session
        try:
            results = session.query(QCResult).all()
            return results
        finally:
            session.close()
    
    def get_results_by_decision(self, decision: str) -> list:
        """Get all images with a specific decision"""
        session = self.session
        try:
            results = session.query(QCResult).filter_by(
                decision=QCDecisionEnum(decision)
            ).all()
            return results
        finally:
            session.close()
    
    def get_results_by_folder(self, folder: Path) -> list:
        """Get QC results for images in a specific folder"""
        session = self.session
        try:
            folder_str = str(folder)
            results = session.query(QCResult).filter_by(
                folder=folder_str
            ).all()
            return results
        finally:
            session.close()
    
    def get_statistics(self) -> dict:
        """Get summary statistics"""
        session = self.session
        try:
            total = session.query(QCResult).count()
            pass_count = session.query(QCResult).filter_by(
                decision=QCDecisionEnum.PASS
            ).count()
            borderline_count = session.query(QCResult).filter_by(
                decision=QCDecisionEnum.BORDERLINE
            ).count()
            reject_count = session.query(QCResult).filter_by(
                decision=QCDecisionEnum.REJECT
            ).count()
            
            return {
                "total": total,
                "pass": pass_count,
                "borderline": borderline_count,
                "reject": reject_count
            }
        finally:
            session.close()

    ############ QC SET METHODS ############

    def save_qc_result(self, name: str, decision: str, notes: str = "") -> bool:
        """
        Save or update QC result for an image.
        
        Args:
            name: Name of the image
            decision: Decision (pass, borderline, reject)
            notes: Optional notes about the image
            
        Returns:
            True if successful, False otherwise
        """
        session = self.session
        try:
            # Check if metadata exists
            meta = session.query(MetaData).filter_by(name=name).first()
            if not meta:
                print(f"Error: No metadata found for {name}")
                return False
            
            # Check if QC result exists
            existing = session.query(QCResult).filter_by(name=name).first()

            if existing:
                # Update existing
                existing.decision = QCDecisionEnum(decision)
                existing.notes = notes
                existing.timestamp = datetime.now(dt.timezone.utc)
            else:
                # Create new
                qc_result = QCResult(
                    id=meta.id,
                    name=name,
                    decision=QCDecisionEnum(decision),
                    notes=notes
                )
                session.add(qc_result)
            
            session.commit()
            return True
        
        except Exception as e:
            session.rollback()
            print(f"Error saving QC result: {str(e)}")
            return False
        
        finally:
            session.close()
    
    def delete_qc_result(self, name: str) -> bool:
        """Delete QC result for an image"""
        session = self.session
        try:
            session.query(QCResult).filter_by(name=name).delete()
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            print(f"Error deleting QC result: {str(e)}")
            return False
        finally:
            session.close()
    
    def clear_all_results(self) -> bool:
        """Clear all QC results (careful!)"""
        session = self.session
        try:
            session.query(QCResult).delete()
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            print(f"Error clearing results: {str(e)}")
            return False
        finally:
            session.close()
    
    ############ SEGMENTATION GET METHODS ############
    
    def get_segmentation_mask_path(self, name: str) -> Path:
        """Get segmentation mask path for a specific image"""
        session = self.session
        try:
            result = session.query(SegmentationResult).filter_by(
                name=name
            ).first()
            if result:
                return Path(result.seg_folder.value)
            else:
                return None
        finally:
            session.close()
    
    def get_pending_segmentations(self) -> list:
        """Get QC results that need segmentation (PASS or BORDERLINE without segmentation)"""
        session = self.session
        results = []
        try:
            results = session.query(MetaData).join(
                QCResult, MetaData.id == QCResult.id
            ).filter(
                QCResult.decision.in_([QCDecisionEnum.PASS, QCDecisionEnum.BORDERLINE])
            ).filter(
                ~QCResult.id.in_(
                    session.query(SegmentationResult.id)
                )
            ).all()
            return results
        finally:
            session.close()
            return results
    
    def get_segmentation_by_filename(self, name: str) -> SegmentationResult:
        """Get segmentation result for a specific image"""
        session = self.session
        try:
            result = session.query(SegmentationResult).filter_by(name=name).first()
            return result
        finally:
            session.close()
    
    def get_segmentation_result_by_id(self, id: int):
        """Get segmentation result for a QC result"""
        session = self.session
        try:
            result = session.query(SegmentationResult).filter_by(id=id).first()
            return result
        finally:
            session.close()
    
    def get_all_segmentation_results(self):
        """Get all segmentation results"""
        session = self.session
        try:
            results = session.query(SegmentationResult).all()
            return results
        finally:
            session.close()
    
    ############ SEGMENTATION SET METHODS ############

    def set_mask_info(self, id: int, mask_path: Path, suffix: Path) -> bool:
        """
        Set mask information for an image.
        
        Args:
            id: ID of the image metadata
            mask_path: Path to the mask file
            mask_type: Type of mask ('av' or 'vessel')
        Returns:
            True if successful, False otherwise
        """
        session = self.session
        try:
            meta = session.query(MetaData).filter_by(id=id).first()
            if not meta:
                print(f"Error: No metadata found for ID {id}")
                return False
            
            seg_result = session.query(SegmentationResult).filter_by(id=id).first()
            # Add new entry if not exists
            if not seg_result:
                seg_result = SegmentationResult(
                    id=id,
                    extension=str(suffix).lower(),
                    name=meta.name,
                    seg_folder=""
                )
                session.add(seg_result)
            
            # Update (now) existing entry
            seg_result.seg_folder = str(mask_path)
            
            session.commit()
            return True
        
        except Exception as e:
            session.rollback()
            print(f"Error setting mask info: {str(e)}")
            return False
        
        finally:
            session.close()
    
    def save_segmentation_result(
        self,
        id: int,
        extension: str,
        seg_folder: str,
        model_name: str = "default_model",
        model_version: str = "1.0") -> bool:
        """
        Save segmentation result for an image.
        
        Args:
            qc_result_id: ID of the associated QC result
            seg_folder: Path to segmentation mask folder
            model_name: Name of the segmentation model
            model_version: Version of the segmentation model
            
        Returns:
            True if successful, False otherwise
        """
        session = self.session
        try:
            # Check if segmentation already exists for this QC result
            existing = session.query(SegmentationResult).filter_by(id=id).first()
            
            if existing:
                # Update existing
                existing.seg_folder = seg_folder
                existing.model_name = model_name
                existing.model_version = model_version
                existing.timestamp = datetime.now(dt.timezone.utc)
            else:
                # Create new
                seg_result = SegmentationResult(
                    id=id,
                    extension=extension,
                    name=session.query(QCResult).filter_by(id=id).first().name,
                    seg_folder=seg_folder,
                    model_name=model_name,
                    model_version=model_version
                )
                session.add(seg_result)
            
            session.commit()
            return True
        
        except Exception as e:
            session.rollback()
            print(f"Error saving segmentation result: {str(e)}")
            return False
        
        finally:
            session.close()
            
    ############ METRICS GET METHODS ############
    
    def get_metrics_by_filename(self, name: str) -> MetricsResult | Any:
        """Get metrics result for a specific image"""
        session = self.session
        try:
            result = session.query(MetricsResult).filter_by(name=name).first()
            if result:
                return result
            else:
                return None
        finally:
            session.close()
    
    def get_fovea_by_filename(self, name: str) -> tuple[float, float] | Any:
        """Get fovea metrics for a specific image"""
        session = self.session
        try:
            result = session.query(MetricsResult).filter_by(name=name).first()
            result = (result.fovea_center_x, result.fovea_center_y) if result else (None, None)
            return result
        finally:
            session.close()
    
    def get_laterality_by_filename(self, name: str) -> str | Any:
        """Get laterality metric for a specific image"""
        session = self.session
        try:
            result = session.query(MetricsResult).filter_by(name=name).first()
            if result:
                return result.laterality
            else:
                return None
        finally:
            session.close()
    
    def get_optic_disc_by_filename(self, name: str) -> tuple[float, float, float] | Any:
        """Get optic disc coordinates and diameter for a specific image"""
        session = self.session
        try:
            result = session.query(MetricsResult).filter_by(name=name).first()
            if result:
                return (result.disc_center_x, result.disc_center_y, result.disc_diameter_px)
            else:
                return (None, None, None)
        finally:
            session.close()
    
    def get_pending_metrics(self) -> list:
        """Get QC results that need segmentation (PASS or BORDERLINE)"""
        session = self.session
        results = []
        try:
            results = session.query(MetaData).join(
                QCResult, MetaData.id == QCResult.id
            ).filter(
                QCResult.decision.in_([QCDecisionEnum.PASS, QCDecisionEnum.BORDERLINE])
            ).all()
        finally:
            session.close()
            return results
    
    ############ METRICS SET METHODS ############
    
    def save_metrics_by_id(
        self,
        id: int,
        metrics: dict) -> bool:
        """
        Save metrics result for an image.
        
        Args:
            id: ID of the associated metadata
            metrics: Dictionary of metrics to save
        
        Returns:
            True if successful, False otherwise
        """
        session = self.session
        try:
            # Check if metrics already exists for this metadata
            existing = session.query(MetricsResult).filter_by(id=id).first()
            
            if existing:
                # Update existing
                for key, value in metrics.items():
                    if hasattr(existing, key):
                        setattr(existing, key, value)
                existing.timestamp = datetime.now(dt.timezone.utc)
            else:
                # Create new
                meta = session.query(MetaData).filter_by(id=id).first()
                if not meta:
                    print(f"Error: No metadata found for ID {id}")
                    return False
                
                metrics_result = MetricsResult(
                    id=id,
                    name=meta.name,
                    **metrics
                )
                session.add(metrics_result)
            
            session.commit()
            return True
        
        except Exception as e:
            session.rollback()
            print(f"Error saving metrics result: {str(e)}")
            return False
        
        finally:
            session.close()
    
    def save_metrics_disc_centroid_by_id(
        self,
        id: int,
        disc_x: float,
        disc_y: float) -> bool:
        """
        Save optic disc centroid metrics for an image.
        
        Args:
            id: ID of the associated metadata
            disc_x: Optic disc center x coordinate
            disc_y: Optic disc center y coordinate
        Returns:
            True if successful, False otherwise
        """
        session = self.session
        try:
            # Check if metrics already exists for this metadata
            existing = session.query(MetricsResult).filter_by(id=id).first()
            
            if existing:
                # Update existing
                existing.disc_center_x = disc_x
                existing.disc_center_y = disc_y
                existing.timestamp = datetime.now(dt.timezone.utc)
            else:
                # Create new
                meta = session.query(MetaData).filter_by(id=id).first()
                if not meta:
                    print(f"Error: No metadata found for ID {id}")
                    return False
                
                metrics_result = MetricsResult(
                    id=id,
                    name=meta.name,
                    disc_center_x=disc_x,
                    disc_center_y=disc_y
                )
                session.add(metrics_result)
            
            session.commit()
            return True
        
        except Exception as e:
            session.rollback()
            print(f"Error saving disc centroid metrics: {str(e)}")
            return False
        
        finally:
            session.close()
        
    def save_metrics_fovea_by_id(
        self,
        id: int,
        fovea_x: float,
        fovea_y: float) -> bool:
        """
        Save fovea metrics for an image.
        
        Args:
            id: ID of the associated metadata
            fovea_x: Fovea center x coordinate
            fovea_y: Fovea center y coordinate
        Returns:
            True if successful, False otherwise
        """
        session = self.session
        try:
            # Check if metrics already exists for this metadata
            existing = session.query(MetricsResult).filter_by(id=id).first()
            
            if existing:
                # Update existing
                existing.fovea_center_x = fovea_x
                existing.fovea_center_y = fovea_y
                existing.timestamp = datetime.now(dt.timezone.utc)
            else:
                # Create new
                meta = session.query(MetaData).filter_by(id=id).first()
                if not meta:
                    print(f"Error: No metadata found for ID {id}")
                    return False
                
                metrics_result = MetricsResult(
                    id=id,
                    name=meta.name,
                    fovea_center_x=fovea_x,
                    fovea_center_y=fovea_y
                )
                session.add(metrics_result)
            
            session.commit()
            return True
        
        except Exception as e:
            session.rollback()
            print(f"Error saving fovea metrics: {str(e)}")
            return False
        
        finally:
            session.close()
    
    def save_metrics_fovea_by_name(
        self,
        name: str,
        fovea_x: float,
        fovea_y: float) -> bool:
        """
        Save fovea metrics for an image by name.
        
        Args:
            name: Name of the image
            fovea_x: Fovea center x coordinate
            fovea_y: Fovea center y coordinate
        Returns:
            True if successful, False otherwise
        """
        session = self.session
        try:
            # Check if metrics already exists for this metadata
            existing = session.query(MetricsResult).filter_by(name=name).first()
            
            if existing:
                # Update existing
                existing.fovea_center_x = fovea_x
                existing.fovea_center_y = fovea_y
                existing.timestamp = datetime.now(dt.timezone.utc)
            else:
                # Create new
                meta = session.query(MetaData).filter_by(name=name).first()
                if not meta:
                    print(f"Error: No metadata found for name {name}")
                    return False
                
                metrics_result = MetricsResult(
                    id=meta.id,
                    name=name,
                    fovea_center_x=fovea_x,
                    fovea_center_y=fovea_y
                )
                session.add(metrics_result)
            
            session.commit()
            return True
        
        except Exception as e:
            session.rollback()
            print(f"Error saving fovea metrics: {str(e)}")
            return False
        
        finally:
            session.close()
    
    def save_metrics_laterality_by_id(
        self,
        id: int,
        laterality: str) -> bool:
        """
        Save laterality metric for an image.
        
        Args:
            id: ID of the associated metadata
            laterality: Laterality value ('left' or 'right')
        Returns:
            True if successful, False otherwise
        """
        # Check if laterality is valid
        if laterality not in ['left', 'right']:
            print(f"Error: Invalid laterality value '{laterality}' for ID {id}")
            return False
        
        session = self.session
        try:
            # Check if metrics already exists for this metadata
            existing = session.query(MetricsResult).filter_by(id=id).first()
            
            if existing:
                # Update existing
                existing.laterality = laterality
                existing.timestamp = datetime.now(dt.timezone.utc)
            else:
                # Create new
                meta = session.query(MetaData).filter_by(id=id).first()
                if not meta:
                    print(f"Error: No metadata found for ID {id}")
                    return False
                
                metrics_result = MetricsResult(
                    id=id,
                    name=meta.name,
                    laterality=laterality
                )
                session.add(metrics_result)
            
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            print(f"Error saving laterality metric: {str(e)}")
            return False
        finally:
            session.close()
    
    