from sqlalchemy import create_engine, Column, String, DateTime, Integer, Enum, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from definitions import DB_DIR, IMAGE_FORMATS
from datetime import datetime
from pathlib import Path
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
    name = Column(String, unique=True, nullable=False)
    folder = Column(String, nullable=False)

    # Processing steps {False: not done, True: done}
    s1_qc = Column(Boolean, default=False)
    s2_segmentation = Column(Boolean, default=False)
    s3_img_props = Column(Boolean, default=False)
    s4_vessel_props = Column(Boolean, default=False)
    s5_av_correction = Column(Boolean, default=False)
    s6_wg = Column(Boolean, default=False)
    s7_fd = Column(Boolean, default=False)
    s8_choroid = Column(Boolean, default=False)


class QCResult(Base):
    """Quality Control result for an image"""
    __tablename__ = "qc_results"

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
    __tablename__ = "segmentation_results"

    id = Column(Integer, ForeignKey("metadata.id"), primary_key=True)
    name = Column(String, ForeignKey("metadata.name"),
                  unique=True, nullable=False)
    
    # Mask paths
    vessel_folder = Column(String, nullable=False)
    av_folder = Column(String, nullable=False)
    
    # Metadata
    model_name = Column(String, default="dummy")
    model_version = Column(String, default="1.0")
    
    timestamp = Column(DateTime, default=datetime.now(dt.timezone.utc))

    # Relationship
    meta = relationship("MetaData", foreign_keys=[id])
    
    def __repr__(self):
        return f"<SegmentationResult(name={self.name}, av_folder={self.av_folder}, vessel_folder={self.vessel_folder})>"
    

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
            
            for img_path in image_files:
                name = str(img_path.name)
                folder_str = str(folder)
                
                # Check if metadata already exists
                existing = session.query(MetaData).filter_by(name=name).first()
                if not existing:
                    meta = MetaData(
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
    
    ############ PROPERTIES ############
    
    @property
    def session(self):
        """Get a new database session"""
        return self.SessionLocal()
    
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
    
    def delete_qc_result(self, image_path: str) -> bool:
        """Delete QC result for an image"""
        session = self.session
        try:
            session.query(QCResult).filter_by(image_path=image_path).delete()
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
    
    def get_segmentation_mask_path(self, name: str, mask_type: str = "av") -> Path:
        """Get segmentation mask path for a specific image"""
        session = self.session
        try:
            result = session.query(SegmentationResult).filter_by(
                name=name
            ).first()
            if result:
                if mask_type == "av":
                    return Path(result.av_folder)
                elif mask_type == "vessel":
                    return Path(result.vessel_folder)
                else:
                    return None
            else:
                return None
        finally:
            session.close()
    
    def get_pending_segmentations(self) -> list:
        """Get QC results that need segmentation (PASS or BORDERLINE without segmentation)"""
        session = self.session
        results = []
        try:
            results = session.query(QCResult).filter(
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
    
    def get_segmentation_result(self, qc_result_id: int):
        """Get segmentation result for a QC result"""
        session = self.session
        try:
            result = session.query(SegmentationResult).filter_by(
                qc_result_id=qc_result_id
            ).first()
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

    def save_segmentation_result(
        self,
        id: int,
        vessel_folder: str,
        av_folder: str,
        model_name: str = "default_model",
        model_version: str = "1.0"
    ) -> bool:
        """
        Save segmentation result for an image.
        
        Args:
            qc_result_id: ID of the associated QC result
            av_folder: Path to arteries and veins mask
            vessel_folder: Path to vessel mask
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
                existing.av_folder = av_folder
                existing.vessel_folder = vessel_folder
                existing.model_name = model_name
                existing.model_version = model_version
                existing.timestamp = datetime.now(dt.timezone.utc)
            else:
                # Create new
                seg_result = SegmentationResult(
                    id=id,
                    name=session.query(QCResult).filter_by(id=id).first().name,
                    av_folder=av_folder,
                    vessel_folder=vessel_folder,
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
            
    
    