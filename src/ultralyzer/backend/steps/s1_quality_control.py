from pathlib import Path
from typing import Dict, Any
from backend.steps.base_step import ProcessingStep
from backend.models.database import DatabaseManager

class QualityControlStep(ProcessingStep):
    """Step 1: Manual quality control and image classification"""
    
    def __init__(self, db_manager: DatabaseManager = None):
        super().__init__("Quality Control", 1)
        self.db_manager = db_manager or DatabaseManager()
    
    def process(self, image_path: Path) -> Dict[str, Any]:
        """
        Validate that image can be loaded (basic check).
        Actual QC decision will come from user interaction.
        """
        if not self.validate_input(image_path):
            return {"success": False, "error": "Invalid input"}
        
        try:
            from PIL import Image
            img = Image.open(image_path)
            img.verify()
            
            # Check if already has a decision
            existing = self.db_manager.get_qc_result(str(image_path))
            
            self.logger.info(f"Image validated: {image_path.name}")
            
            return {
                "success": True,
                "input_path": str(image_path),
                "status": "pending" if not existing else "reviewed",
                "decision": existing.decision.value if existing else None,
                "notes": existing.notes if existing else ""
            }
        except Exception as e:
            self.logger.error(f"Error validating {image_path}: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def save_decision(self, image_path: Path, decision: str, notes: str = "") -> bool:
        """Save user's QC decision to database"""
        valid_decisions = ["pass", "borderline", "reject"]
        if decision not in valid_decisions:
            self.logger.error(f"Invalid decision: {decision}")
            return False
        
        success = self.db_manager.save_qc_result(
            str(image_path),
            decision,
            notes
        )
        
        if success:
            self.logger.info(f"QC decision saved for {image_path.name}: {decision}")
        else:
            self.logger.error(f"Failed to save QC decision for {image_path.name}")
        
        return success
    
    def get_statistics(self) -> dict:
        """Get QC statistics"""
        return self.db_manager.get_statistics()
    
    @staticmethod
    def _get_timestamp() -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()