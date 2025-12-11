from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any
import logging

class ProcessingStep(ABC):
    """Abstract base class for all processing steps"""
    
    def __init__(self, step_name: str, step_number: int):
        self.step_name = step_name
        self.step_number = step_number
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def process(self, image_path: Path) -> Dict[str, Any]:
        """
        Process a single image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with results and metadata
        """
        pass
    
    def validate_input(self, image_path: Path) -> bool:
        """Validate that the input image exists and is valid"""
        if not image_path.exists():
            self.logger.error(f"Image not found: {image_path}")
            return False
        return True