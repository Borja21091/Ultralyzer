import sys
import argparse
from pathlib import Path

from PySide6.QtWidgets import QApplication
from frontend.main_window import MainWindow


def main():
    """Main entry point with step selection"""
    parser = argparse.ArgumentParser(
        description="Ultralyzer - Retinal Image Processing Pipeline"
    )
    parser.add_argument(
        "--step", "-s",
        type=str,
        default="qc",
        choices=["qc", "seg"],
        help="Processing step to run (default: qc)"
    )
    parser.add_argument(
        "--folder", "-f",
        type=str,
        default=None,
        help="Initial image folder to load"
    )
    
    args = parser.parse_args()
    
    # Create application
    app = QApplication(sys.argv)
    
    # Create main window with selected step
    window = MainWindow(step=args.step)
    
    # Load folder if provided
    if args.folder:
        folder_path = Path(args.folder)
        if folder_path.exists():
            window.image_folder = folder_path
            window.folder_label.setText(f"📂 {folder_path.name}")
            
            # Load images if QC step
            if args.step == "qc":
                if window.step_widget.load_images(folder_path):
                    window.statusBar().showMessage(
                        f"Loaded {len(window.step_widget.image_list)} images"
                    )
    
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()