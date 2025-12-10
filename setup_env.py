import os
import sys
import platform
import urllib.request
import zipfile
import shutil
import subprocess
import glob

# --- CONFIGURATION ---
# URL to the folder/release containing your 3 zip files.
BASE_URL = "https://github.com/Borja21091/Ultralyzer/releases/download/v0.1.0/"

def get_os_key():
    """Returns the OS key used in the zip filename."""
    system = platform.system().lower()
    if system == "windows":
        return "windows"
    elif system == "linux":
        return "ubuntu"
    elif system == "darwin":
        return "macos"
    else:
        print(f"Unsupported OS: {system}")
        sys.exit(1)

def download_file(url, dest):
    print(f"Downloading {url}...")
    try:
        # Note: If your repo is private, you might need to add headers here
        # or use a public link (S3/Google Drive) for the zip files.
        urllib.request.urlretrieve(url, dest)
        print("Download complete.")
    except Exception as e:
        print(f"Failed to download file: {e}")
        sys.exit(1)

def extract_zip(zip_path, extract_to):
    print(f"Extracting {zip_path}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("Extraction complete.")
    except Exception as e:
        print(f"Failed to extract zip: {e}")
        sys.exit(1)

def find_compatible_wheel(directory):
    """Finds a wheel file compatible with the current Python version AND architecture."""
    
    # 1. Detect Python Version Tag (e.g., cp310, cp311)
    py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    
    # 2. Detect System Info
    system = platform.system().lower()
    machine = platform.machine().lower()
    
    print(f"Searching for wheel compatible with: Python {py_tag} on {system} ({machine})")
    
    # Search recursively for .whl files
    whl_files = glob.glob(os.path.join(directory, "**", "*.whl"), recursive=True)
    
    if not whl_files:
        print("No .whl files found in the extracted directory.")
        return None

    for wheel in whl_files:
        filename = os.path.basename(wheel)
        
        # --- Check 1: Python Version ---
        # We look for the specific tag (e.g. cp311) OR abi3 (compatible with multiple)
        if py_tag not in filename and "abi3" not in filename:
            continue
            
        # --- Check 2: Platform/Architecture ---
        if system == "windows":
            # Windows is usually win_amd64 (64-bit) or win32 (32-bit)
            if "amd64" in machine.lower() or "x86_64" in machine.lower():
                if "win_amd64" in filename:
                    return wheel
            elif "x86" in machine.lower():
                if "win32" in filename:
                    return wheel
                    
        elif system == "linux":
            # Linux wheels usually have 'manylinux' and 'x86_64' or 'aarch64'
            if "x86_64" in machine and ("manylinux" in filename and "x86_64" in filename):
                return wheel
            elif "aarch64" in machine and ("manylinux" in filename and "aarch64" in filename):
                return wheel
                
        elif system == "darwin":
            # Mac: Check for Apple Silicon (arm64) vs Intel (x86_64)
            # Note: 'universal2' wheels work on both
            if "universal2" in filename:
                return wheel
                
            if "arm64" in machine:
                if "arm64" in filename:
                    return wheel
            else:
                if "x86_64" in filename:
                    return wheel

    print(f"No compatible wheel found for {py_tag} on {system} ({machine}).")
    print(f"Available wheels: {[os.path.basename(f) for f in whl_files]}")
    return None

def install_wheel(wheel_path):
    print(f"Installing {wheel_path}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", wheel_path])
        print("Wheel installed successfully.")
    except subprocess.CalledProcessError:
        print("Failed to install wheel.")
        sys.exit(1)

def install_requirements():
    req_file = "requirements.txt"
    if os.path.exists(req_file):
        print(f"Installing dependencies from {req_file}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_file])
            print("Requirements installed successfully.")
        except subprocess.CalledProcessError:
            print("Failed to install requirements.")
            sys.exit(1)
    else:
        print(f"{req_file} not found. Skipping.")

def main():
    # 1. Determine OS and URL
    os_key = get_os_key()
    zip_filename = f"wheels-{os_key}-latest.zip"
    url = f"{BASE_URL}{zip_filename}"
    
    # Setup temporary directory
    temp_dir = "wheels"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    zip_path = os.path.join(temp_dir, zip_filename)
    
    try:
        # 2. Download
        download_file(url, zip_path)
        
        # 3. Extract
        extract_zip(zip_path, temp_dir)
        
        # 4. Find and Install Wheel
        wheel_path = find_compatible_wheel(temp_dir)
        if wheel_path:
            install_wheel(wheel_path)
        else:
            print("Could not find a suitable wheel to install.")
            sys.exit(1)
            
        # 5. Install requirements
        install_requirements()
        
    finally:
        # 6. Clean up
        if os.path.exists(temp_dir):
            print("Cleaning up temporary files...")
            shutil.rmtree(temp_dir)
            
    print("Setup completed successfully!")

if __name__ == "__main__":
    main()