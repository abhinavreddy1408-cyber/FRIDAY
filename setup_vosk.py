import logging
import os
import shutil
import subprocess
import sys
import zipfile

# Configure logging for setup
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("VoskSetup")

MODELS_DIR = "models"
MODEL_NAME = "vosk-model-small-en-us"
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_NAME)
MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
ZIP_FILE = "vosk_model.zip"

def install_vosk():
    """Install vosk if not present."""
    try:
        import vosk  # noqa: F401
        logger.info("Vosk is already installed.")
    except ImportError:
        logger.info("Installing Vosk...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "vosk"])
        logger.info("Vosk installed successfully.")

def download_model(url: str, dest: str):
    """Download the model with progress tracking."""
    import requests
    logger.info(f"Downloading model from {url}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(dest, 'wb') as f:
        if total_size == 0:
            f.write(response.content)
        else:
            downloaded = 0
            for data in response.iter_content(chunk_size=4096):
                downloaded += len(data)
                f.write(data)
                done = int(50 * downloaded / total_size)
                sys.stdout.write(f"\r[{'=' * done}{' ' * (50-done)}] {downloaded/1024/1024:.2f}MB / {total_size/1024/1024:.2f}MB")
                sys.stdout.flush()
    sys.stdout.write("\n")
    logger.info("Download complete.")

def setup_model():
    """Check, download, and extract the Vosk model."""
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        logger.info(f"Created {MODELS_DIR} directory.")

    # Check if a model already exists but might be misnamed
    found_misnamed = False
    for item in os.listdir(MODELS_DIR):
        item_path = os.path.join(MODELS_DIR, item)
        if os.path.isdir(item_path) and "vosk-model-small-en-us" in item and item != MODEL_NAME:
            logger.info(f"Found misnamed model folder: {item}. Renaming...")
            shutil.move(item_path, MODEL_PATH)
            found_misnamed = True
            break

    if os.path.exists(MODEL_PATH) and not found_misnamed:
        logger.info(f"Model already exists at {MODEL_PATH}")
    elif not os.path.exists(MODEL_PATH):
        # Need to download
        download_model(MODEL_URL, ZIP_FILE)

        logger.info(f"Extracting {ZIP_FILE}...")
        with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
            zip_ref.extractall(MODELS_DIR)

        # Determine the extracted folder name (it's usually vosk-model-small-en-us-0.15)
        extracted_folder = ""
        for name in zip_ref.namelist():
            if "/" in name:
                extracted_folder = name.split("/")[0]
                break

        extracted_path = os.path.join(MODELS_DIR, extracted_folder)
        if extracted_path != MODEL_PATH:
            if os.path.exists(MODEL_PATH):
                shutil.rmtree(MODEL_PATH)
            shutil.move(extracted_path, MODEL_PATH)
            logger.info(f"Renamed {extracted_folder} to {MODEL_NAME}")

        logger.info("Cleaning up zip file...")
        os.remove(ZIP_FILE)
        logger.info("Cleanup complete.")

def verify_model():
    """Verify that the model can be loaded by Vosk."""
    try:
        import vosk
        logger.info("Verifying model loading...")
        vosk.Model(MODEL_PATH)
        logger.info("Success! Vosk model loaded correctly.")
        return True
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        return False

if __name__ == "__main__":
    logger.info("--- F.R.I.D.A.Y. Vosk Setup ---")
    try:
        install_vosk()
        setup_model()
        if verify_model():
            logger.info("Vosk is fully configured and ready for F.R.I.D.A.Y.")
        else:
            logger.error("Vosk setup failed verification.")
            sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred during setup: {e}")
        sys.exit(1)
