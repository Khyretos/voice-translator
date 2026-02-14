#!/usr/bin/env python3
"""
Vosk Model Downloader
Downloads and extracts Vosk speech recognition models
"""

import os
import sys
import urllib.request
import zipfile
import tarfile
from pathlib import Path

MODELS = {
    'en-us-small': {
        'name': 'vosk-model-small-en-us-0.15',
        'url': 'https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip',
        'size': '40 MB',
        'description': 'Lightweight English (US) model'
    },
    'en-us': {
        'name': 'vosk-model-en-us-0.22',
        'url': 'https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip',
        'size': '1.8 GB',
        'description': 'Full English (US) model - high accuracy'
    },
    'es': {
        'name': 'vosk-model-es-0.42',
        'url': 'https://alphacephei.com/vosk/models/vosk-model-es-0.42.zip',
        'size': '1.4 GB',
        'description': 'Spanish model'
    },
    'fr': {
        'name': 'vosk-model-fr-0.22',
        'url': 'https://alphacephei.com/vosk/models/vosk-model-fr-0.22.zip',
        'size': '1.4 GB',
        'description': 'French model'
    },
    'de': {
        'name': 'vosk-model-de-0.21',
        'url': 'https://alphacephei.com/vosk/models/vosk-model-de-0.21.zip',
        'size': '1.9 GB',
        'description': 'German model'
    },
    'cn': {
        'name': 'vosk-model-cn-0.22',
        'url': 'https://alphacephei.com/vosk/models/vosk-model-cn-0.22.zip',
        'size': '1.3 GB',
        'description': 'Chinese model'
    },
    'ru': {
        'name': 'vosk-model-ru-0.42',
        'url': 'https://alphacephei.com/vosk/models/vosk-model-ru-0.42.zip',
        'size': '1.5 GB',
        'description': 'Russian model'
    },
    'it': {
        'name': 'vosk-model-it-0.22',
        'url': 'https://alphacephei.com/vosk/models/vosk-model-it-0.22.zip',
        'size': '1.2 GB',
        'description': 'Italian model'
    },
}

def show_progress(block_num, block_size, total_size):
    """Display download progress"""
    downloaded = block_num * block_size
    percent = min(100, (downloaded / total_size) * 100)
    bar_length = 50
    filled = int(bar_length * percent / 100)
    bar = '█' * filled + '░' * (bar_length - filled)
    
    downloaded_mb = downloaded / (1024 * 1024)
    total_mb = total_size / (1024 * 1024)
    
    sys.stdout.write(f'\r[{bar}] {percent:.1f}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)')
    sys.stdout.flush()

def download_model(model_key: str, models_dir: Path):
    """Download and extract a Vosk model"""
    if model_key not in MODELS:
        print(f"❌ Unknown model: {model_key}")
        print(f"Available models: {', '.join(MODELS.keys())}")
        return False
    
    model = MODELS[model_key]
    model_path = models_dir / model['name']
    
    if model_path.exists():
        print(f"✅ Model already exists: {model['name']}")
        return True
    
    print(f"📥 Downloading {model['name']} ({model['size']})")
    print(f"   {model['description']}")
    
    # Create temp directory
    temp_dir = models_dir / 'temp'
    temp_dir.mkdir(exist_ok=True)
    
    # Download
    archive_name = model['url'].split('/')[-1]
    archive_path = temp_dir / archive_name
    
    try:
        print(f"   URL: {model['url']}")
        urllib.request.urlretrieve(model['url'], archive_path, show_progress)
        print()  # New line after progress bar
        
        print(f"📦 Extracting...")
        
        # Extract
        if archive_name.endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(models_dir)
        elif archive_name.endswith('.tar.gz') or archive_name.endswith('.tgz'):
            with tarfile.open(archive_path, 'r:gz') as tar_ref:
                tar_ref.extractall(models_dir)
        
        # Clean up
        archive_path.unlink()
        
        print(f"✅ Successfully installed {model['name']}")
        return True
        
    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")
        if archive_path.exists():
            archive_path.unlink()
        return False
    finally:
        # Clean up temp directory
        if temp_dir.exists() and not any(temp_dir.iterdir()):
            temp_dir.rmdir()

def list_models():
    """List available models"""
    print("\n📋 Available Vosk Models:\n")
    for key, model in MODELS.items():
        print(f"  {key:15} - {model['description']} ({model['size']})")
    print()

def main():
    """Main function"""
    print("🎤 Vosk Model Downloader\n")
    
    # Create models directory
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    if len(sys.argv) < 2:
        list_models()
        print("Usage:")
        print(f"  python {sys.argv[0]} <model-key>")
        print(f"  python {sys.argv[0]} all          # Download all models")
        print(f"\nExample:")
        print(f"  python {sys.argv[0]} en-us-small")
        return
    
    model_keys = sys.argv[1:]
    
    if 'all' in model_keys:
        model_keys = list(MODELS.keys())
    
    success_count = 0
    for model_key in model_keys:
        if download_model(model_key, models_dir):
            success_count += 1
        print()
    
    print(f"\n✨ Downloaded {success_count}/{len(model_keys)} models successfully")
    print(f"📁 Models directory: {models_dir.absolute()}")

if __name__ == '__main__':
    main()
