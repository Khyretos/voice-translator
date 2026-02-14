#!/usr/bin/env python3
"""
Download and install Argos Translate models
Usage: python download_argos_models.py en es fr de
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

current_directory = os.getcwd()
print("Current Directory:", current_directory)
try:
    import argostranslate.package
    import argostranslate.translate
except ImportError:
    print("ERROR: argostranslate not installed")
    print("Install with: pip install argostranslate")
    sys.exit(1)


def setup_argos_directory():
    """Setup Argos models directory"""
    models_dir = Path("argos_models")
    models_dir.mkdir(exist_ok=True)
    argostranslate.package.settings.package_data_dir = str(models_dir)
    print(f"✓ Argos models directory: {models_dir.absolute()}")
    return models_dir


def update_package_index():
    """Update Argos package index"""
    print("\n📦 Updating package index...")
    try:
        argostranslate.package.settings.package_dir = str(
            os.environ.get("ARGOS_PACKAGE_DIR", "Not found")
        )
        argostranslate.package.update_package_index()
        print("✓ Package index updated")
        return True
    except Exception as e:
        print(f"✗ Failed to update package index: {str(e)}")
        return False


def list_available_packages():
    """List all available Argos packages"""
    try:
        available = argostranslate.package.get_available_packages()

        print("\n📋 Available Language Pairs:")
        print("-" * 50)

        lang_pairs = {}
        for pkg in available:
            key = (pkg.from_code, pkg.to_code)
            lang_pairs[key] = f"{pkg.from_name} → {pkg.to_name}"

        for (from_code, to_code), desc in sorted(lang_pairs.items()):
            print(f"  {from_code} → {to_code}: {desc}")

        return available
    except Exception as e:
        print(f"✗ Error listing packages: {str(e)}")
        return []


def list_installed_packages():
    """List installed Argos packages"""
    try:
        installed = argostranslate.translate.get_installed_languages()
        print(str(installed))

        if not installed:
            print("\n⚠️  No models installed yet")
            return []

        print("\n✓ Installed Language Pairs:")
        print("-" * 50)

        pairs = []
        for source in installed:
            for target in source.translations_from:
                pairs.append(
                    (source.code, target.code, f"{source.name} → {target.name}")
                )

        for from_code, to_code, desc in sorted(pairs):
            print(f"  {from_code} → {to_code}: {desc}")

        return pairs
    except Exception as e:
        print(f"✗ Error listing installed: {str(e)}")
        return []


def install_language_pair(from_lang, to_lang, available_packages):
    """Install a specific language pair"""
    print(f"\n📥 Installing {from_lang} → {to_lang}...")

    # Find matching package
    matching = [
        pkg
        for pkg in available_packages
        if pkg.from_code == from_lang and pkg.to_code == to_lang
    ]

    if not matching:
        print(f"✗ No package found for {from_lang} → {to_lang}")
        return False

    package = matching[0]

    try:
        # Download and install
        download_path = package.download()
        argostranslate.package.install_from_path(download_path)
        print(f"✓ Installed {package.from_name} → {package.to_name}")
        return True
    except Exception as e:
        print(f"✗ Failed to install: {str(e)}")
        return False


def install_bidirectional(lang1, lang2, available_packages):
    """Install both directions of a language pair"""
    print(f"\n🔄 Installing bidirectional: {lang1} ↔ {lang2}")

    success1 = install_language_pair(lang1, lang2, available_packages)
    success2 = install_language_pair(lang2, lang1, available_packages)

    if success1 and success2:
        print(f"✓ Bidirectional installation complete")
        return True
    elif success1 or success2:
        print(f"⚠️  Partial installation (one direction failed)")
        return True
    else:
        print(f"✗ Installation failed")
        return False


def install_common_pairs():
    """Install commonly used language pairs"""
    common_pairs = [
        ("en", "es"),  # English <-> Spanish
        ("en", "fr"),  # English <-> French
        ("en", "de"),  # English <-> German
        ("en", "it"),  # English <-> Italian
        ("en", "pt"),  # English <-> Portuguese
        ("en", "ru"),  # English <-> Russian
        ("en", "zh"),  # English <-> Chinese
        ("en", "ja"),  # English <-> Japanese
        ("en", "ko"),  # English <-> Korean
        ("en", "ar"),  # English <-> Arabic
    ]

    print("\n🌍 Installing common language pairs...")
    print("This may take a while...\n")

    available = argostranslate.package.get_available_packages()

    installed = 0
    failed = 0

    for lang1, lang2 in common_pairs:
        if install_bidirectional(lang1, lang2, available):
            installed += 1
        else:
            failed += 1

    print(f"\n📊 Summary: {installed} pairs installed, {failed} failed")


def main():
    parser = argparse.ArgumentParser(
        description="Download and install Argos Translate models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available models
  python download_argos_models.py --list
  
  # Install English -> Spanish
  python download_argos_models.py en es
  
  # Install bidirectional English <-> French
  python download_argos_models.py en fr --bidirectional
  
  # Install common pairs
  python download_argos_models.py --common
  
  # List installed models
  python download_argos_models.py --installed
        """,
    )

    parser.add_argument(
        "languages", nargs="*", help="Language codes (e.g., en es fr de)"
    )
    parser.add_argument(
        "--list", action="store_true", help="List all available language pairs"
    )
    parser.add_argument(
        "--installed", action="store_true", help="List installed language pairs"
    )
    parser.add_argument(
        "--common",
        action="store_true",
        help="Install common language pairs (en <-> es, fr, de, etc.)",
    )
    parser.add_argument(
        "--bidirectional",
        action="store_true",
        help="Install both directions (e.g., en->es and es->en)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("  Argos Translate Model Manager")
    print("=" * 60)

    # Setup directory
    setup_argos_directory()

    # Update package index
    if not update_package_index():
        sys.exit(1)

    # List available
    if args.list:
        list_available_packages()
        return

    # List installed
    if args.installed:
        list_installed_packages()
        return

    # Install common pairs
    if args.common:
        install_common_pairs()
        list_installed_packages()
        return

    # Install specific languages
    if args.languages:
        available = argostranslate.package.get_available_packages()

        if len(args.languages) < 2:
            print("\n✗ Please specify at least 2 languages (e.g., en es)")
            sys.exit(1)

        if args.bidirectional:
            # Install all pairs bidirectionally
            for i in range(len(args.languages) - 1):
                for j in range(i + 1, len(args.languages)):
                    install_bidirectional(
                        args.languages[i], args.languages[j], available
                    )
        else:
            # Install in sequence: first -> second, second -> third, etc.
            for i in range(len(args.languages) - 1):
                install_language_pair(
                    args.languages[i], args.languages[i + 1], available
                )

        list_installed_packages()
        return

    # No arguments - show help
    parser.print_help()


if __name__ == "__main__":
    main()
