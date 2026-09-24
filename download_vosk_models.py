#!/usr/bin/env python3
"""
Vosk Model Downloader

Downloads and extracts Vosk speech recognition models — from the live,
official catalog at https://alphacephei.com/vosk/models/model-list.json,
not a hardcoded list. Whatever Vosk publishes today is what you can
download today; no waiting on this script to be updated for a new model.

Usage:
    python download_vosk_models.py --list                  # every current language
    python download_vosk_models.py --list --lang en-us      # models for one language
    python download_vosk_models.py vosk-model-small-en-us-0.15
    python download_vosk_models.py vosk-model-small-en-us-0.15 vosk-model-small-fr-0.22
    python download_vosk_models.py --lang es --small        # convenience: download the
                                                              # current (non-obsolete) small
                                                              # model for a language
"""

import argparse
import json
import sys
import tarfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

MODEL_LIST_URL = "https://alphacephei.com/vosk/models/model-list.json"

# This is where voice_translator.py's get_available_models() actually looks
# — it used to be "models" here, which meant every download had to be
# manually moved into vosk_models/ afterward. Fixed to match.
MODELS_DIR = Path("vosk_models")


def fetch_catalog() -> list[dict]:
    """
    Fetch the live model catalog. Raises on failure — callers should give
    the user a clear, actionable error rather than silently falling back
    to a stale hardcoded list (which is exactly the problem this replaces).
    """
    try:
        with urllib.request.urlopen(MODEL_LIST_URL, timeout=15) as resp:
            return json.load(resp)
    except (urllib.error.URLError, TimeoutError) as e:
        print(f"❌ Could not reach {MODEL_LIST_URL}: {e}")
        print("   Check your internet connection, or that this host isn't blocked.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Vosk's model catalog didn't parse as JSON ({e}) — it may have")
        print("   changed format. Check https://alphacephei.com/vosk/models/ directly.")
        sys.exit(1)


def is_obsolete(entry: dict) -> bool:
    # The catalog's "obsolete" field is the *string* "true"/"false", not a
    # JSON boolean — a naive `if entry["obsolete"]:` check would treat the
    # string "false" as truthy and hide every non-obsolete model.
    return str(entry.get("obsolete", "false")).strip().lower() == "true"


def show_progress(block_num, block_size, total_size):
    """Display download progress"""
    downloaded = block_num * block_size
    percent = min(100, (downloaded / total_size) * 100) if total_size > 0 else 0
    bar_length = 50
    filled = int(bar_length * percent / 100)
    bar = "█" * filled + "░" * (bar_length - filled)

    downloaded_mb = downloaded / (1024 * 1024)
    total_mb = total_size / (1024 * 1024)

    sys.stdout.write(
        f"\r[{bar}] {percent:.1f}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)"
    )
    sys.stdout.flush()


def find_model(name: str, catalog: list[dict]) -> dict | None:
    for entry in catalog:
        if entry["name"] == name:
            return entry
    return None


def download_model(entry: dict, models_dir: Path) -> bool:
    """Download and extract one model, given its catalog entry."""
    model_path = models_dir / entry["name"]

    if model_path.exists():
        print(f"✅ Model already exists: {entry['name']}")
        return True

    print(f"📥 Downloading {entry['name']} ({entry.get('size_text', '?')})")
    print(f"   {entry.get('lang_text', entry.get('lang', ''))} — {entry.get('type', '')}")

    temp_dir = models_dir / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    url = entry["url"]
    archive_name = url.split("/")[-1]
    archive_path = temp_dir / archive_name

    try:
        print(f"   URL: {url}")
        urllib.request.urlretrieve(url, archive_path, show_progress)
        print()  # newline after progress bar

        print("📦 Extracting...")
        if archive_name.endswith(".zip"):
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(models_dir)
        elif archive_name.endswith((".tar.gz", ".tgz")):
            with tarfile.open(archive_path, "r:gz") as tf:
                tf.extractall(models_dir)
        else:
            print(f"❌ Unrecognized archive type: {archive_name}")
            return False

        archive_path.unlink()

        if not model_path.exists():
            # The archive didn't extract to the folder name we expected —
            # rare, but better to say so clearly than to silently report
            # success while get_available_models() still can't find it.
            print(
                f"⚠️  Extracted, but {model_path} doesn't exist — check what "
                f"folder name the archive actually produced under {models_dir}/"
            )
            return False

        print(f"✅ Successfully installed {entry['name']} -> {model_path}")
        return True

    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")
        if archive_path.exists():
            archive_path.unlink()
        return False
    finally:
        if temp_dir.exists() and not any(temp_dir.iterdir()):
            temp_dir.rmdir()


def print_catalog(catalog: list[dict], lang_filter: str | None, show_obsolete: bool):
    entries = [e for e in catalog if show_obsolete or not is_obsolete(e)]
    if lang_filter:
        entries = [e for e in entries if e["lang"] == lang_filter]

    if not entries:
        print("No matching models.")
        if lang_filter:
            langs = sorted({e["lang"] for e in catalog if not is_obsolete(e)})
            print(f"Known language codes: {', '.join(langs)}")
        return

    by_lang: dict[str, list[dict]] = {}
    for e in entries:
        by_lang.setdefault(e["lang"], []).append(e)

    print(f"\n📋 {len(entries)} model(s){' (including obsolete)' if show_obsolete else ''}:\n")
    for lang in sorted(by_lang):
        lang_text = by_lang[lang][0].get("lang_text", lang)
        print(f"{lang_text} ({lang})")
        for e in sorted(by_lang[lang], key=lambda x: x["name"]):
            flag = " [obsolete]" if is_obsolete(e) else ""
            print(f"    {e['name']:45} {e.get('size_text', '?'):>10}  [{e['type']}]{flag}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Download and install Vosk speech recognition models "
        "from the live official catalog.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_vosk_models.py --list
  python download_vosk_models.py --list --lang en-us
  python download_vosk_models.py vosk-model-small-en-us-0.15
  python download_vosk_models.py vosk-model-small-en-us-0.15 vosk-model-small-fr-0.22
  python download_vosk_models.py --lang es --small
        """,
    )
    parser.add_argument(
        "names", nargs="*", help="Exact model name(s) from --list, e.g. vosk-model-small-en-us-0.15"
    )
    parser.add_argument("--list", action="store_true", help="List available models and exit")
    parser.add_argument("--lang", help="Filter by language code (e.g. en-us, es, fr)")
    parser.add_argument(
        "--all", action="store_true", help="Include obsolete models in --list"
    )
    parser.add_argument(
        "--small",
        action="store_true",
        help="With --lang: download the current (non-obsolete) small model "
        "for that language, without needing to know its exact name",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Vosk Model Downloader")
    print("=" * 60)

    MODELS_DIR.mkdir(exist_ok=True)
    print(f"📁 Models directory: {MODELS_DIR.absolute()}\n")

    print("Fetching current model catalog...")
    catalog = fetch_catalog()
    print(f"✓ {len(catalog)} models in the catalog\n")

    if args.list:
        print_catalog(catalog, args.lang, args.all)
        return

    if args.small:
        if not args.lang:
            print("❌ --small requires --lang <code>, e.g. --lang es --small")
            sys.exit(1)
        candidates = [
            e
            for e in catalog
            if e["lang"] == args.lang and e["type"] == "small" and not is_obsolete(e)
        ]
        if not candidates:
            print(f"❌ No current small model for language '{args.lang}'.")
            print_catalog(catalog, args.lang, False)
            sys.exit(1)
        entry = sorted(candidates, key=lambda e: e["version"])[-1]
        success = download_model(entry, MODELS_DIR)
        sys.exit(0 if success else 1)

    if not args.names:
        parser.print_help()
        print()
        print_catalog(catalog, None, False)
        return

    success_count = 0
    for name in args.names:
        entry = find_model(name, catalog)
        if entry is None:
            print(f"❌ Unknown model name: {name}")
            print("   Run with --list to see exact names.")
            continue
        if download_model(entry, MODELS_DIR):
            success_count += 1
        print()

    print(f"✨ Downloaded {success_count}/{len(args.names)} model(s) successfully")
    print(f"📁 Models directory: {MODELS_DIR.absolute()}")


if __name__ == "__main__":
    main()
