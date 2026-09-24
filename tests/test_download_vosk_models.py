"""
Tests for download_vosk_models.py — the dynamic-catalog rewrite.

Covers the two real bugs this replaced (models saved to "models/" instead
of "vosk_models/"; a hardcoded, quickly-stale model list) plus the
obsolete-string-vs-boolean footgun in the live catalog's own JSON shape.
"""

import json
import shutil
import tempfile
import zipfile
from pathlib import Path

import pytest

from conftest import ROOT

import importlib.util

spec = importlib.util.spec_from_file_location(
    "download_vosk_models", ROOT / "download_vosk_models.py"
)
dvm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dvm)


# A small, realistic slice of the real catalog shape (fetched live from
# https://alphacephei.com/vosk/models/model-list.json while building this)
# — including the "obsolete" field as the *string* "true"/"false" it
# actually is, not a JSON boolean.
SAMPLE_CATALOG = [
    {
        "lang": "en-us", "lang_text": "US English", "md5": "x",
        "name": "vosk-model-small-en-us-0.15", "obsolete": "false",
        "size": 41205931, "size_text": "39.3MiB", "type": "small",
        "url": "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
        "version": "0.15",
    },
    {
        "lang": "en-us", "lang_text": "US English", "md5": "x",
        "name": "vosk-model-small-en-us-0.3", "obsolete": "true",
        "size": 37279669, "size_text": "35.6MiB", "type": "small",
        "url": "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.3.zip",
        "version": "0.3",
    },
    {
        "lang": "en-us", "lang_text": "US English", "md5": "x",
        "name": "vosk-model-en-us-0.22", "obsolete": "false",
        "size": 1913365522, "size_text": "1.8GiB", "type": "big",
        "url": "https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip",
        "version": "0.22",
    },
    {
        "lang": "es", "lang_text": "Spanish", "md5": "x",
        "name": "vosk-model-small-es-0.42", "obsolete": "false",
        "size": 39817833, "size_text": "38.0MiB", "type": "small",
        "url": "https://alphacephei.com/vosk/models/vosk-model-small-es-0.42.zip",
        "version": "0.42",
    },
]


class TestModelsDirIsCorrect:
    def test_models_dir_matches_what_the_app_expects(self):
        # The original bug: this was Path("models"), but
        # voice_translator.py's get_available_models() only ever looks in
        # "vosk_models" — every download had to be moved manually.
        assert dvm.MODELS_DIR == Path("vosk_models")


class TestIsObsolete:
    def test_string_false_is_not_obsolete(self):
        # The catalog's "obsolete" field is the string "false", which is
        # truthy in Python — a naive `if entry["obsolete"]` check would
        # treat every model as obsolete.
        assert dvm.is_obsolete({"obsolete": "false"}) is False

    def test_string_true_is_obsolete(self):
        assert dvm.is_obsolete({"obsolete": "true"}) is True

    def test_missing_field_defaults_to_not_obsolete(self):
        assert dvm.is_obsolete({}) is False

    def test_case_insensitive(self):
        assert dvm.is_obsolete({"obsolete": "TRUE"}) is True


class TestFindModel:
    def test_finds_exact_name(self):
        entry = dvm.find_model("vosk-model-small-en-us-0.15", SAMPLE_CATALOG)
        assert entry is not None
        assert entry["lang"] == "en-us"

    def test_returns_none_for_unknown_name(self):
        assert dvm.find_model("does-not-exist", SAMPLE_CATALOG) is None


class TestPrintCatalog:
    def test_hides_obsolete_by_default(self, capsys):
        dvm.print_catalog(SAMPLE_CATALOG, None, show_obsolete=False)
        out = capsys.readouterr().out
        assert "vosk-model-small-en-us-0.15" in out
        assert "vosk-model-small-en-us-0.3" not in out

    def test_shows_obsolete_when_requested(self, capsys):
        dvm.print_catalog(SAMPLE_CATALOG, None, show_obsolete=True)
        out = capsys.readouterr().out
        assert "vosk-model-small-en-us-0.3" in out
        assert "[obsolete]" in out

    def test_lang_filter(self, capsys):
        dvm.print_catalog(SAMPLE_CATALOG, "es", show_obsolete=False)
        out = capsys.readouterr().out
        assert "vosk-model-small-es-0.42" in out
        assert "en-us" not in out

    def test_unknown_lang_filter_lists_known_langs(self, capsys):
        dvm.print_catalog(SAMPLE_CATALOG, "xx-nonexistent", show_obsolete=False)
        out = capsys.readouterr().out
        assert "No matching models" in out
        assert "en-us" in out  # suggests known codes


class TestDownloadModelExtraction:
    """End-to-end: build a fake model zip, download (network mocked),
    verify it lands exactly where get_available_models() looks."""

    def test_extracts_to_models_dir_with_matching_name(self, monkeypatch):
        workdir = Path(tempfile.mkdtemp())
        try:
            models_dir = workdir / "vosk_models"
            models_dir.mkdir()

            fake_zip = workdir / "vosk-model-small-xx-0.1.zip"
            with zipfile.ZipFile(fake_zip, "w") as zf:
                zf.writestr("vosk-model-small-xx-0.1/am/final.mdl", "fake")
                zf.writestr("vosk-model-small-xx-0.1/conf/mfcc.conf", "fake")

            entry = {
                "name": "vosk-model-small-xx-0.1",
                "url": "https://example.com/vosk-model-small-xx-0.1.zip",
                "size_text": "1 KB",
                "lang_text": "Test",
                "type": "small",
            }

            def fake_urlretrieve(url, dest, reporthook=None):
                shutil.copy(fake_zip, dest)

            monkeypatch.setattr(dvm.urllib.request, "urlretrieve", fake_urlretrieve)

            ok = dvm.download_model(entry, models_dir)
            assert ok is True

            result_dir = models_dir / "vosk-model-small-xx-0.1"
            assert result_dir.is_dir()
            assert (result_dir / "am" / "final.mdl").exists()
            # temp download dir must be cleaned up, not left as a stray
            # "model" for get_available_models() to trip over
            assert not (models_dir / "temp").exists()
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    def test_already_downloaded_model_is_skipped(self, monkeypatch, capsys):
        workdir = Path(tempfile.mkdtemp())
        try:
            models_dir = workdir / "vosk_models"
            (models_dir / "vosk-model-small-xx-0.1" / "am").mkdir(parents=True)

            def fail_urlretrieve(*a, **k):
                raise AssertionError("should not attempt to download an existing model")

            monkeypatch.setattr(dvm.urllib.request, "urlretrieve", fail_urlretrieve)

            entry = {"name": "vosk-model-small-xx-0.1", "url": "https://example.com/x.zip"}
            ok = dvm.download_model(entry, models_dir)
            assert ok is True
            assert "already exists" in capsys.readouterr().out
        finally:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
