# VT2: follow-up round (python gate, models folder, progress, model prompt)

Job: `jobs2.json` (vt-w1, vt-w2, vt-w4), check `w1.sh`. Inputs `input-setup.bat`; wrong `wrong-setup.bat`, `wrong-download_vosk_models.py`; corrected `corrected-*`.

Mistakes (OVMS Coder, all checks passed first try):
1. vt-w1 put the `VT_FORCE_PYTHON` gate AFTER the closing `)` of the 3.13+ warning block, so setup.bat would have exited for every Python version. The first version of the check only looked for the strings; the check was too weak (it also let the fixed version through, because it only tested the block range, not the nesting).
2. vt-w4 kept a redundant `if block_num == 0` branch and left a whitespace-only line (black/lint).

The model prompt (VT_DOWNLOAD_MODEL, `choice /t 30 /d n`, no-console branch) was given to Claude as literal flat goto code in the review thread, so Claude applied it (not a Coder job).

Fix: Claude moved the gate inside the warning block and cleaned the Python. Lesson: a gate meant to run only in a branch needs a check that it is nested in that branch (indent) and a test with a version that must NOT trigger it.
