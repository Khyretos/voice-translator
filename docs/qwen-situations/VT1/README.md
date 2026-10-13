# VT1: setup.bat pip section

Job: `jobs.json` job vt-v3 (patch `setup.bat`), check `v3.sh`; 3 check attempts, passed. Input `input-setup.bat`, wrong `wrong-setup.bat`, corrected `corrected-setup.bat`.

Mistakes (OVMS Coder):
1. It put `pip install moonshine-voice`, `tokenizers` and `websockets` unconditionally after `pip install -r requirements.txt`, so the following `if errorlevel 1` tested websockets, not requirements.txt. The spec said these go into the fallback list.
2. It left out `huggingface_hub` and `websockets`/`moonshine-voice`/`tokenizers` from the fallback list.
3. It dropped the `echo Installing core dependencies...` line.

Fix: Claude moved the installs into the fallback list and restored the echo. Lesson: a needle check cannot see ordering relative to `if errorlevel`; check that the lines sit between the `if errorlevel 1 (` and the PIP_FAILED test, and name the exact line that must stay the first command after `pip install -r`.
