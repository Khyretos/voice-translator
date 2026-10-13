#!/bin/bash
python3 /tmp/kk-nbwin-chk/needles.py /home/khyretos/Docker/Personal-projects/kk-wt-nbwin/vt/setup.bat '+VT_FORCE_PYTHON' '+exit /b 1' || exit 1
python3 - /home/khyretos/Docker/Personal-projects/kk-wt-nbwin/vt/setup.bat <<'P'
import sys
t=open(sys.argv[1]).read()
a=t.index('REM Warn about Python 3.13'); b=t.index('REM Create virtual environment')
if 'VT_FORCE_PYTHON' not in t[a:b] or 'exit /b 1' not in t[a:b]:
    print('error: setup.bat: the VT_FORCE_PYTHON gate with exit /b 1 must sit in the 3.13+ warning block'); sys.exit(1)
P
