#!/bin/bash
python3 /tmp/kk-nbwin-chk/needles.py /home/khyretos/Docker/Personal-projects/kk-wt-nbwin/vt/setup.bat '+set "PIP_FAILED=0"' '+pip install moonshine-voice' '+pip install tokenizers' '+pip install websockets' '+pip install webrtcvad-wheels' '+[ERROR] Dependencies could not be installed' '+exit /b 1' '-echo [OK] Dependencies installed
echo.' || exit 1
python3 - /home/khyretos/Docker/Personal-projects/kk-wt-nbwin/vt/setup.bat <<'P'
import sys
t=open(sys.argv[1]).read()
a=t.index('pip install -r requirements.txt'); b=t.index('REM Create directories')
s=t[a:b]
if s.count('[ERROR] Dependencies could not be installed')!=1 or '[OK] Dependencies installed' not in s:
    print('error: setup.bat: the [OK] line must only print when no pip install failed, the [ERROR]+exit /b 1 otherwise'); sys.exit(1)
P
