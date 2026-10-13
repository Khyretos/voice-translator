# Lessons learned (Coder loop)

| Id | Date | Lesson | Situation |
|---|---|---|---|
| VT1 | 2026-10-13 | A needle check cannot see ordering relative to `if errorlevel`. The Coder put fallback installs before the test, so the test checked the wrong command. Check that the lines sit inside the fallback block, and name the line that stays right after `pip install -r`. | docs/qwen-situations/VT1 |
| VT2 | 2026-10-13 | The Coder placed a "stop unless VT_FORCE_PYTHON=1" gate after the closing `)` of the 3.13+ warning block, so setup.bat would stop on every Python version. A string check passed. Check the nesting (indent) of any gate and run it with an input that must NOT trigger it. | docs/qwen-situations/VT2 |
