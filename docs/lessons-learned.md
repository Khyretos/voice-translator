# Lessons learned (Coder loop)

| Id | Date | Lesson | Situation |
|---|---|---|---|
| VT1 | 2026-10-13 | A needle check cannot see ordering relative to `if errorlevel`. The Coder put fallback installs before the test, so the test checked the wrong command. Check that the lines sit inside the fallback block, and name the line that stays right after `pip install -r`. | docs/qwen-situations/VT1 |
