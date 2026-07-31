@echo off
REM LlamaForge launcher - runs the TUI dashboard (run.py).
REM Press 'q' to quit (stops both servers). Press 'b' to open browser.
setlocal
python "%~dp0run.py" %*
endlocal