' LlamaForge one-click launcher (double-click me).
' Runs the router + dashboard hidden, then opens your browser.
Dim here : here = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)
CreateObject("Wscript.Shell").Run "powershell -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File """ & here & "\run.ps1""", 0, False
