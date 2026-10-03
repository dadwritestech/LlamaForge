' LlamaForge one-click launcher (double-click me).
' Runs the router + dashboard hidden, then opens your browser.
' run.ps1 exits non-zero when something stops it; say so instead of nothing happening.
Dim here : here = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)
Dim code : code = CreateObject("Wscript.Shell").Run("powershell -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File """ & here & "\run.ps1""", 0, True)
If code <> 0 Then
  MsgBox "LlamaForge did not start." & vbCrLf & vbCrLf & _
         "What happened is in:" & vbCrLf & here & "\logs\launcher.log", _
         vbExclamation, "LlamaForge"
End If
