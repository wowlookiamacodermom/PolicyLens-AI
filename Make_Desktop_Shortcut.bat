@echo off
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$s=(New-Object -ComObject WScript.Shell).CreateShortcut([Environment]::GetFolderPath('Desktop')+'\PolicyLens AI.lnk'); $s.TargetPath='%~dp0PolicyLens.bat'; $s.WorkingDirectory='%~dp0'; $s.IconLocation='%~dp0assets\policylens.ico'; $s.WindowStyle=7; $s.Description='Open PolicyLens AI'; $s.Save()"
if errorlevel 1 (
  echo Could not create the shortcut. You can still open PolicyLens by double-clicking PolicyLens.bat.
) else (
  echo Done. A "PolicyLens AI" shortcut with its own icon is now on your Desktop.
)
pause
