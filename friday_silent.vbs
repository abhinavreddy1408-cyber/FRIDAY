Set WinScriptHost = CreateObject("WScript.Shell")
' Define the project directory
ProjectDir = "c:\Users\abhin\Desktop\F.R.I.D.A.Y"
' Set the working directory to the project root
WinScriptHost.CurrentDirectory = ProjectDir
' Run F.R.I.D.A.Y. using pythonw.exe
' 0 = vbHide, False = don't wait
WinScriptHost.Run "pythonw.exe main.py", 0, False
Set WinScriptHost = Nothing
