
# Powershell environment
In Windows Powershell I had to set an environment variable so Python could decode UTF-8 when
running the find-skills skill.
`[Environment]::SetEnvironmentVariable("PYTHONUTF8", "1", "User")`

