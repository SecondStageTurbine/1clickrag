@echo off
REM SPDX-License-Identifier: MPL-2.0
REM
REM Double-click this to start the RAG. It asks which folder of documents to
REM make searchable the first time, then remembers the answer.
REM
REM Why a .bat at all: a .ps1 cannot be launched by double-click, and on a
REM managed machine an unsigned script extracted from a zip is refused outright
REM ("running scripts is disabled" / "not digitally signed"). -ExecutionPolicy
REM Bypass applies to this one invocation only - it changes nothing on the
REM machine and needs no admin rights.
REM
REM Arguments pass straight through:  Rag.bat status   Rag.bat down

setlocal
cd /d "%~dp0"

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0rag-up.ps1" %*
set EXITCODE=%ERRORLEVEL%

REM Keep the window open when launched by double-click, so an error is
REM readable instead of vanishing with the console.
if not "%EXITCODE%"=="0" (
  echo.
  echo [exit code %EXITCODE%]
)
echo.
pause
exit /b %EXITCODE%
