@echo off
setlocal enabledelayedexpansion

:: Set the folder path where the WAV files are located
set "folder_path=%~dp0"

:: Check if ffmpeg is installed
where ffmpeg >nul 2>nul
if %errorlevel% neq 0 (
    echo FFmpeg is not installed or not in the system PATH.
    echo Please install FFmpeg and add it to your system PATH.
    pause
    exit /b
)

:: Loop through all WAV files in the folder
for %%F in ("%folder_path%\*.wav") do (
    set "input_file=%%F"
    set "output_file=%%~dpnF_processed.wav"
    
    :: Get duration of the file
    for /f "delims=" %%a in ('ffprobe -v error -show_entries format^=duration -of default^=noprint_wrappers^=1:nokey^=1 "!input_file!"') do set "duration=%%a"
    
    :: Convert duration to seconds (assuming it's in seconds.milliseconds format)
    for /f "tokens=1 delims=." %%a in ("!duration!") do set "duration_seconds=%%a"
    
    :: Check if duration is greater than 18 minutes (1080 seconds)
    if !duration_seconds! gtr 1080 (
        :: Truncate to 18 minutes and compress
        ffmpeg -i "!input_file!" -t 1080 -acodec pcm_s16le -ar 44100 "!output_file!"
    ) else (
        :: Just compress without truncating
        ffmpeg -i "!input_file!" -acodec pcm_s16le -ar 44100 "!output_file!"
    )
    
    :: If processing was successful, replace the original file with the processed one
    if !errorlevel! equ 0 (
        del "!input_file!"
        ren "!output_file!" "%%~nxF"
        echo Processed: %%~nxF
    ) else (
        echo Failed to process: %%~nxF
        del "!output_file!"
    )
)

echo Processing completed.
pause