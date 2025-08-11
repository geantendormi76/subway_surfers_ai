# --- Configuration ---
# This section defines the paths used in the script.
$ProjectRoot = $PSScriptRoot # Gets the directory where the script is located, which is your project root.
$VideosDir = Join-Path $ProjectRoot "data/raw_videos"
$FramesDir = Join-Path $ProjectRoot "data/frames"

# --- Execution ---
# This section contains the main logic of the script.
Write-Host "--- Starting Batch Video Slicing Task (Smart Scan Mode) ---" -ForegroundColor Green

# [Core Improvement] Use Get-ChildItem to intelligently find all video files that match the naming pattern.
# This makes the script automatically adapt to the number of expert videos you have.
$videoFiles = Get-ChildItem -Path $VideosDir -Filter "gameplay_*_actions.mp4"

# Check if any video files were found.
if ($videoFiles.Count -eq 0) {
    Write-Host "Error: No 'gameplay_*_actions.mp4' files found in $VideosDir." -ForegroundColor Red
    exit
}

Write-Host "Found $($videoFiles.Count) expert videos to process..."

# Loop through each found video file and process it.
foreach ($video in $videoFiles) {
    $video_path = $video.FullName
    Write-Host "`n[Processing $($video.Name)]..." -ForegroundColor Yellow
    
    # Call the Python script as a command-line tool, passing the paths as arguments.
    python scripts/01_video_qiege.py --video $video_path --output_dir $FramesDir
}

Write-Host "`n--- Batch Video Slicing Task Completed! ---" -ForegroundColor Green