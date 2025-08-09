# --- Configuration ---
$ProjectRoot = $PSScriptRoot
$VideosDir = Join-Path $ProjectRoot "data/raw_videos"
$FramesDir = Join-Path $ProjectRoot "data/frames"

# --- Execution ---
Write-Host "--- Starting Batch Video Slicing Task (Smart Scan Mode) ---" -ForegroundColor Green

# Find all expert videos matching the pattern
$videoFiles = Get-ChildItem -Path $VideosDir -Filter "gameplay_*_actions.mp4"

if ($videoFiles.Count -eq 0) {
    Write-Host "Error: No 'gameplay_*_actions.mp4' files found in $VideosDir." -ForegroundColor Red
    exit
}

Write-Host "Found $($videoFiles.Count) expert videos to process..."

# Loop through each found video file
foreach ($video in $videoFiles) {
    $video_path = $video.FullName
    Write-Host "`n[Processing $($video.Name)]..." -ForegroundColor Yellow
    # Call the Python tool
    python scripts/01_video_qiege.py --video $video_path --output_dir $FramesDir
}

Write-Host "`n--- Batch Video Slicing Task Completed! ---" -ForegroundColor Green