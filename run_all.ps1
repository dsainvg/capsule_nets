# run_all.ps1
# Automates training for both MNIST and CIFAR-10 and generates reconstruction plots.

# Ensure output directory exists
if (-not (Test-Path "out")) {
    New-Item -ItemType Directory -Force -Path "out"
}

# Set Python Path to include current directory for module imports
$env:PYTHONPATH = "."

# Use the virtual environment Python
$py = ".\venv\Scripts\python.exe"

Write-Host "--- Starting MNIST Training (1 Epoch) ---" -ForegroundColor Cyan
& $py -m src.train --dataset mnist --epochs 1 --batch_size 128 --out_dir out

Write-Host "`n--- Starting CIFAR-10 Training (1 Epoch) ---" -ForegroundColor Cyan
& $py -m src.train --dataset cifar10 --epochs 1 --batch_size 64 --out_dir out

Write-Host "`nDone! Check the 'out/' directory for reconstruction plots." -ForegroundColor Green
