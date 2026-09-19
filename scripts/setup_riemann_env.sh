# Set up a Linux virtual environment for Riemannian Relational GFM training.
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"

cd "${PROJECT_ROOT}" || exit 1

python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

python -c "import torch; import torch_geometric; import geoopt; print(torch.__version__); print(torch_geometric.__version__); print(geoopt.__version__); print(torch.cuda.is_available())"
