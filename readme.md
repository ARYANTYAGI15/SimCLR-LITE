# 🖼️ SimCLR-LITE (Self-Supervised Representation Learning)

A lightweight PyTorch implementation of **SimCLR** for learning visual representations without labels.  
Designed to run on modest hardware (CPU / single GPU, 8GB RAM).

---

## 📂 Project Structure
SimCLR-LITE/
│── notebooks/ # Jupyter notebooks for experiments
│── modules/ # Core code
│ ├── data.py # Dataset & augmentations
│ ├── model.py # Backbone (CNN)
│ ├── projection.py # Projection head
│ ├── contrastive.py # NT-Xent loss
│ ├── train.py # Training loop
│ ├── eval.py # Linear evaluation
│── requirements.txt # Dependencies
│── .gitignore
│── README.md


---

## ⚙️ Installation

```bash
# Clone the repo
git clone https://github.com/<your-username>/SimCLR-LITE.git
cd SimCLR-LITE

# Create virtual environment
python -m venv simclr_env
source simclr_env/bin/activate   # (Linux/Mac)
simclr_env\Scripts\activate      # (Windows)

# Install dependencies
pip install -r requirements.txt
