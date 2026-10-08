# utils/train_subprocess.py
import sys
import os

# Tambahin folder utils ke sys.path supaya Python bisa nemuin training.py & config.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training import train_model  # import tetap seperti di training.py

if __name__ == "__main__":
    train_model()
