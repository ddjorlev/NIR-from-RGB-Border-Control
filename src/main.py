"""
Simple script to start training the PID model
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from src.train import main

if __name__ == "__main__":
    print("=" * 80)
    print("Physics-Informed Diffusion Model Training")
    print("RGB-to-NIR Generation for Border Control")
    print("=" * 80)
    
    main()