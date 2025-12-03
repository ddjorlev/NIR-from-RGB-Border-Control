# NIR-from-RGB Border Control Diffusion Model

A state-of-the-art diffusion model for generating Near-Infrared (NIR) thermal images from RGB inputs, specifically designed for border control and surveillance applications using the FLIR ADAS dataset.

## 🌟 Features

- **Advanced Diffusion Architecture**: Custom UNet with attention mechanisms optimized for thermal image generation
- **FLIR ADAS Dataset Integration**: Seamless loading and preprocessing of thermal imaging data
- **Robust Training Pipeline**: Mixed precision training, gradient accumulation, and EMA for stable convergence  
- **Advanced Data Augmentation**: Specialized augmentations for thermal-RGB image pairs including MixUp and CutMix
- **Comprehensive Monitoring**: Weights & Biases integration with sample generation and metric tracking
- **Flexible Configuration**: Dataclass-based configuration system for easy experimentation

## 🏗️ Architecture

The model uses a **U-Net based diffusion architecture** with:
- **Sinusoidal time embeddings** for diffusion timestep conditioning
- **Multi-scale residual blocks** with group normalization and SiLU activations
- **Multi-head self-attention** at multiple resolutions for spatial relationships
- **Skip connections** between encoder and decoder for detail preservation
- **DDIM sampling** for fast inference with configurable steps

## 📁 Project Structure

```
NIR-from-RGB-Border-Control/
├── config.py              # Configuration management
├── src/
│   └── data.py            # Dataset and data loading utilities  
├── dataloader.py          # Advanced data pipeline with augmentations
├── train.py               # Complete training script with diffusion model
├── run_training.py        # Quick start script
├── requirements.txt       # Python dependencies
├── data/                  # FLIR ADAS dataset directory
│   └── FLIR ADAS priv 1000.v1-oryginal.multiclass/
│       ├── train/         # Training images and labels
│       ├── valid/         # Validation images and labels  
│       └── test/          # Test images and labels (optional)
├── checkpoints/           # Model checkpoints (auto-created)
├── logs/                  # Training logs (auto-created)
└── samples/               # Generated sample images (auto-created)
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset

Ensure your FLIR ADAS dataset is structured as:
```
data/FLIR ADAS priv 1000.v1-oryginal.multiclass/
├── train/
│   ├── *.jpg              # Training images
│   └── _classes.csv       # Labels file
├── valid/
│   ├── *.jpg              # Validation images  
│   └── _classes.csv       # Labels file
└── test/ (optional)
```

### 3. Start Training

**Option A: Quick Start**
```bash
python run_training.py
```

**Option B: Direct Training**
```bash
python train.py
```

### 4. Monitor Progress

- **Sample Images**: Check `./samples/` for generated images during training
- **Checkpoints**: Model saves to `./checkpoints/` every 5000 steps
- **Weights & Biases**: View real-time metrics at [wandb.ai](https://wandb.ai)

## ⚙️ Configuration

The model is highly configurable through `config.py`. Key settings:

### Model Architecture
```python
model_channels = 128        # Base channel count
channel_mult = (1,1,2,2,4,4) # Channel multipliers
num_res_blocks = 2          # Residual blocks per level
attention_resolutions = (32,16,8) # Attention at these resolutions
```

### Training Parameters
```python
batch_size = 8              # Adjust based on GPU memory
learning_rate = 1e-4        # Learning rate
num_epochs = 500            # Training epochs
mixed_precision = True      # Use automatic mixed precision
```

### Diffusion Settings
```python
noise_steps = 1000          # Diffusion timesteps
beta_start = 0.0001         # Noise schedule start
beta_end = 0.02             # Noise schedule end
ddim_steps = 50             # Sampling steps for inference
```

## 📊 Dataset Details

The FLIR ADAS dataset contains thermal images with vehicle and pedestrian annotations:
- **Classes**: `car`, `person` (binary labels)
- **Format**: RGB-format thermal images (simulated RGB pairs generated)
- **Augmentations**: Spatial transforms, color jitter, noise injection
- **Preprocessing**: Resize to 256×256, normalization, thermal enhancement

## 🔬 Model Details

### Diffusion Process
1. **Forward Process**: Gradually add Gaussian noise to NIR images over T timesteps
2. **Reverse Process**: Learn to denoise and generate NIR from RGB + noise
3. **Training**: Predict noise added at random timesteps
4. **Sampling**: Generate NIR via iterative denoising from pure noise

### Loss Function
- **Primary**: L2 loss between predicted and actual noise
- **Options**: L1, Huber loss for different noise characteristics
- **Regularization**: Gradient clipping, weight decay, EMA averaging

### Advanced Features
- **Time Conditioning**: Sinusoidal embeddings for timestep information
- **Attention Mechanisms**: Multi-head self-attention for spatial relationships
- **Skip Connections**: U-Net architecture preserves fine details
- **Mixed Precision**: Faster training with maintained accuracy

## 📈 Training Tips

### GPU Memory Optimization
- Reduce `batch_size` if out of memory (try 4 or 2)
- Enable `gradient_checkpointing` for memory savings
- Use `accumulate_grad_batches` for effective larger batch sizes

### Hyperparameter Tuning
- **Learning Rate**: Start with 1e-4, adjust based on loss curves
- **Noise Schedule**: Cosine schedule often works better than linear
- **Sampling Steps**: More steps = better quality but slower inference

### Monitoring Training
- Watch sample images for visual quality improvement
- Monitor loss convergence (should steadily decrease)
- Check attention maps for meaningful feature learning

## 🎯 Use Cases

This model is designed for:
- **Border Security**: Generate thermal signatures from visible images
- **Surveillance Enhancement**: Create NIR views for better night vision
- **Data Augmentation**: Expand thermal datasets using RGB images
- **Cross-Modal Translation**: Research in visible-to-thermal conversion

## 🔬 Research Applications

- Study thermal signature patterns in different weather conditions
- Analyze vehicle vs. pedestrian heat signatures
- Investigate domain adaptation between visible and thermal spectra
- Develop improved night vision and low-light surveillance systems

## 🤝 Contributing

This is a research project. For improvements or extensions:
1. Fork the repository
2. Create a feature branch
3. Implement changes with proper documentation
4. Submit a pull request with detailed description

## 📄 License

This project is intended for academic and research purposes. Please check dataset licensing requirements for commercial use.

## 🙏 Acknowledgments

- **FLIR ADAS Dataset**: Thermal imaging data for automotive applications
- **Hugging Face Diffusers**: Inspiration for diffusion model architecture
- **PyTorch Team**: Deep learning framework and utilities

---

**Happy Training! 🚀**

For questions or issues, please check the logs in `./logs/` and ensure all requirements are properly installed.