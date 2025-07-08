# FLUX.1-Kontext Multi-Image

A high-performance multi-image composition tool powered by FLUX.1-Kontext with GGUF quantization support for efficient memory usage.

## 🌟 Features

- **Multi-Image Input**: Upload and combine multiple images into unified compositions
- **GGUF Quantization**: Memory-efficient quantized models for various GPU configurations
- **Intelligent Memory Management**: Automatic fallbacks and optimizations for stable generation
- **Real-time Processing**: Fast image generation with optimized pipelines
- **Flexible Model Selection**: Choose between different quantization levels based on your hardware

## 🚀 Quick Start

### One-Click Deployment

```bash
# Download and run the deployment script
wget https://raw.githubusercontent.com/nexusjuan12/FLUX.1-Kontext-multi-image/main/deploy_flux_simple.sh
chmod +x deploy_flux_simple.sh
./deploy_flux_simple.sh
```

### Manual Installation

#### Prerequisites
- NVIDIA GPU with 6GB+ VRAM (recommended)
- CUDA 12.1+ installed
- Python 3.10+
- Git

#### Step-by-Step Setup

1. **Install Miniconda** (if not already installed)
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda
export PATH="~/miniconda/bin:$PATH"
conda init bash
source ~/.bashrc
```

2. **Create Environment**
```bash
conda create -n flux-kontext python=3.10 -y
conda activate flux-kontext
```

3. **Clone Repository**
```bash
git clone https://github.com/nexusjuan12/FLUX.1-Kontext-multi-image.git
cd FLUX.1-Kontext-multi-image
```

4. **Install Dependencies**
```bash
# Install PyTorch with CUDA
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install core packages
conda install -c conda-forge numpy pillow requests opencv -y

# Install Python packages
pip install diffusers>=0.30.0 transformers>=4.44.0 accelerate>=0.33.0 \
    sentencepiece>=0.2.0 gradio>=4.0.0 huggingface_hub>=0.24.0 \
    safetensors>=0.4.0 protobuf>=3.19.0 spaces scipy matplotlib tqdm

# Install latest diffusers with GGUF support
pip install git+https://github.com/huggingface/diffusers.git
```

5. **Set up HuggingFace Token**
```bash
# Get token from https://huggingface.co/settings/tokens
# Accept license at https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev
export HF_TOKEN="your_token_here"
echo 'export HF_TOKEN="your_token_here"' >> ~/.bashrc
```

6. **Run the Application**
```bash
python gguf_flux_app.py
```

## 🎯 GGUF Model Selection

The app automatically selects the best quantized model for your GPU, but you can customize it by editing the `SELECTED_MODEL` variable in `gguf_flux_app.py`:

| Model | VRAM Usage | Quality | Recommended For |
|-------|------------|---------|-----------------|
| `Q2_K` | ~4.0GB | Lower | 6-8GB GPUs |
| `Q3_K_M` | ~5.4GB | Decent | 8-12GB GPUs |
| `Q4_K_S` | ~6.8GB | Good | 12-16GB GPUs |
| `Q5_K_M` | ~8.4GB | Very Good | 16-20GB GPUs |
| `Q6_K` | ~9.8GB | Excellent | 20-24GB GPUs |
| `Q8_0` | ~12.7GB | Highest | 24GB+ GPUs |

### Example: Change Model Selection
```python
# In gguf_flux_app.py, modify this line:
SELECTED_MODEL = "Q8_0"  # For 24GB+ GPUs
```

## 🖥️ Hardware Requirements

### Minimum Requirements
- **GPU**: 6GB VRAM (GTX 1060 6GB, RTX 2060)
- **RAM**: 16GB system RAM
- **Storage**: 20GB free space

### Recommended Requirements
- **GPU**: 16GB+ VRAM (RTX 4070 Ti, RTX 4080, RTX 4090)
- **RAM**: 32GB system RAM
- **Storage**: 50GB free space (SSD preferred)

### GPU-Specific Recommendations
- **RTX 4090 (24GB)**: Use `Q8_0` for best quality
- **RTX 4080 (16GB)**: Use `Q5_K_M` or `Q6_K`
- **RTX 4070 Ti (12GB)**: Use `Q4_K_S` or `Q5_K_M`
- **RTX 3060 (12GB)**: Use `Q4_K_S`
- **GTX 1660/2060 (6-8GB)**: Use `Q2_K` or `Q3_K_M`

## 🎨 Usage Guide

### Basic Workflow

1. **Launch the Application**
   ```bash
   conda activate flux-kontext
   cd FLUX.1-Kontext-multi-image
   python gguf_flux_app.py
   ```

2. **Access the Interface**
   - Open your browser to `http://localhost:7860`
   - Or use the Gradio share link for remote access

3. **Upload Images**
   - Click the gallery area to upload multiple images
   - Supports JPG, PNG, WebP formats
   - Images will be automatically concatenated horizontally

4. **Enter Editing Instructions**
   - Describe what you want to achieve
   - Keep prompts under 60 words to avoid truncation
   - Examples:
     - "Combine these people in a park setting"
     - "Make them all the same size and add sunset lighting"
     - "Remove backgrounds and place in a modern office"

5. **Adjust Settings** (Optional)
   - **Guidance Scale**: 1-10 (higher = more prompt adherence)
   - **Seed**: For reproducible results
   - **Randomize**: For varied outputs

6. **Generate**
   - Click "Run" to start generation
   - First run downloads model weights (~8-13GB)
   - Subsequent runs are much faster

### Pro Tips

- **Image Quality**: Upload high-resolution images for better results
- **Aspect Ratios**: Similar aspect ratios work best for combination
- **Prompt Length**: Keep under 60 words to avoid CLIP tokenizer limits
- **Memory Management**: The app automatically handles OOM by resizing
- **Batch Processing**: Use the "Reuse" button to iterate on results

## 🔧 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```
Solution: Lower the SELECTED_MODEL to a smaller quantization
- Change Q8_0 → Q5_K_M → Q4_K_S → Q3_K_M
```

**2. GGUF Model Download Fails**
```
Solution: Check internet connection and HuggingFace token
- Ensure HF_TOKEN is set correctly
- Try switching to a different GGUF model
```

**3. Slow Generation**
```
Solution: Optimize settings
- Use smaller input images
- Lower guidance scale
- Reduce inference steps if customizable
```

**4. Import Errors**
```bash
# Reinstall dependencies
pip install git+https://github.com/huggingface/diffusers.git --force-reinstall
pip install --upgrade transformers accelerate
```

**5. Environment Issues**
```bash
# Reset environment
conda env remove -n flux-kontext -y
# Then re-run deployment script
```

### Performance Optimization

**For Low VRAM GPUs (6-8GB):**
```python
SELECTED_MODEL = "Q3_K_M"  # Use smaller model
# The app will also automatically resize images
```

**For High VRAM GPUs (20GB+):**
```python
SELECTED_MODEL = "Q8_0"  # Maximum quality
# Can handle larger images and batch sizes
```

## 📁 File Structure

```
FLUX.1-Kontext-multi-image/
├── gguf_flux_app.py          # Main GGUF-optimized application
├── app.py                    # Standard application (fallback)
├── deploy_flux_simple.sh     # One-click deployment script
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── examples/                 # Example images and outputs
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the FLUX.1 [dev] Non-Commercial License. See the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Black Forest Labs](https://blackforestlabs.ai/) for FLUX.1-Kontext
- [QuantStack](https://huggingface.co/QuantStack) for GGUF quantizations
- [Hugging Face](https://huggingface.co/) for diffusers library
- [Gradio](https://gradio.app/) for the web interface

## ⭐ Star History

If this project helped you, please consider giving it a star! ⭐

---

**Made with ❤️ for the AI community**
