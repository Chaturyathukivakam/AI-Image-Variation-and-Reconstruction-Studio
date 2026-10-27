# Image Transformation Explorer

Streamlit app that applies a pretrained image-to-image model (Hugging Face Diffusers) to an uploaded image and compares the result with the original.

## Features
- Upload an image (JPG / PNG / WEBP)
- Models: **SD-Turbo img2img** (strength control) and **InstructPix2Pix** (text instruction, image/text guidance)
- Adjustable strength, guidance, steps, seed, image size
- Original vs transformed side by side
- **MSE**, PSNR and a per-pixel difference map

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```
A GPU is recommended. On CPU use SD-Turbo with a small image size (256-384 px).

## Run in Google Colab
Set runtime to GPU, install the requirements, write `app.py`, then launch Streamlit and expose port 8501 with a cloudflared tunnel.

## Metric
MSE = mean((original - transformed)^2) over all pixels and channels (0-255 scale). Lower means the output is closer to the original.
