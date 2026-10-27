import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageOps

st.set_page_config(page_title="Image Transformation Explorer", layout="wide")

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

MODELS = {
    "SD-Turbo img2img (fast, CPU-friendly)": "stabilityai/sd-turbo",
    "InstructPix2Pix (edit by instruction)": "timbrooks/instruct-pix2pix",
}


@st.cache_resource(show_spinner="Loading model (first run downloads weights)...")
def load_pipe(model_id: str):
    if "instruct-pix2pix" in model_id:
        from diffusers import StableDiffusionInstructPix2PixPipeline
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_id, torch_dtype=DTYPE, safety_checker=None
        )
    else:
        from diffusers import AutoPipelineForImage2Image
        pipe = AutoPipelineForImage2Image.from_pretrained(
            model_id, torch_dtype=DTYPE, variant="fp16" if DTYPE == torch.float16 else None
        )
    pipe = pipe.to(DEVICE)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def prep(img: Image.Image, size: int) -> Image.Image:
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((size, size))
    w, h = img.size
    return img.resize((w - w % 8, h - h % 8))


def mse(a: Image.Image, b: Image.Image) -> float:
    x = np.asarray(a, dtype=np.float32)
    y = np.asarray(b.resize(a.size), dtype=np.float32)
    return float(np.mean((x - y) ** 2))


st.title("Image Transformation Explorer")
st.caption(f"Compute: **{DEVICE.upper()}** | Upload an image, transform it, compare with the original.")

with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox("Model", list(MODELS))
    model_id = MODELS[model_name]
    is_ip2p = "instruct-pix2pix" in model_id
    default_prompt = "make it look like a watercolor painting" if is_ip2p else "a watercolor painting"
    prompt = st.text_input("Prompt / instruction", default_prompt)
    size = st.select_slider("Max image size (px)", [256, 384, 512, 640], value=384 if DEVICE == "cpu" else 512)
    seed = st.number_input("Seed", 0, 2**31 - 1, 42)

    if is_ip2p:
        steps = st.slider("Inference steps", 5, 50, 10)
        img_guid = st.slider("Image guidance (higher = closer to original)", 1.0, 3.0, 1.5, 0.1)
        txt_guid = st.slider("Text guidance", 1.0, 15.0, 7.5, 0.5)
    else:
        strength = st.slider("Strength (0 = original, 1 = fully redrawn)", 0.1, 1.0, 0.5, 0.05)
        steps = st.slider("Inference steps", 2, 8, 4)

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded:
    original = prep(Image.open(uploaded), size)
    if st.button("Transform", type="primary"):
        pipe = load_pipe(model_id)
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        with st.spinner("Generating..."):
            with torch.inference_mode():
                if is_ip2p:
                    out = pipe(prompt, image=original, num_inference_steps=steps,
                               image_guidance_scale=img_guid, guidance_scale=txt_guid,
                               generator=gen).images[0]
                else:
                    steps_eff = max(steps, int(np.ceil(1 / strength)))  # SD-Turbo needs steps*strength >= 1
                    out = pipe(prompt, image=original, strength=strength,
                               num_inference_steps=steps_eff, guidance_scale=0.0,
                               generator=gen).images[0]
        st.session_state["result"] = (original, out.resize(original.size))

    if "result" in st.session_state:
        orig, out = st.session_state["result"]
        c1, c2 = st.columns(2)
        c1.subheader("Original"); c1.image(orig, use_container_width=True)
        c2.subheader("Transformed"); c2.image(out, use_container_width=True)

        err = mse(orig, out)
        psnr = 10 * np.log10(255**2 / err) if err > 0 else float("inf")
        m1, m2 = st.columns(2)
        m1.metric("MSE (0-255 scale, lower = more similar)", f"{err:.2f}")
        m2.metric("PSNR (dB)", f"{psnr:.2f}")

        diff = np.abs(np.asarray(orig, np.float32) - np.asarray(out, np.float32)).mean(axis=2)
        st.subheader("Per-pixel difference")
        st.image((diff / max(diff.max(), 1e-6) * 255).astype(np.uint8), clamp=True, use_container_width=True,
                 caption="Brighter = bigger change (normalized)")
else:
    st.info("Upload an image to begin. (Your zip contains `Photo on 27-10-26 at 2.34 PM.jpg` — upload that one.)")
