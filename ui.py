import streamlit as st
import requests
import base64
import numpy as np
import cv2
import os
import defaults.params as params

API_URL = "http://127.0.0.1:8000/generate"
DEFAULT_IMAGE_PATH = "./defaults/mri_brain.jpg"

st.title("AI Image Generator UI")

if os.path.exists(DEFAULT_IMAGE_PATH):
    default_image = open(DEFAULT_IMAGE_PATH, "rb").read()
else:
    st.error("Default image not found!")
    st.stop()

uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image_data = uploaded_file.read()
    st.image(image_data, caption="Uploaded Image", use_column_width=True)
else:
    image_data = default_image
    st.image(DEFAULT_IMAGE_PATH, caption="Default Image", use_column_width=True)

st.sidebar.header("Parameters")
prompt = st.sidebar.text_input("Prompt", params.prompt, help="The prompt to generate images.")
num_samples = st.sidebar.number_input("Number of Samples", min_value=1, max_value=10, value=params.num_samples, help="Number of images to generate.")
image_resolution = st.sidebar.number_input("Image Resolution", min_value=0, max_value=512, value=params.image_resolution, help="The target resolution for processing the input image.")
strength = st.sidebar.number_input("Strength", min_value=0.0, max_value=2.0, value=params.strength, step=0.1, help="A scaling factor for controlling the influence of the input image in the generation process.")
guess_mode = st.sidebar.checkbox("Guess Mode", params.guess_mode, help="Enable guess mode to use a single strength value for all control scales and disable unconditional prompts.")
low_threshold = st.sidebar.number_input("Low Threshold", min_value=0, max_value=255, value=params.low_threshold, help="The lower threshold for the Canny edge detection algorithm.")
high_threshold = st.sidebar.number_input("High Threshold", min_value=0, max_value=255, value=params.high_threshold, help="The upper threshold for the Canny edge detection algorithm.")
ddim_steps = st.sidebar.number_input("DDIM Steps", min_value=1, max_value=100, value=params.ddim_steps, help="The number of steps to run the DDIM algorithm.")
scale = st.sidebar.number_input("Scale", min_value=1.0, max_value=15.0, value=params.scale, step=0.1, help="A scaling factor for controlling the influence of the unconditional prompts in the generation process.")
seed = st.sidebar.number_input("Seed", value=params.seed, help="The seed value for the random number generator.")
eta = st.sidebar.number_input("Eta", min_value=0.0, max_value=1.0, value=params.eta, step=0.05, help="Hyperparameter of the DDIM algorithm.")
a_prompt = st.sidebar.text_area("A Prompt", params.a_prompt, help="The positive attributes to guide the generation process.")
n_prompt = st.sidebar.text_area("N Prompt", params.n_prompt, help="The negative attributes to guide the generation process.")

if st.button("Generate Images"):
    np_array = np.frombuffer(image_data, np.uint8)
    image_array = cv2.imencode(".png", cv2.imdecode(np_array, cv2.IMREAD_COLOR))[1].tolist()
    
    params = {
        "prompt": prompt,
        "num_samples": num_samples,
        "image_resolution": image_resolution,
        "strength": strength,
        "guess_mode": guess_mode,
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
        "ddim_steps": ddim_steps,
        "scale": scale,
        "seed": seed,
        "eta": eta,
        "a_prompt": a_prompt,
        "n_prompt": n_prompt
    }

    image_payload = {"image_array": image_array}
    
    response = requests.post(API_URL, json={"params": params, "image_data": image_payload})
    
    if response.status_code == 200:
        images = response.json().get("images", [])
        st.subheader("Generated Images")
        for img_data in images:
            img_bytes = base64.b64decode(img_data)
            st.image(img_bytes, caption="Generated Image", use_column_width=True)
    else:
        st.error(f"Error: {response.status_code} - {response.text}")

st.markdown("Image generation takes ~10min with default parameters on the following hardware: Intel i5-1335U, 16GB RAM, NVIDIA MX550.")