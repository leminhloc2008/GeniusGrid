from flask import Flask, render_template, request, send_file
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
import io
from PIL import Image

app = Flask(__name__)

# Stable Diffusion model setup
authorization_token = "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"  # Add your Hugging Face token here if needed
modelid = "CompVis/stable-diffusion-v1-4"
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionPipeline.from_pretrained(
    modelid,
    revision="fp16",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    use_auth_token=authorization_token
)
pipe.to(device)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        prompt = request.form["prompt"]

        # Generate image using Stable Diffusion
        with autocast(device):
            image = pipe(prompt, guidance_scale=8.5).images[0]

        # Convert PIL Image to bytes
        img_io = io.BytesIO()
        image.save(img_io, 'PNG')
        img_io.seek(0)

        return send_file(img_io, mimetype='image/png')

    return render_template("templates/index.html")


if __name__ == "__main__":
    app.run(debug=True)