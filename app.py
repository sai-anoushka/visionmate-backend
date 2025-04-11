from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import requests
import io
import base64

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Hugging Face Inference API details
API_URL = "https://api-inference.huggingface.co/models/microsoft/git-base-coco"
import os

HF_TOKEN = os.getenv("HF_TOKEN")
HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}"
}


@app.get("/")
def read_root():
    return {"message": "VisionMate API (HF-powered) is running!"}

@app.post("/caption/")
async def generate_caption(file: UploadFile = File(...)):
    print("📥 Received image upload request")

    # Read image and encode it to base64
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()

    print("🖼️ Sending image to Hugging Face Inference API...")
    response = requests.post(API_URL, headers=HEADERS, data=img_bytes)

    try:
        result = response.json()
        caption = result[0]["generated_text"]
        print("📝 Caption:", caption)
        return {"caption": caption}
    except Exception as e:
        print("❌ Error:", e)
        return {"error": "Something went wrong with the inference request."}
