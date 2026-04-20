from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uuid
import os
from inference import colorize_image

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.post("/colorize")
async def colorize(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())

    input_path = f"{UPLOAD_DIR}/{file_id}.png"
    output_path = f"{OUTPUT_DIR}/{file_id}.png"

    # Save uploaded image
    with open(input_path, "wb") as f:
        f.write(await file.read())

    # Run model
    output_img = colorize_image(input_path)
    output_img.save(output_path)

    return FileResponse(output_path)