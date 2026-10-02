from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request

from models.classifier import ImageClassifier


# ============================================================
# APPLICATION CONFIGURATION
# ============================================================

APP_NAME = "VisionPredict API"
APP_VERSION = "1.0.0"

APP_DESCRIPTION = (
    "A professional image classification API "
    "built with FastAPI and TensorFlow."
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title=APP_NAME,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)


# ============================================================
# STATIC FILES
# ============================================================

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)


# ============================================================
# HTML TEMPLATES
# ============================================================

templates = Jinja2Templates(
    directory="templates"
)


# ============================================================
# MACHINE LEARNING MODEL
# ============================================================

classifier = ImageClassifier()


# ============================================================
# HOME PAGE
# ============================================================

@app.get("/")
async def home():

    return FileResponse(
        "templates/index.html"
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
async def health_check():

    return {
        "status": "healthy",
        "service": APP_NAME,
        "version": APP_VERSION
    }


# ============================================================
# API INFORMATION
# ============================================================

@app.get("/api/info")
async def api_info():

    return {
        "name": APP_NAME,
        "version": APP_VERSION,
        "framework": "FastAPI",
        "model": "MobileNetV2",
        "task": "Image Classification"
    }


# ============================================================
# IMAGE PREDICTION
# ============================================================

@app.post("/api/predict")
async def predict_image(
    file: UploadFile = File(...)
):

    # Allowed image formats
    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/webp"
    }

    # Check file type
    if file.content_type not in allowed_types:

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid file type. "
                "Please upload a JPG, PNG, or WEBP image."
            )
        )

    try:

        # Read uploaded file
        image_data = await file.read()

        # Check empty file
        if not image_data:

            raise HTTPException(
                status_code=400,
                detail="The uploaded image is empty."
            )

        # Generate prediction
        prediction = classifier.predict(
            image_data
        )

        # Return prediction
        return {
            "success": True,
            "filename": file.filename,
            "prediction": prediction
        }

    except HTTPException:
        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )


# ============================================================
# APPLICATION START
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )