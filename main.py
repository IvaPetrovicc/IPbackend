from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import io
from PIL import Image, ImageDraw
import base64

app = FastAPI()

# Configure CORS to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],  # Replace with your frontend URL if necessary
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the YOLO model
license_plate_detector = YOLO('models/license_plate_detector.pt')

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    try:
        # Read the uploaded image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Run YOLO model to detect license plates with a confidence threshold
        results = license_plate_detector(image, conf=0.5)  # Set confidence threshold to 0.5

        # Process YOLO detections to extract the most confident detection
        detections = []
        max_confidence = 0
        best_detection = None

        for result in results:
            for box in result.boxes:
                confidence = box.conf[0].item()  # Get confidence score
                if confidence > max_confidence:
                    max_confidence = confidence
                    x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
                    best_detection = {
                        "name": "License Plate",
                        "xmin": int(x_min),
                        "ymin": int(y_min),
                        "xmax": int(x_max),
                        "ymax": int(y_max),
                        "confidence": confidence
                    }

        # If a detection is found, draw the bounding box
        if best_detection:
            detections.append(best_detection)
            
            # Draw the bounding box on the image
            draw = ImageDraw.Draw(image)
            draw.rectangle(
                [(best_detection["xmin"], best_detection["ymin"]),
                 (best_detection["xmax"], best_detection["ymax"])],
                outline="red",
                width=3
            )

        # Save the annotated image to a buffer
        buf = io.BytesIO()
        image.save(buf, format='JPEG')
        buf.seek(0)

        # Encode the image to Base64
        encoded_image = base64.b64encode(buf.getvalue()).decode('utf-8')

        # Prepare the response
        response_data = {
            "detections": detections,
            "image": encoded_image  # Send Base64 encoded image as a string
        }

        return JSONResponse(content=response_data)

    except Exception as e:
        print(f"Error processing image: {e}")
        return JSONResponse(content={"message": "Failed to process image"}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
