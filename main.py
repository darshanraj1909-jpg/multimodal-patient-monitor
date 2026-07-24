# main.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import base64
from PIL import Image
import io
import json
import random # For mocking results
from ultralytics import YOLO

app = FastAPI()

# Mount static files so we can serve index.html
app.mount("/static", StaticFiles(directory="."), name="static")

# ==========================================================
# ### PLACEHOLDER: Load Your Models Here ###
# Example: 
import torch
fire_model = YOLO('fire_smoke_model.pt') 
posture_model = YOLO('posture_detection.pt')
# ==========================================================

class MockModels:
    """A dummy class to simulate models so this script runs immediately."""
    def predict_fire(self, image):
        # Your model will analyze the 'image' (OpenCV matrix)
        states = ["Safe", "Smoke Detected", "Fire! WARNING"]
        # Dummy: Randomly returns a state with weights (mostly safe)
        return random.choices(states, weights=[90, 8, 2])[0]

    def predict_posture(self, image):
        states = ["Sitting", "Lying Down", "Walking", "Fallen! HELP"]
        # Dummy: Randomly returns a state
        return random.choice(states)

# Instantiate the models (or the mock)
my_models = MockModels()

# ==========================================================


def decode_base64_image(base64_string):
    """Decodes a base64 string from browser into OpenCV image format (BGR)"""
    try:
        # Remove metadata header (data:image/jpeg;base64,) if present
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]
            
        img_bytes = base64.b64decode(base64_string)
        img_pill = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        # Convert RGB to BGR for OpenCV
        open_cv_image = np.array(img_pill)
        open_cv_image = open_cv_image[:, :, ::-1].copy() 
        return open_cv_image
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None


@app.websocket("/ws/inference")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected")
    
    try:
        while True:
            # 1. Receive base64 frame from browser
            data = await websocket.receive_text()
            
            # 2. Convert to OpenCV format
            frame = decode_base64_image(data)
            
            if frame is None:
                continue

            
           # ==========================================================
            # Run Inference (verbose=False prevents spamming your terminal)
            fire_results = fire_model(frame, verbose=False)[0]
            posture_results = posture_model(frame, verbose=False)[0]
            
            # --- 1. Parse Fire & Smoke ---
            fire_status = "Safe"
            fire_alert = "safe"
            detected_fire_classes = []
            
            # Check if any bounding boxes were found
            if len(fire_results.boxes) > 0:
                # Extract the names of what was detected
                for cls_id in fire_results.boxes.cls:
                    class_name = fire_model.names[int(cls_id)]
                    detected_fire_classes.append(class_name)
                
                # Remove duplicates (e.g., if it finds 3 "smoke" boxes)
                unique_detections = list(set(detected_fire_classes))
                fire_status = " & ".join(unique_detections) + " Detected!"
                fire_alert = "alert" # Trigger the red flashing UI

            # --- 2. Parse Posture ---
            posture_status = "No patient detected"
            posture_alert = "normal"
            
            if len(posture_results.boxes) > 0:
                # Assuming the model might detect multiple things, get the one with highest confidence
                best_box = posture_results.boxes[0] 
                class_id = int(best_box.cls[0])
                posture_status = posture_model.names[class_id]
                
                # Trigger an alert if they are on the floor
                if posture_status == "lying on floor":
                    posture_alert = "alert" # Trigger the red flashing UI
            # ==========================================================

            # 3. Create payload for the frontend
            response = {
                "fire_smoke": fire_status,
                "posture": posture_status,
                "fire_alert": fire_alert,
                "posture_alert": posture_alert
            }
            
            # 4. Send results back
            await websocket.send_json(response)

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket Error: {e}")


@app.get("/")
async def get():
    # Redirect to the static index.html
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url='/static/index.html')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)