import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import io
import base64
import cv2

# ✅ CREATE APP FIRST
app = Flask(__name__)
CORS(app)

# ✅ MODEL SETUP
model = None

def load_model():
    global model
    if model is None:
        model = YOLO("best.pt")
    return model


# ✅ THEN ROUTES
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
        
    file = request.files['file']
    
    try:
        image = Image.open(file.stream).convert('RGB')
        
        model = load_model()
        results = model.predict(image, conf=0.15, imgsz=320)

        res_plotted = results[0].plot()
        res_plotted = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        
        img_pil = Image.fromarray(res_plotted)
        buffered = io.BytesIO()
        img_pil.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        boxes = results[0].boxes
        count = len(boxes) if boxes is not None else 0
        
        return jsonify({
            'success': True,
            'image_base64': img_str,
            'boxes_found': count
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})


# ✅ RUN APP
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)