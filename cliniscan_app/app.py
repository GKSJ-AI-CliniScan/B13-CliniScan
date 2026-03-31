import os  # <-- ADD THIS LINE
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import io
import base64
import cv2

app = Flask(__name__)
# This allows your HTML website to securely talk to this Python brain
CORS(app) 

print("⏳ Loading YOLO model...")
model = YOLO('best.pt') 
print("✅ Model loaded successfully!")

@app.route('/predict', methods=['POST'])
def predict():
    # 1. Check if the HTML sent a file
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
        
    file = request.files['file']
    
    try:
        # 2. Open the image
        image = Image.open(file.stream).convert('RGB')
        
        # 3. Run your AI
        results = model.predict(image, conf=0.15)
        
        # 4. Draw the bounding boxes
        res_plotted = results[0].plot()
        res_plotted = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        
        # 5. Convert the drawn image into text (Base64) so it can travel over the internet
        img_pil = Image.fromarray(res_plotted)
        buffered = io.BytesIO()
        img_pil.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        # 6. Send the data back to the HTML website
        return jsonify({
            'success': True,
            'image_base64': img_str,
            'boxes_found': len(results[0].boxes)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)