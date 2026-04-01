import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import io
import base64
import cv2
import logging
import sys
import traceback
from datetime import datetime

# ✅ SETUP LOGGING
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('cliniscan.log')
    ]
)
logger = logging.getLogger(__name__)
logger.info("="*60)
logger.info("CliniScan Backend Started")
logger.info("="*60)

# ✅ CREATE APP FIRST
app = Flask(__name__)
CORS(app)
logger.info("Flask app initialized with CORS enabled")

# ✅ MODEL SETUP
model = None

def load_model():
    global model
    if model is None:
        logger.info("🔄 Loading YOLO model from best.pt...")
        try:
            model = YOLO("best.pt")
            logger.info("✅ Model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    return model

# ✅ THEN ROUTES
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'timestamp': datetime.now().isoformat()}), 200

@app.route('/predict', methods=['POST'])
def predict():
    logger.info("="*60)
    logger.info("📨 /predict endpoint called")
    logger.info(f"Request method: {request.method}")
    logger.info(f"Content-Type: {request.content_type}")
    
    if 'file' not in request.files:
        logger.warning("❌ No file in request.files")
        return jsonify({'error': 'No file uploaded'}), 400
        
    file = request.files['file']
    logger.info(f"📁 File received: {file.filename} ({file.content_length} bytes)")
    
    try:
        # Step 1: Load image
        logger.info("🖼️  Step 1: Opening image...")
        image = Image.open(file.stream).convert('RGB')
        logger.info(f"✅ Image opened successfully - Size: {image.size}")
        
        # Step 2: Load model
        logger.info("🔄 Step 2: Loading model...")
        model = load_model()
        logger.info("✅ Model ready")
        
        # Step 3: Run prediction
        logger.info("🧠 Step 3: Running YOLO prediction...")
        results = model.predict(image, conf=0.15, imgsz=320)
        logger.info(f"✅ Prediction complete - {len(results)} result(s)")

        # Step 4: Plot results
        logger.info("🎨 Step 4: Plotting results...")
        res_plotted = results[0].plot()
        res_plotted = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        logger.info("✅ Plot complete")
        
        # Step 5: Encode to base64
        logger.info("📤 Step 5: Encoding image to base64...")
        img_pil = Image.fromarray(res_plotted)
        buffered = io.BytesIO()
        img_pil.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        logger.info(f"✅ Image encoded - Base64 length: {len(img_str)} chars")
        
        # Step 6: Extract boxes
        logger.info("📊 Step 6: Extracting detection boxes...")
        boxes = results[0].boxes
        count = len(boxes) if boxes is not None else 0
        logger.info(f"✅ Found {count} boxes")
        
        logger.info("✅ Prediction successful - Sending response")
        logger.info("="*60)
        
        return jsonify({
            'success': True,
            'image_base64': img_str,
            'boxes_found': count
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Error processing image: {str(e)}")
        logger.error(traceback.format_exc())
        logger.info("="*60)
        return jsonify({'error': str(e)}), 500
@app.route('/')
def home():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), 'cliniscan.html')

# ✅ ERROR HANDLERS
@app.errorhandler(500)
def internal_error(error):
    logger.error(f"500 Internal Server Error: {str(error)}")
    logger.error(traceback.format_exc())
    return jsonify({'error': 'Internal server error', 'details': str(error)}), 500

@app.errorhandler(404)
def not_found(error):
    logger.warning(f"404 Not Found: {request.path}")
    return jsonify({'error': 'Endpoint not found'}), 404

# ✅ RUN APP
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"🚀 Starting server on 0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)