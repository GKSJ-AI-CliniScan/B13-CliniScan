@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
        
    file = request.files['file']
    
    try:
        image = Image.open(file.stream).convert('RGB')
        
        # 🔥 LOAD MODEL HERE (LAZY LOAD)
        model = load_model()

        # 🔥 REDUCE MEMORY USAGE
        results = model.predict(image, conf=0.15, imgsz=320)

        res_plotted = results[0].plot()
        res_plotted = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        
        img_pil = Image.fromarray(res_plotted)
        buffered = io.BytesIO()
        img_pil.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return jsonify({
            'success': True,
            'image_base64': img_str,
            'boxes_found': len(results[0].boxes)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})