from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
from werkzeug.utils import secure_filename
from PIL import Image
import uuid
import time

app = Flask(__name__, static_folder='static')

# Конфигурация
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(input_path):
    try:
        # Имитация обработки (можно заменить на реальные операции)
        time.sleep(5)
        
        img = Image.open(input_path)
        
        # Пример обработки: конвертация в черно-белое
        img = img.convert('L')
        
        processed_filename = 'processed_' + os.path.basename(input_path)
        processed_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
        img.save(processed_path)
        return processed_path
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = str(uuid.uuid4()) + '.' + secure_filename(file.filename).rsplit('.', 1)[1].lower()
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(original_path)
        
        processed_path = process_image(original_path)
        if processed_path:
            return jsonify({
                'processed_image': processed_path,
                'message': 'Image processed successfully'
            })
    
    return jsonify({'error': 'Failed to process image'}), 500

if __name__ == '__main__':
    app.run(port=667)