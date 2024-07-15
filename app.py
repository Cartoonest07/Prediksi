import os
import torch
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path="path/to/your/model/best.pt")

def measure_height(image_path, model, pixel_to_meter):
    results = model(image_path)
    df = results.pandas().xyxy[0]
    if len(df) > 0:
        bbox = df.iloc[0]
        height_pixels = bbox['ymax'] - bbox['ymin']
        height_meters = height_pixels * pixel_to_meter
        return height_meters
    else:
        return None

@app.route('/measure_height', methods=['POST'])
def measure_height_route():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    image_path = os.path.join('/tmp', image_file.filename)
    image_file.save(image_path)

    pixel_to_meter = 0.0018  # Scale conversion, adjust as needed
    height_meters = measure_height(image_path, model, pixel_to_meter)

    if height_meters is not None:
        return jsonify({'height_meters': height_meters}), 200
    else:
        return jsonify({'error': 'No object detected'}), 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
