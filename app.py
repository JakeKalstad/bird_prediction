from webapp import MLFlask
from flask import request, jsonify
from webapp.helpers import get_prediction
from flask_cors import cross_origin, logging

app = MLFlask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
logging.getLogger('flask_cors').level = logging.DEBUG


@app.route('/')
def hello():
    return "heart beat"


@app.route('/predict', methods=['POST', 'OPTIONS'])
@cross_origin()
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        class_id, class_name = get_prediction(app, image_bytes=img_bytes)
        response = jsonify(
            {'class_id': class_id, 'class_name': class_name})

        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
