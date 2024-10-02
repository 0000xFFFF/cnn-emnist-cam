#!/usr/bin/env python3

import sys
import os
import numpy as np
import pandas as pd
from utils_tf_selectmodel import selectmodel
model = selectmodel()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from emnist_maps import emnist_class_mapping_reversed

from flask import Flask, request, jsonify, render_template, send_file, redirect
from flask_cors import CORS
import base64
from io import BytesIO
from PIL import Image

# make output dataframe
df = pd.DataFrame(emnist_class_mapping_reversed.items(), columns=["key", "char"])
clm = "prediction"
df[clm] = "0"

app = Flask(__name__)
CORS(app)


@app.route('/')
def root():
    return redirect('/index')


@app.route('/index', methods=['GET'])
def index():
    return send_file('test_draw.html')


@app.route('/endpoint', methods=['POST'])
def receive_image():
    data_url = request.json.get('image')
    base64_string = data_url.split("data:image/png;base64,")[1]
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    image_array = np.array(image)
    image_array = np.max(image_array, axis=2, keepdims=True)
    image_array = np.expand_dims(image_array, axis=0)
    predictions = model.predict(image_array, verbose=0)
    # value = np.argmax(predictions, axis=1)
    # value_id = value[0]
    # label = emnist_class_mapping_reversed[value_id]
    values = predictions[0]
    df[clm] = [pred * 100 for pred in values]
    sorted_df = df.sort_values(by=clm, ascending=False)
    return jsonify({'prediction': str(sorted_df.to_string())})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

