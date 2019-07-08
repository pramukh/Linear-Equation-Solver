from flask import Flask, request
from flask_cors import CORS, cross_origin
from keras.models import load_model
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras import backend
from tensorflow.python.client import device_lib
import time

# assert 'GPU' in str(device_lib.list_local_devices())
#
# assert len(backend.tensorflow_backend._get_available_gpus()) > 0

# config = tf.ConfigProto()
# config.gpu_options.allow_growth=True
# sess = tf.Session(config=config)

# Load in the saved neural network
model = load_model('saved_model.h5')
labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '*',
          '/', '=']

# Setting up the Flask app
app = Flask(__name__)

# Allow Cross-Origin Resource Sharing
# cors = CORS(app)

import cv2 as cv
import numpy as np
from copy import deepcopy
from itertools import combinations
import base64

graph = tf.get_default_graph()


def union(a, b):
    x = min(a[0], b[0])
    y = min(a[1], b[1])
    w = max(a[0] + a[2], b[0] + b[2]) - x
    h = max(a[1] + a[3], b[1] + b[3]) - y
    return (x, y, w, h)


def intersection(a, b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0] + a[2], b[0] + b[2]) - x
    h = min(a[1] + a[3], b[1] + b[3]) - y
    if w < 0 or h < 0: return ()  # or (0,0,0,0) ?
    return (x, y, w, h)


def rect_area(rect):
    try:
        return rect[2] * rect[3]
    except IndexError:
        return 0


def preprocess(image):
    img = data_uri_to_cv2_img(image)
    gray = img.copy()
    blurred = cv.GaussianBlur(gray, (5, 5), 5)
    thresh = cv.adaptiveThreshold(blurred, 255, 0, 1, 115, 1)

    kernel = np.ones((3, 3), np.uint8)
    dilation = cv.dilate(thresh, kernel, iterations=0)

    _, contours, hierarchy = cv.findContours(dilation, cv.RETR_EXTERNAL,
                                             cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda ctr: cv.boundingRect(ctr)[0])
    rects = [cv.boundingRect(ctr) for ctr in contours]

    print(rects)
    new_rects = []
    pad = 2
    for rect in rects:
        if rect[2] / rect[3] > 5:
            temp = []
            temp.append(rect[0] - pad)
            temp.append(rect[1] - int(rect[2]))
            temp.append(rect[2] + pad)
            temp.append(rect[3] + int(2 * rect[2]))
            new_rects.append(tuple(temp))
        else:
            new_rects.append(rect)

    windows = list(zip(new_rects, new_rects[1:], new_rects[2:]))
    final_rects = deepcopy(new_rects)
    for window in windows:
        for combination in combinations(window, 2):
            combination = list(combination)
            common = intersection(combination[0], combination[1])
            area_common = rect_area(common)
            area1 = rect_area(combination[0])
            area2 = rect_area(combination[1])
            combination = sorted(combination, key=lambda x: rect_area(x),
                                 reverse=True)
            try:
                if area_common == area1 or area_common == area2:
                    final_rects.remove(combination[-1])
                if area_common > area1 / 2 and area_common > area2 / 2:
                    final_rects.remove(combination[-1])
            except ValueError:
                pass

    output_img = img.copy()
    extracted_rects = []
    pad = 1
    for (x, y, w, h) in final_rects:
        extracted_rects.append(
            dilation.copy()[y - pad: y + h + pad, x - pad: x + w + pad])
        rect = cv.rectangle(output_img, (x, y), (x + w, y + h), (0, 0, 255), 3)

    final_extracted_rects = []
    pad = 0

    for i in extracted_rects:
        boxes = []

        _, cnts, hierarchy = cv.findContours(i, cv.RETR_EXTERNAL,
                                             cv.CHAIN_APPROX_SIMPLE)

        for c in cnts:
            (x, y, w, h) = cv.boundingRect(c)
            boxes.append([x, y, x + w, y + h])

        boxes = np.asarray(boxes)
        left = np.min(boxes[:, 0])
        top = np.min(boxes[:, 1])
        right = np.max(boxes[:, 2])
        bottom = np.max(boxes[:, 3])

        final_extracted_rects.append(
            i.copy()[top - pad: bottom + pad, left - pad: right + pad])

    square_padded_digits = []
    b_pad = 1
    for i in final_extracted_rects:
        old_size = i.shape
        desired_size = max(i.shape)

        delta_w = desired_size - old_size[1]
        delta_h = desired_size - old_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        square_padded_digits.append(
            cv.copyMakeBorder(i, top + b_pad, bottom + b_pad, left + b_pad,
                              right + b_pad, borderType=cv.BORDER_CONSTANT,
                              value=[0, 0, 0]))

    final_symbols = []
    for i in square_padded_digits:
        #     i = cv.dilate(i,kernel,iterations = 1)
        resized = cv.resize(i, (47, 47))
        _, segmented_thresh = cv.threshold(resized, 0, 1, cv.THRESH_BINARY)
        final_symbols.append(segmented_thresh)
        #     closing = cv.morphologyEx(segmented_thresh, cv.MORPH_CLOSE, kernel)

    return final_symbols


def data_uri_to_cv2_img(uri):
    """
    Convert a data URL to an OpenCV image
    Credit: https://stackoverflow.com/a/54205640/2415512
    : param uri : data URI representing a BW image
    : returns   : OpenCV image
    """

    encoded_data = uri.split(',')[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv.imdecode(nparr, cv.IMREAD_GRAYSCALE)
    return img


# Serve a canvas interface on /

@app.route('/')
def api_root():
    return app.send_static_file('index.html')


@app.route('/post-data-url', methods=['POST'])
@cross_origin()
def api_predict_from_dataurl():
    # Read the image data from a base64 data URL
    imgstring = request.form.get('data')

    final_symbols = preprocess(imgstring)

    final_symbols = np.asarray(final_symbols).reshape(-1, 47, 47, 1)

    with graph.as_default():
        model_predictions = model.predict_classes(final_symbols)

    predicted_symbols = [labels[i] for i in model_predictions]

    if predicted_symbols[-1] == '=':
        predicted_symbols.pop()

    print(predicted_symbols)

    answer = eval(''.join(predicted_symbols))

    # Convert to OpenCV image

    # Return the prediction
    return str(answer)


# Start flask app
if __name__ == '__main__':
    from os import environ

    # app.run(debug=False, port=environ.get("PORT", 5000), host='0.0.0.0')
    app.run(debug=False)
