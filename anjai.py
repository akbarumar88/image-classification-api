from flask import Flask, request, jsonify
import tensorflow as tf
import cv2
import numpy as np
from flask_cors import CORS, cross_origin
from sklearn.preprocessing import LabelEncoder

import keras
from keras.models import load_model
# from flask import request


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

arr = [
    {"id": 1, 'name': 'Akbar'},
    {'id': 2, 'name': 'Riza'},
    {'id': 3, 'name': 'Fahmi'},
    {'id': 4, 'name': 'Demetrious J.'},
]

print("TensorFlow version", tf.__version__)
print("keras version", keras.__version__)
# model = load_model('caltech-2.h5')
print('**start loading model')
model_batik = load_model('mymodel-final.h5')
print('**finish loading model')


@app.route('/get/<int:id>', methods=['GET', 'POST', 'PUT'])
def hello(id):
    print("id nya boss - ", id)

    item = [item for item in arr if item['id'] == id]
    return jsonify(item)


@app.route('/test', methods=['POST'])
@cross_origin()
def test():
    return jsonify({
        'message': "hello",
        "status": 1
    })


@app.route('/detectbatik', methods=['POST'])
@cross_origin()
def detectbatik():

    print("baca file upload", request.files)
    # return jsonify({})
    filestr = request.files['sample'].read()
    print("convert ke bytes")
    filebyte = np.fromstring(filestr, np.uint8)
    # sample.filenam
    # print(type(sample))
    # img = cv2.imread(sample)

    print("imdecode")
    img = cv2.imdecode(filebyte, cv2.IMREAD_UNCHANGED)
    # img = cv2.imread("caracal.jpeg")

    print("start predict", type(img))
    # return ({})
    result = predictbatik(img)
    return (result)


@app.route('/detect', methods=['POST'])
@cross_origin()
def detect():

    print("baca file upload")
    filestr = request.files['sample'].read()
    print("convert ke bytes")
    filebyte = np.fromstring(filestr, np.uint8)
    # sample.filenam
    # print(type(sample))
    # img = cv2.imread(sample)

    print("imdecode")
    img = cv2.imdecode(filebyte, cv2.IMREAD_UNCHANGED)
    # img = cv2.imread("caracal.jpeg")

    print("start predict", type(img))
    # return ({})
    result = predict(img)
    return jsonify(result)


global label_names

# Must be same as Annotations list we used to choose the data
label_names = ['butterfly', 'cougar_face', 'elephant']
classes_list = label_names

# This function will preprocess images.


def preprocess(img, image_size=300):

    image = cv2.resize(img, (image_size, image_size))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = image.astype("float") / 255.0
    image = image.astype("float")

    # Expand dimensions as predict expect image in batches
    image = np.expand_dims(image, axis=0)
    return image


def postprocess(image, results):
    # Split the results into class probabilities and box coordinates
    bounding_box, class_probs = results

    # First let's get the class label

    # The index of class with the highest confidence is our target class
    class_index = np.argmax(class_probs)

    # Use this index to get the class name.
    class_label = label_names[class_index]

    # Now you can extract the bounding box too.

    # Get the height and width of the actual image
    h, w = image.shape[:2]

    # Extract the Coordinates
    x1, y1, x2, y2 = bounding_box[0]

    # Convert the coordinates from relative (i.e. 0-1) to actual values
    x1 = int(w * x1)
    x2 = int(w * x2)
    y1 = int(h * y1)
    y2 = int(h * y2)

    # return the lable and coordinates
    return class_label, (x1, y1, x2, y2), class_probs

# We will use this function to make prediction on images.


def predict(image, returnimage=False,  scale=0.9):

    # Before we can make a prediction we need to preprocess the image.
    processed_image = preprocess(image)

    # Now we can use our model for prediction
    results = model.predict(processed_image)

    # Now we need to postprocess these results.
    # After postprocessing, we can easily use our results
    label, (x1, y1, x2, y2), confidence = postprocess(image, results)

    return {
        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'label': label
    }
    # Now annotate the image
    # cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 100), 2)
    # cv2.putText(
    #     image,
    #     '{}'.format(label, confidence),
    #     (x1, y2 + int(35 * scale)),
    #     cv2.FONT_HERSHEY_COMPLEX, scale,
    #     (200, 55, 100),
    #     2
    # )

    # Show the Image with matplotlib
    # plt.figure(figsize=(10, 10))
    # plt.imshow(image[:, :, ::-1])


def predictbatik(image):
    # return jsonify({
    #     'message': 'success'
    # })
    # Before we can make a prediction we need to preprocess the image.
    IMG_SIZE = 150
    try:
        print("preprocessing image!")
        processed_image = preprocess(image, image_size=IMG_SIZE)
        # Now we can use our model for prediction

        print("done preprocessing image!")
        # print(processed_image)

        Z=['batik-kawung','batik-megamendung','batik-nitik','batik-parang']
        # Z = ['batik-bali', 'batik-betawi', 'batik-celup', 'batik-cendrawasih', 'batik-ceplok', 'batik-ciamis', 'batik-garutan', 'batik-gentongan', 'batik-kawung', 'batik-keraton',
        #      'batik-lasem', 'batik-megamendung', 'batik-parang', 'batik-pekalongan', 'batik-priangan', 'batik-sekar', 'batik-sidoluhur', 'batik-sidomukti', 'batik-sogan']
        X = [processed_image]

        # getting predictions on val set.
        # pred=model.predict(x_test)
        print("start predict", type(X))
        pred = model_batik.predict(X)
        print("finish predict", type(pred))

        pred_digits = np.argmax(pred, axis=1)
        indexclass = pred_digits[0]

        print('pred', pred)
        print('pred_digits', pred_digits)

        return jsonify({
            'status': 1,
            'message': 'Success!',
            'label': Z[indexclass]
        })
    except Exception as err:
        return jsonify({
            'status': 0,
            'message': 'Something went wrong!',
            'err': repr(err)
        })
