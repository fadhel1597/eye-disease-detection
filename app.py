import io
import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)

tf.config.set_visible_devices([], 'GPU')

model = load_model('weights/model.h5', compile=False)
classes = ['A', 'C', 'D', 'G', 'H', 'M', 'N', 'O']



def image_resize(image_path, dim):
  img = cv2.imread(image_path)
  if img.shape[1] != img.shape[0]:
    x = img.shape[1]//2
    y = img.shape[0]//2
    x = x-y
    img = img[0:0+img.shape[0], x:x+img.shape[0]]
  # resize image
  return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

def CLAHE(image_path, dim, clipLimit, tileGridSize):
  img = image_resize(image_path, dim)
  clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
  lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
  l, a, b = cv2.split(lab)  # split on 3 different channels
  l2 = clahe.apply(l)  # apply CLAHE to the L-channel
  lab = cv2.merge((l2,a,b))  # merge channels
  img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  return img

@app.route("/image-classification",  methods=["POST"])
def image_classification():

    image = request.files["img"]
    filename = secure_filename(image.filename)

    # Save the image to a temporary location
    temp_path = os.path.join("/tmp", filename)
    image.save(temp_path)

    # Read and prepare image
    img = CLAHE(temp_path, (230, 230), 20, (10,10))
    imgarray = tf.keras.preprocessing.image.img_to_array(img)
    imgarray = np.expand_dims(imgarray, axis=0)
    images = np.vstack([imgarray])/255

    # Generate prediction
    prediction = (model.predict(images) > 0.5).astype('float')
    prediction = pd.Series(prediction[0])
    prediction.index = classes
    prediction = prediction[prediction==1].index.values
    prediction_list = prediction.tolist()

    response = {'detected_labels' : prediction_list}

    os.remove(temp_path)

    return jsonify(response)

if __name__ == "__main__":
    
    app.run(host="0.0.0.0", port=5002, debug=True)  # debug=True causes Restarting with stat