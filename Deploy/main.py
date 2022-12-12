import os
from app import app
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow import keras
import face_recognition
import cv2
import numpy as np
import imageio

IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 100
MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048
SEQ_LENGTH = 20

@app.route('/')
def upload_form():
	return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_video():
	if 'file' not in request.files:
		flash('No file part')
		return redirect(request.url)
	file = request.files['file']
	if file.filename == '':
		flash('No image selected for uploading')
		return redirect(request.url)
	else:
		filename = secure_filename(file.filename)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		#print('upload_video filename: ' + filename)
		flash('Video successfully uploaded and displayed below')
		return render_template('upload.html', filename=filename)


def load_video(path, max_frames=0, resize=(IMG_SIZE, IMG_SIZE)):
	cap = cv2.VideoCapture(path)
	total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	skip_frames_window = max(int(total_frames/SEQ_LENGTH),1)
	frames = []
	try:
		for frame_cntr in range(SEQ_LENGTH):
			cap.set(cv2.CAP_PROP_POS_FRAMES, frame_cntr*skip_frames_window)
			ret, frame = cap.read()
			if not ret:
				break
			frame = crop_face_center(frame)
			if frame is None:
				continue
			frame = cv2.resize(frame, resize)
			frame = frame[:, :, [2, 1, 0]]
			frames.append(frame)
			if len(frames) == max_frames:
				break
		print("Completed extracting frames")
	finally:
		cap.release()
	return np.array(frames)


@app.route('/display/<filename>')
def display_video(filename):
 #print('display_video filename: ' + filename)
 return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/predict/<filename>')
def sequence_prediction(filename):
 sequence_model = load_model('./models/inceptionNet_model.h5')
 class_vocab = ['FAKE','REAL']
 frames = load_video('static/uploads/' + filename)
 frame_features, frame_mask = prepare_single_video(frames)
 probabilities = sequence_model.predict([frame_features, frame_mask])[0]
 pred = probabilities.argmax()
 return render_template('upload.html', filename=filename, prediction=class_vocab[pred])


def prepare_single_video(frames):
 print("Preparing Frames")
 frames = frames[None, ...]
#  feature_extractor = load_model('./models/feature_extractor.h5')
 frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
 frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")
 for i, batch in enumerate(frames):
  video_length = batch.shape[0]
  length = min(MAX_SEQ_LENGTH, video_length)
  for j in range(length):
   frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
   frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked
 print("Completed preparing frames")
 return frame_features, frame_mask


def build_feature_extractor():

    # Using pretrained EfficientNetB1 Model
    feature_extractor = keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )

    # Adding a Preprocessing layer in the Model
    preprocess_input = keras.applications.inception_v3.preprocess_input

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")

feature_extractor = build_feature_extractor()

def crop_face_center(frame):
  face_locations = face_recognition.face_locations(frame)
  # images = []
  if(len(face_locations)>0):
    face_location = face_locations[0]
  else:
    # face_locations = face_recognition.face_locations(default_image)
    # face_location = face_locations[0]
   return None
  top,right,bottom,left = face_location
  face_image = frame[top:bottom, left:right]
  return face_image


if __name__ == "__main__":
    app.run()