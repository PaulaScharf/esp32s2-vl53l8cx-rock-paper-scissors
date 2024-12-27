import tensorflow as tf
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix

# Convert model to tflite
def convert_tflite_model(model):
	converter = tf.lite.TFLiteConverter.from_keras_model(model)
	tflite_model = converter.convert()
	return tflite_model

def save_tflite_model(tflite_model, save_dir, model_name):
	import os
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	save_path = os.path.join(save_dir, model_name)
	with open(save_path, "wb") as f:
		f.write(tflite_model)
	print("Tflite model saved to %s", save_dir)

def test_tflite(tflite_model, X_test, y_test):
	interpreter = tf.lite.Interpreter(model_content=tflite_model)
	interpreter.allocate_tensors()
	# Get input and output tensors
	input_tensor_index = interpreter.get_input_details()[0]['index']
	output_tensor_index = interpreter.get_output_details()[0]['index']
	# Run inference on test data
	total_predictions = len(X_test)
	lite_predictions = []
	y_test = np.argmax(y_test, axis=1)

	for i in range(total_predictions):
		input_data = np.expand_dims(X_test[i], axis=0).astype(np.float32)
		interpreter.set_tensor(input_tensor_index, input_data)
		interpreter.invoke()
		lite_prediction = interpreter.get_tensor(output_tensor_index)[0]
		
		# Compare predicted label with true label
		lite_predictions.append(np.argmax(lite_prediction))
			
	precision = precision_score(y_test, lite_predictions, average=None)
	recall = recall_score(y_test, lite_predictions, average=None)
	accuracy = accuracy_score(y_test, lite_predictions)

	# Compute accuracy
	print("TensorFlow Lite model:")
	return accuracy,precision,recall