import tensorflow as tf
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
from tensorflow.keras import layers, models

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

def convert_to_c_array(file_path, output_file):
    with open(file_path, "rb") as f:
        data = f.read()
    
    data_length = len(data)
    
    with open(output_file, "w") as f:
        f.write(f"unsigned char model_tflite[] = {{\n")
        for i, byte in enumerate(data):
            f.write(f"0x{byte:02x}, ")
            if (i + 1) % 12 == 0:
                f.write("\n")
        f.write(f"\n}};\n\n")
        f.write(f"unsigned int model_tflite_len = {data_length};\n")

# Function to load the TFLite model and run inference
def run_tflite_inference(tflite_model_path, X_test):
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    predictions = []
    for sample in X_test:
        # Prepare input tensor
        sample = np.expand_dims(sample, axis=0).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], sample)

        # Run inference
        interpreter.invoke()

        # Get output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predictions.append(output_data[0])

    return np.array(predictions)

def create_model():
    # Build a larger CNN model
    model = models.Sequential([
        layers.Reshape((8,8,1), input_shape=(64,), name='reshape'),
        layers.Conv2D(8, (3, 3), activation='relu', padding='same', input_shape=(8, 8, 1), name='conv2D_1'),
        layers.Conv2D(16, (3, 3), activation='relu', padding='same', name='conv2D_2'),
        layers.MaxPooling2D((2, 2), padding='same', name='maxPooling2D'),
        layers.Flatten(name='flatten'),
        layers.Dense(32, activation='relu', name='dense_1'),
        layers.Dense(3, activation='softmax', name='dense_2')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy', 
			tf.keras.metrics.Precision(name='precision'),
			tf.keras.metrics.Recall(name='recall')])
    
    return model
