import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Constants
SEQUENCE_LENGTH = 10  # Adjust this if different in your training code
NUM_KEYPOINTS = 17
NUM_COORDS = 2
INPUT_SHAPE = (SEQUENCE_LENGTH, NUM_KEYPOINTS * NUM_COORDS)
NUM_CLASSES = 12

# Load the trained model
try:
    model = keras.models.load_model('MoveNet.h5', compile=False)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
except Exception as e:
    print(f"Error loading the model: {e}")
    print("Creating a model with the same architecture as in training.")

    inputs = keras.Input(shape=INPUT_SHAPE)
    x = keras.layers.BatchNormalization()(inputs)
    x = tf.keras.layers.Bidirectional(keras.layers.LSTM(256, activation='tanh', return_sequences=True,
                                                        kernel_regularizer=tf.keras.regularizers.l2(0.001)))(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.LSTM(128, activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Initialize webcam
cap = cv2.VideoCapture(0)

# MoveNet parameters
input_size = 192

# Load MoveNet model
interpreter = tf.lite.Interpreter(model_path="movenet_lightning.tflite")
interpreter.allocate_tensors()


def movenet(input_image):
    """Runs detection on an input image."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_image = tf.cast(input_image, dtype=tf.uint8)
    input_image = tf.image.resize_with_pad(input_image, input_size, input_size)
    input_image = tf.cast(input_image, dtype=tf.float32)
    input_image = tf.expand_dims(input_image, axis=0)

    interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
    interpreter.invoke()

    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    return keypoints_with_scores[0][0]


def process_frame(frame):
    image_height, image_width, _ = frame.shape
    keypoints = movenet(frame)

    # Convert keypoints to the format expected by your model
    processed_keypoints = []
    for kp in keypoints:
        processed_keypoints.extend([kp[1], kp[0]])  # x, y (excluding score)

    return np.array(processed_keypoints)


# Initialize sequence buffer
sequence_buffer = []

# Placeholder for class names (replace with your actual class names)
class_names = [f"Class_{i}" for i in range(NUM_CLASSES)]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Process the frame
    keypoints = process_frame(frame)

    # Add to sequence buffer
    sequence_buffer.append(keypoints)
    if len(sequence_buffer) > SEQUENCE_LENGTH:
        sequence_buffer.pop(0)

    # If we have a full sequence, make a prediction
    if len(sequence_buffer) == SEQUENCE_LENGTH:
        sequence = np.array(sequence_buffer)
        sequence = np.expand_dims(sequence, axis=0)
        prediction = model.predict(sequence)
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class]

        class_name = class_names[predicted_class]

        # Display the prediction on the frame
        cv2.putText(frame, f"{class_name}: {confidence:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Action Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()