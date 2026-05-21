import os
import json
import cv2
import librosa
import IPython
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
from warnings import filterwarnings

filterwarnings(action='ignore')

def audio_to_tensors(audio_file):
    audio, sample_rate = librosa.load(audio_file)
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_features = np.mean(mfccs_features, axis=1)
    return mfccs_features

directory = "C:/Users/mukun/Downloads/ML_Project/Voice of Birds"

extracted_features = []

total_files = sum(len(files) for _, _, files in os.walk(directory))

with tqdm(total=total_files, desc='Processing files') as pbar:
    for target_class in os.listdir(directory):
        target_class_path = os.path.join(directory, target_class)

        if os.path.isdir(target_class_path):
            for audio_file in os.listdir(target_class_path):
                audio_path = os.path.join(directory, target_class, audio_file)

                try:
                    features = audio_to_tensors(audio_path)
                    extracted_features.append([features, target_class])

                except Exception as e:
                    print(f"Skipping {audio_file}: {e}")
                    continue

                pbar.update(1)

features_df = pd.DataFrame(extracted_features, columns=['features', 'class'])

target_encoding = LabelEncoder().fit_transform(features_df['class'])

features_df['target'] = target_encoding.tolist()

prediction_dict = features_df.set_index('target')['class'].to_dict()

with open(r"C:\Users\mukun\Downloads\ML_Project\prediction.json", "w") as f:
    json.dump(prediction_dict, f)

features = np.array(features_df['features'].tolist())
features = np.expand_dims(features, axis=2)

target = np.array(features_df['target'].tolist())

features_tensor = tf.convert_to_tensor(features, dtype=tf.float32)
target_tensor = tf.convert_to_tensor(target)

dataset = tf.data.Dataset.from_tensor_slices((features_tensor, target_tensor))

batch_size = 32

dataset = dataset.shuffle(len(features))

train_size = int(0.8 * len(features))
val_size = int(0.1 * len(features))

train_ds = dataset.take(train_size)
validation_ds = dataset.skip(train_size).take(val_size)
test_ds = dataset.skip(train_size + val_size)

train_ds = train_ds.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
validation_ds = validation_ds.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)

for audio_batch, label_batch in train_ds.take(1):
    print(audio_batch.numpy()[0].shape)
    print(audio_batch.numpy()[0])
    print()
    print(label_batch.numpy().shape)
    print(label_batch.numpy()[0])

mfcc_features = 40
channel = 1
target_classes = 114

input_shape = (mfcc_features, channel)

model = keras.Sequential([
    keras.layers.Input(shape=input_shape),
    keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool1D(pool_size=2, padding='same'),
    keras.layers.Conv1D(filters=256, kernel_size=3, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool1D(pool_size=2, padding='same'),
    keras.layers.Conv1D(filters=256, kernel_size=3, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool1D(pool_size=2, padding='same'),
    keras.layers.Flatten(),
    keras.layers.Dense(units=512, activation='relu', kernel_regularizer=keras.regularizers.L2(1e-2)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(units=512, activation='relu', kernel_regularizer=keras.regularizers.L2(1e-2)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(units=target_classes, activation='softmax')
])

model.summary()

model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

history = model.fit(train_ds, epochs=150, verbose=1, validation_data=validation_ds, callbacks=[early_stopping])

test_loss, test_accuracy = model.evaluate(test_ds)

print(f"\nTest Accuracy : {test_accuracy * 100:.2f}%")
print(f"Test Loss : {test_loss:.4f}")

y_true = []
y_pred = []

for x_batch, y_batch in test_ds:
    predictions = model.predict(x_batch, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)

    y_true.extend(y_batch.numpy())
    y_pred.extend(predicted_labels)

print(classification_report(y_true, y_pred))

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(12, 10))
plt.imshow(cm, interpolation='nearest')
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

model.save(r'C:\Users\mukun\Downloads\ML_Project\model.keras')

plt.figure(figsize=(12, 3))

plt.subplot(1, 2, 1)
plt.plot(range(len(acc)), acc, label='Training Accuracy')
plt.plot(range(len(val_acc)), val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training vs Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(range(len(loss)), loss, label='Training Loss')
plt.plot(range(len(val_loss)), val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training vs Validation Loss')

def prediction(audio_file):
    with open(r'C:\Users\mukun\Downloads\ML_Project\prediction.json', mode='r') as f:
        prediction_dict = json.load(f)

    audio, sample_rate = librosa.load(audio_file)

    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_features = np.mean(mfccs_features, axis=1)

    mfccs_features = np.expand_dims(mfccs_features, axis=0)
    mfccs_features = np.expand_dims(mfccs_features, axis=2)

    mfccs_tensors = tf.convert_to_tensor(mfccs_features, dtype=tf.float32)

    model = tf.keras.models.load_model(r'C:\Users\mukun\Downloads\ML_Project\model.keras')

    prediction = model.predict(mfccs_tensors)

    target_label = np.argmax(prediction)

    predicted_class = prediction_dict[str(target_label)]

    confidence = round(np.max(prediction) * 100, 2)

    print(f'Predicted Class : {predicted_class}')
    print(f'Confidence : {confidence}%')