# Based on MNIST CNN from Keras' examples with ResNet architecture
# Implementation for Kuzushiji-MNIST dataset

from __future__ import print_function
import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input, Add, BatchNormalization, Activation
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import json
from tensorflow.keras.callbacks import Callback
import os

# Ensure output directory exists
output_dir = os.path.dirname(os.path.abspath(__file__))  # ResNet directory
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

def load(f):
    # Load data from project root directory
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Get project root directory
    file_path = os.path.join(root_dir, f)
    return np.load(file_path)['arr_0']

# Load the data
x_train = load('kmnist-train-imgs.npz')
x_test = load('kmnist-test-imgs.npz')
y_train = load('kmnist-train-labels.npz')
y_test = load('kmnist-test-labels.npz')

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('{} train samples, {} test samples'.format(len(x_train), len(x_test)))

# Convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Define residual block
def residual_block(x, filters, kernel_size=3, strides=1, use_conv_shortcut=False):
    shortcut = x
    
    # First convolution layer
    x = Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Second convolution layer
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    
    # Process shortcut connection
    if use_conv_shortcut:
        shortcut = Conv2D(filters, 1, strides=strides, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)
    
    # Add residual connection
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    
    return x

# Create ResNet model
def create_resnet_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    
    # Initial convolution layer
    x = Conv2D(32, 3, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Residual blocks
    x = residual_block(x, 32)
    x = residual_block(x, 32)
    
    # Downsampling
    x = residual_block(x, 64, strides=2, use_conv_shortcut=True)
    x = residual_block(x, 64)
    
    # Global average pooling
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Define plotting function
def plot_metrics(history, title, filename):
    epochs_range = range(1, len(history.history['loss']) + 1)
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, history.history['loss'], label='Train Loss')
    plt.plot(epochs_range, history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.legend()
    plt.xticks(epochs_range)  # Set x-axis to integers

    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, history.history['accuracy'], label='Train Accuracy')
    plt.plot(epochs_range, history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.legend()
    plt.xticks(epochs_range)  # Set x-axis to integers

    plt.suptitle(title)
    # Save to ResNet directory
    output_path = os.path.join(output_dir, f'resnet_{filename}')
    plt.savefig(output_path)
    plt.close()  # Close the plot

# Save training data to file
def save_history(history, filename):
    # Save to ResNet directory
    output_path = os.path.join(output_dir, f'resnet_{filename}')
    with open(output_path, 'w') as f:
        json.dump(history.history, f)

# Create custom callback
class InterruptTrainingCallback(Callback):
    def __init__(self, interrupt_epoch):
        super(InterruptTrainingCallback, self).__init__()
        self.interrupt_epoch = interrupt_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch == self.interrupt_epoch:
            print(f"Interrupting training at epoch {epoch + 1}")
            self.model.stop_training = True

# Define the epoch to interrupt training
interrupt_epoch = 13

# Create custom callback instance
interrupt_callback = InterruptTrainingCallback(interrupt_epoch=interrupt_epoch)

# Train ResNet model and record history
model = create_resnet_model(input_shape, num_classes)
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
model.summary()  # Print model structure

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test),
                    callbacks=[interrupt_callback])

# Plot and save training loss and accuracy
plot_metrics(history, 'ResNet Training with Categorical Crossentropy', 'categorical_crossentropy.png')
save_history(history, 'categorical_crossentropy_history.json')

# Train with different loss function
model = create_resnet_model(input_shape, num_classes)
model.compile(loss='mean_squared_error',
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
history_mse = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test),
                        callbacks=[interrupt_callback])

# Plot and save training loss and accuracy
plot_metrics(history_mse, 'ResNet Training with Mean Squared Error', 'mean_squared_error.png')
save_history(history_mse, 'mean_squared_error_history.json')

# Train with different learning rates and record results
learning_rates = [0.1, 0.01, 0.001, 0.0001]
histories_lr = []

for lr in learning_rates:
    model = create_resnet_model(input_shape, num_classes)
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adadelta(learning_rate=lr),
                  metrics=['accuracy'])
    history_lr = model.fit(x_train, y_train,
                           batch_size=batch_size,
                           epochs=epochs,
                           verbose=1,
                           validation_data=(x_test, y_test),
                           callbacks=[interrupt_callback])
    histories_lr.append(history_lr)
    plot_metrics(history_lr, f'ResNet Training with Learning Rate {lr}', f'learning_rate_{lr}.png')
    save_history(history_lr, f'learning_rate_{lr}_history.json')

# Train with different batch sizes and record results
batch_sizes = [8, 16, 32, 64, 128]
histories_bs = []

for bs in batch_sizes:
    model = create_resnet_model(input_shape, num_classes)
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    history_bs = model.fit(x_train, y_train,
                           batch_size=bs,
                           epochs=epochs,
                           verbose=1,
                           validation_data=(x_test, y_test),
                           callbacks=[interrupt_callback])
    histories_bs.append(history_bs)
    plot_metrics(history_bs, f'ResNet Training with Batch Size {bs}', f'batch_size_{bs}.png')
    save_history(history_bs, f'batch_size_{bs}_history.json')

# Get predictions for the first 100 test samples
predictions = model.predict(x_test[:100])
predicted_labels = np.argmax(predictions, axis=1)
actual_labels = np.argmax(y_test[:100], axis=1)

# Plot and save predictions for the first 100 test samples
plt.figure(figsize=(20, 20))
for i in range(100):
    plt.subplot(10, 10, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title(f'P: {predicted_labels[i]}, A: {actual_labels[i]}')
    plt.axis('off')
plt.subplots_adjust(wspace=0.5, hspace=0.5)  # Adjust spacing between subplots
# Save to ResNet directory
output_path = os.path.join(output_dir, 'resnet_predictions.png')
plt.savefig(output_path)
plt.close()  # Close the plot

# Compare CNN and ResNet model performance
def compare_models(cnn_history_file, resnet_history_file, output_filename):
    # Get benchmarks directory path
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    benchmarks_dir = os.path.join(root_dir, 'benchmarks')
    
    # Load history data
    cnn_history_path = os.path.join(benchmarks_dir, cnn_history_file)
    resnet_history_path = os.path.join(output_dir, resnet_history_file)
    
    with open(cnn_history_path, 'r') as f:
        cnn_history = json.load(f)
    
    with open(resnet_history_path, 'r') as f:
        resnet_history = json.load(f)
    
    epochs_range = range(1, min(len(cnn_history['loss']), len(resnet_history['loss'])) + 1)
    
    plt.figure(figsize=(12, 8))
    
    # Compare training loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs_range, cnn_history['loss'][:len(epochs_range)], label='CNN Train Loss')
    plt.plot(epochs_range, resnet_history['loss'][:len(epochs_range)], label='ResNet Train Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.xticks(epochs_range)
    
    # Compare validation loss
    plt.subplot(2, 2, 2)
    plt.plot(epochs_range, cnn_history['val_loss'][:len(epochs_range)], label='CNN Val Loss')
    plt.plot(epochs_range, resnet_history['val_loss'][:len(epochs_range)], label='ResNet Val Loss')
    plt.title('Validation Loss Comparison')
    plt.legend()
    plt.xticks(epochs_range)
    
    # Compare training accuracy
    plt.subplot(2, 2, 3)
    plt.plot(epochs_range, cnn_history['accuracy'][:len(epochs_range)], label='CNN Train Accuracy')
    plt.plot(epochs_range, resnet_history['accuracy'][:len(epochs_range)], label='ResNet Train Accuracy')
    plt.title('Training Accuracy Comparison')
    plt.legend()
    plt.xticks(epochs_range)
    
    # Compare validation accuracy
    plt.subplot(2, 2, 4)
    plt.plot(epochs_range, cnn_history['val_accuracy'][:len(epochs_range)], label='CNN Val Accuracy')
    plt.plot(epochs_range, resnet_history['val_accuracy'][:len(epochs_range)], label='ResNet Val Accuracy')
    plt.title('Validation Accuracy Comparison')
    plt.legend()
    plt.xticks(epochs_range)
    
    plt.tight_layout()
    # Save to ResNet directory
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path)
    plt.close()

# Add model comparison code after training
try:
    compare_models('categorical_crossentropy_history.json', 
                  'resnet_categorical_crossentropy_history.json', 
                  'model_comparison.png')
    print("Model comparison completed, results saved in ResNet/model_comparison.png")
except Exception as e:
    print(f"Unable to compare models: {e}")
    print("Please ensure both model history files exist")