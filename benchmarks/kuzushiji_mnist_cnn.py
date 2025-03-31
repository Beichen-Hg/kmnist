# Based on MNIST CNN from Keras' examples: https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py (MIT License)

from __future__ import print_function
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import json
import os

# Ask user whether to use GPU
use_gpu = input("Use GPU acceleration for training? (y/n): ").lower().strip() == 'y'

# Configure GPU
if use_gpu:
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        print(f"Found {len(physical_devices)} GPU(s):")
        for device in physical_devices:
            print(f" - {device}")
        # Configure GPU memory growth
        try:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            print("GPU memory growth enabled")
        except Exception as e:
            print(f"GPU memory configuration failed: {e}")
        
        # If multiple GPUs, ask user which one to use
        if len(physical_devices) > 1:
            gpu_index = int(input(f"Select GPU to use (0-{len(physical_devices)-1}): "))
            if 0 <= gpu_index < len(physical_devices):
                tf.config.set_visible_devices(physical_devices[gpu_index], 'GPU')
                print(f"Selected GPU {gpu_index}")
            else:
                print("Invalid GPU index, will use all GPUs")
    else:
        print("No GPU detected, will use CPU")
        use_gpu = False
else:
    print("Selected to use CPU")
    # Disable all GPUs
    tf.config.set_visible_devices([], 'GPU')

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Output directory is the current file's directory
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Ensure output directory exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# Load data from project root
x_train = np.load(os.path.join(PROJECT_ROOT, 'kmnist-train-imgs.npz'))['arr_0']
x_test = np.load(os.path.join(PROJECT_ROOT, 'kmnist-test-imgs.npz'))['arr_0']
y_train = np.load(os.path.join(PROJECT_ROOT, 'kmnist-train-labels.npz'))['arr_0']
y_test = np.load(os.path.join(PROJECT_ROOT, 'kmnist-test-labels.npz'))['arr_0']

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
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Create model initialization function
def create_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
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
    # Save to benchmarks directory
    output_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(output_path)
    plt.close()  # Close the plot

# Save training data to file
def save_history(history, filename):
    # Save to benchmarks directory
    output_path = os.path.join(OUTPUT_DIR, filename)
    with open(output_path, 'w') as f:
        json.dump(history.history, f)

# Create custom callback function
class InterruptTrainingCallback(Callback):
    def __init__(self, interrupt_epoch):
        super(InterruptTrainingCallback, self).__init__()
        self.interrupt_epoch = interrupt_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch == self.interrupt_epoch:
            print(f"Interrupting training at epoch {epoch + 1}")
            self.model.stop_training = True

# Add GPU monitoring callback
class GPUMonitorCallback(Callback):
    def __init__(self, use_gpu=False):
        super(GPUMonitorCallback, self).__init__()
        self.use_gpu = use_gpu
        self.epoch_times = []
        self.start_time = None
        
    def on_epoch_begin(self, epoch, logs=None):
        import time
        self.start_time = time.time()
        
    def on_epoch_end(self, epoch, logs=None):
        import time
        epoch_time = time.time() - self.start_time
        self.epoch_times.append(epoch_time)
        
        if self.use_gpu:
            try:
                # Try to get GPU utilization
                import subprocess
                result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], 
                                       stdout=subprocess.PIPE, text=True, check=True)
                gpu_util = result.stdout.strip()
                print(f"Epoch {epoch+1} - Time: {epoch_time:.2f} seconds, GPU Utilization: {gpu_util}%")
            except:
                print(f"Epoch {epoch+1} - Time: {epoch_time:.2f} seconds")
        else:
            print(f"Epoch {epoch+1} - Time: {epoch_time:.2f} seconds")
    
    def get_avg_epoch_time(self):
        if not self.epoch_times:
            return 0
        return sum(self.epoch_times) / len(self.epoch_times)

# Define interrupt training epoch
interrupt_epoch = 13

# Create custom callback function instance
interrupt_callback = InterruptTrainingCallback(interrupt_epoch=interrupt_epoch)
gpu_monitor = GPUMonitorCallback(use_gpu=use_gpu)

# Create callback list
callbacks = [interrupt_callback, gpu_monitor]

# Step 1: Test different loss functions
loss_functions = ['categorical_crossentropy', 'mean_squared_error']
loss_functions_map = {
    '1': 'categorical_crossentropy',
    '2': 'mean_squared_error'
}
loss_results = {}

print("\n===== Step 1: Testing Different Loss Functions =====")
print(f"Using {'GPU' if use_gpu else 'CPU'} for training")
print("1 - categorical_crossentropy")
print("2 - mean_squared_error")

for loss_idx, loss_fn in loss_functions_map.items():
    print(f"\nTraining CNN model, Loss Function [{loss_idx}]: {loss_fn}")
    model = create_model(input_shape, num_classes)
    model.compile(loss=loss_fn,
                  optimizer=tf.keras.optimizers.Adadelta(),  # Use default learning rate
                  metrics=['accuracy'])
    
    if loss_fn == 'categorical_crossentropy':  # Print summary for the first model only
        model.summary()
        
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,  # Use default batch size
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test),
                        callbacks=callbacks)
    
    loss_results[loss_fn] = {
        'accuracy': history.history['accuracy'][-1],
        'val_accuracy': history.history['val_accuracy'][-1],
        'loss': history.history['loss'][-1],
        'val_loss': history.history['val_loss'][-1],
        'avg_epoch_time': gpu_monitor.get_avg_epoch_time()
    }
    
    plot_metrics(history, f'Training with {loss_fn} ({("GPU" if use_gpu else "CPU")})', f'{loss_fn}.png')
    save_history(history, f'{loss_fn}_history.json')

# Print loss function results
print("\n===== Loss Function Experiment Results =====")
for loss_idx, loss_fn in loss_functions_map.items():
    metrics = loss_results[loss_fn]
    print(f"[{loss_idx}] {loss_fn}: Train Accuracy={metrics['accuracy']:.4f}, Validation Accuracy={metrics['val_accuracy']:.4f}, "
          f"Train Loss={metrics['loss']:.4f}, Validation Loss={metrics['val_loss']:.4f}, "
          f"Avg Epoch Time={metrics['avg_epoch_time']:.2f} seconds")

# User selects the best loss function
selected_loss_idx = input("\nSelect the loss function to use (Enter number 1-2): ")
while selected_loss_idx not in loss_functions_map:
    selected_loss_idx = input("Invalid input, please select again (Enter number 1-2): ")
selected_loss = loss_functions_map[selected_loss_idx]
print(f"Selected Loss Function: {selected_loss}")

# Step 2: Test different batch sizes using the selected loss function
batch_sizes = [8, 16, 32, 64, 128]
batch_sizes_map = {
    '1': 8,
    '2': 16,
    '3': 32,
    '4': 64,
    '5': 128
}
batch_results = {}

print(f"\n===== Step 2: Testing Different Batch Sizes with Loss Function {selected_loss} =====")
print("1 - Batch Size 8")
print("2 - Batch Size 16")
print("3 - Batch Size 32")
print("4 - Batch Size 64")
print("5 - Batch Size 128")

for bs_idx, bs in batch_sizes_map.items():
    print(f"\nTraining CNN model, Loss Function: {selected_loss}, Batch Size [{bs_idx}]: {bs}")
    model = create_model(input_shape, num_classes)
    model.compile(loss=selected_loss,
                  optimizer=tf.keras.optimizers.Adadelta(),  # Use default learning rate
                  metrics=['accuracy'])
    
    # Reset GPU monitor
    gpu_monitor = GPUMonitorCallback(use_gpu=use_gpu)
    callbacks = [interrupt_callback, gpu_monitor]
    
    history = model.fit(x_train, y_train,
                        batch_size=bs,  # Use different batch sizes
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test),
                        callbacks=callbacks)
    
    batch_results[bs] = {
        'accuracy': history.history['accuracy'][-1],
        'val_accuracy': history.history['val_accuracy'][-1],
        'loss': history.history['loss'][-1],
        'val_loss': history.history['val_loss'][-1],
        'avg_epoch_time': gpu_monitor.get_avg_epoch_time()
    }
    
    plot_metrics(history, f'Training with Batch Size {bs} ({("GPU" if use_gpu else "CPU")})', f'batch_size_{bs}.png')
    save_history(history, f'batch_size_{bs}_history.json')

# Print batch size results
print("\n===== Batch Size Experiment Results =====")
for bs_idx, bs in batch_sizes_map.items():
    metrics = batch_results[bs]
    print(f"[{bs_idx}] Batch Size {bs}: Train Accuracy={metrics['accuracy']:.4f}, Validation Accuracy={metrics['val_accuracy']:.4f}, "
          f"Train Loss={metrics['loss']:.4f}, Validation Loss={metrics['val_loss']:.4f}, "
          f"Avg Epoch Time={metrics['avg_epoch_time']:.2f} seconds")

# User selects the best batch size
selected_batch_idx = input("\nSelect the batch size to use (Enter number 1-5): ")
while selected_batch_idx not in batch_sizes_map:
    selected_batch_idx = input("Invalid input, please select again (Enter number 1-5): ")
selected_batch = batch_sizes_map[selected_batch_idx]
print(f"Selected Batch Size: {selected_batch}")

# Step 3: Test different learning rates using the selected loss function and batch size
learning_rates = [0.1, 0.01, 0.001, 0.0001]
learning_rates_map = {
    '1': 0.1,
    '2': 0.01,
    '3': 0.001,
    '4': 0.0001
}
lr_results = {}

print(f"\n===== Step 3: Testing Different Learning Rates with Loss Function {selected_loss} and Batch Size {selected_batch} =====")
print("1 - Learning Rate 0.1")
print("2 - Learning Rate 0.01")
print("3 - Learning Rate 0.001")
print("4 - Learning Rate 0.0001")

for lr_idx, lr in learning_rates_map.items():
    print(f"\nTraining CNN model, Loss Function: {selected_loss}, Batch Size: {selected_batch}, Learning Rate [{lr_idx}]: {lr}")
    model = create_model(input_shape, num_classes)
    model.compile(loss=selected_loss,
                  optimizer=keras.optimizers.Adadelta(learning_rate=lr),  # Use different learning rates
                  metrics=['accuracy'])
    
    # Reset GPU monitor
    gpu_monitor = GPUMonitorCallback(use_gpu=use_gpu)
    callbacks = [interrupt_callback, gpu_monitor]
    
    history = model.fit(x_train, y_train,
                        batch_size=selected_batch,  # Use selected batch size
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test),
                        callbacks=callbacks)
    
    lr_results[lr] = {
        'accuracy': history.history['accuracy'][-1],
        'val_accuracy': history.history['val_accuracy'][-1],
        'loss': history.history['loss'][-1],
        'val_loss': history.history['val_loss'][-1],
        'avg_epoch_time': gpu_monitor.get_avg_epoch_time()
    }
    
    plot_metrics(history, f'Training with Learning Rate {lr} ({("GPU" if use_gpu else "CPU")})', f'learning_rate_{lr}.png')
    save_history(history, f'learning_rate_{lr}_history.json')

# Print learning rate results
print("\n===== Learning Rate Experiment Results =====")
for lr_idx, lr in learning_rates_map.items():
    metrics = lr_results[lr]
    print(f"[{lr_idx}] Learning Rate {lr}: Train Accuracy={metrics['accuracy']:.4f}, Validation Accuracy={metrics['val_accuracy']:.4f}, "
          f"Train Loss={metrics['loss']:.4f}, Validation Loss={metrics['val_loss']:.4f}, "
          f"Avg Epoch Time={metrics['avg_epoch_time']:.2f} seconds")

# User selects the best learning rate for the final model comparison
selected_lr_idx = input("\nSelect the learning rate to use for model comparison (Enter number 1-4): ")
while selected_lr_idx not in learning_rates_map:
    selected_lr_idx = input("Invalid input, please select again (Enter number 1-4): ")
selected_lr = learning_rates_map[selected_lr_idx]
print(f"Selected Learning Rate: {selected_lr}")

# Train the final model with the best parameters and save the results
print(f"\n===== Training Final Model with Best Parameters =====")
print(f"Loss Function: {selected_loss}, Batch Size: {selected_batch}, Learning Rate: {selected_lr}")

model = create_model(input_shape, num_classes)
model.compile(loss=selected_loss,
              optimizer=keras.optimizers.Adadelta(learning_rate=selected_lr),
              metrics=['accuracy'])

# Reset GPU monitor
gpu_monitor = GPUMonitorCallback(use_gpu=use_gpu)
callbacks = [gpu_monitor]  # Do not interrupt training for the final model

final_history = model.fit(x_train, y_train,
                          batch_size=selected_batch,
                          epochs=epochs,
                          verbose=1,
                          validation_data=(x_test, y_test),
                          callbacks=callbacks)

# Save final model results
plot_metrics(final_history, f'Final CNN Model (Loss={selected_loss}, Batch={selected_batch}, LR={selected_lr}, {("GPU" if use_gpu else "CPU")})', 
             'final_model.png')
save_history(final_history, 'final_model_history.json')

# Print final training time
print(f"\nFinal model average epoch time: {gpu_monitor.get_avg_epoch_time():.2f} seconds")

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
# Save to benchmarks directory
output_path = os.path.join(OUTPUT_DIR, 'predictions.png')
plt.savefig(output_path)
plt.close()  # Close the plot

# Save training information about GPU/CPU
hardware_info = {
    'device': 'GPU' if use_gpu else 'CPU',
    'final_model_avg_epoch_time': gpu_monitor.get_avg_epoch_time(),
    'selected_loss': selected_loss,
    'selected_batch_size': selected_batch,
    'selected_learning_rate': selected_lr
}

with open(os.path.join(OUTPUT_DIR, 'hardware_info.json'), 'w') as f:
    json.dump(hardware_info, f)

print(f"\nTraining complete! The final model trained using {'GPU' if use_gpu else 'CPU'} has been saved.")

