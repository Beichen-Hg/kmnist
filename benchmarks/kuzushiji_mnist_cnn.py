# Based on MNIST CNN from Keras' examples: https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py (MIT License)

from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import json
from keras.callbacks import Callback

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

def load(f):
    return np.load(f)['arr_0']

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

# 创建模型初始化函数
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

# 定义绘图函数
def plot_metrics(history, title, filename):
    epochs_range = range(1, len(history.history['loss']) + 1)
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, history.history['loss'], label='Train Loss')
    plt.plot(epochs_range, history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.legend()
    plt.xticks(epochs_range)  # 设置横坐标为整数

    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, history.history['accuracy'], label='Train Accuracy')
    plt.plot(epochs_range, history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.legend()
    plt.xticks(epochs_range)  # 设置横坐标为整数

    plt.suptitle(title)
    plt.savefig(filename)
    plt.close()  # 关闭图表

# 保存训练数据到文件
def save_history(history, filename):
    with open(filename, 'w') as f:
        json.dump(history.history, f)

# 创建自定义回调函数
class InterruptTrainingCallback(Callback):
    def __init__(self, interrupt_epoch):
        super(InterruptTrainingCallback, self).__init__()
        self.interrupt_epoch = interrupt_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch == self.interrupt_epoch:
            print(f"Interrupting training at epoch {epoch + 1}")
            self.model.stop_training = True

# 定义中断训练的epoch
interrupt_epoch = 13

# 创建自定义回调函数实例
interrupt_callback = InterruptTrainingCallback(interrupt_epoch=interrupt_epoch)

# 训练模型并记录历史
model = create_model(input_shape, num_classes)
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test),
                    callbacks=[interrupt_callback])

# 绘制并保存训练过程中的损失和准确率
plot_metrics(history, 'Training with Categorical Crossentropy', 'categorical_crossentropy.png')
save_history(history, 'categorical_crossentropy_history.json')

# 使用不同的损失函数
model = create_model(input_shape, num_classes)
model.compile(loss='mean_squared_error',
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
history_mse = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test),
                        callbacks=[interrupt_callback])

# 绘制并保存训练过程中的损失和准确率
plot_metrics(history_mse, 'Training with Mean Squared Error', 'mean_squared_error.png')
save_history(history_mse, 'mean_squared_error_history.json')

# 使用不同的学习率进行训练并记录结果
learning_rates = [0.1, 0.01, 0.001, 0.0001]
histories_lr = []

for lr in learning_rates:
    model = create_model(input_shape, num_classes)
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
    plot_metrics(history_lr, f'Training with Learning Rate {lr}', f'learning_rate_{lr}.png')
    save_history(history_lr, f'learning_rate_{lr}_history.json')

# 使用不同的批量大小进行训练并记录结果
batch_sizes = [8, 16, 32, 64, 128]
histories_bs = []

for bs in batch_sizes:
    model = create_model(input_shape, num_classes)
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
    plot_metrics(history_bs, f'Training with Batch Size {bs}', f'batch_size_{bs}.png')
    save_history(history_bs, f'batch_size_{bs}_history.json')

# 获取前100个测试样本的预测结果
predictions = model.predict(x_test[:100])
predicted_labels = np.argmax(predictions, axis=1)
actual_labels = np.argmax(y_test[:100], axis=1)

# 绘制并保存前100个测试样本的预测结果
plt.figure(figsize=(20, 20))
for i in range(100):
    plt.subplot(10, 10, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title(f'P: {predicted_labels[i]}, A: {actual_labels[i]}')
    plt.axis('off')
plt.subplots_adjust(wspace=0.5, hspace=0.5)  # 调整子图之间的间距
plt.savefig('predictions.png')
plt.close()  # 关闭图表
