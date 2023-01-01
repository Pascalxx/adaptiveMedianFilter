import numpy as np
import tensorflow as tf
import tensorflow.keras as K
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D
import os
from tqdm.notebook import tqdm


# 編碼器(Encoder)
class Encoder(K.layers.Layer):
    def __init__(self, filters):
        super(Encoder, self).__init__()
        self.conv1 = Conv2D(filters=filters[0], kernel_size=3, strides=1, activation='relu', padding='same')
        self.conv2 = Conv2D(filters=filters[1], kernel_size=3, strides=1, activation='relu', padding='same')
        self.conv3 = Conv2D(filters=filters[2], kernel_size=3, strides=1, activation='relu', padding='same')
        self.pool = MaxPooling2D((2, 2), padding='same')

    def call(self, input_features):
        x = self.conv1(input_features)
        # print("Ex1", x.shape)
        x = self.pool(x)
        # print("Ex2", x.shape)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.pool(x)
        return x


class Decoder(K.layers.Layer):
    def __init__(self, filters):
        super(Decoder, self).__init__()
        self.conv1 = Conv2D(filters=filters[2], kernel_size=3, strides=1, activation='relu', padding='same')
        self.conv2 = Conv2D(filters=filters[1], kernel_size=3, strides=1, activation='relu', padding='same')
        self.conv3 = Conv2D(filters=filters[0], kernel_size=3, strides=1, activation='relu', padding='valid')
        self.conv4 = Conv2D(1, 3, 1, activation='sigmoid', padding='same')
        self.upsample = UpSampling2D((2, 2))

    def call(self, encoded):
        x = self.conv1(encoded)
        # 上採樣
        x = self.upsample(x)

        x = self.conv2(x)
        x = self.upsample(x)

        x = self.conv3(x)
        x = self.upsample(x)

        return self.conv4(x)

class Autoencoder(K.Model):
    def __init__(self, filters):
        super(Autoencoder, self).__init__()
        self.loss = []
        self.encoder = Encoder(filters)
        self.decoder = Decoder(filters)

    def call(self, input_features):
        #print(input_features.shape)
        encoded = self.encoder(input_features)
        #print(encoded.shape)
        reconstructed = self.decoder(encoded)
        #print(reconstructed.shape)
        return reconstructed

def image_noise_removal():
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    tf.compat.v1.disable_eager_execution()

    # Creates a graph.
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)
    config = tf.compat.v1.ConfigProto(log_device_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.7
    # Creates a session with log_device_placement set to True.
    sess = tf.compat.v1.Session(config=config)
    # Runs the op.
    print(sess.run(c))

    print(tf.__version__)
    print(tf.test.is_gpu_available())


    # # Import Noisy Images
    # bad_frames = 'imgData/bad_frames'
    # noisy_frames = []
    # for file in sorted(os.listdir(bad_frames)):
    #     if any(extension in file for extension in ['.jpg', 'jpeg', '.png']):
    #         image = tf.keras.preprocessing.image.load_img(bad_frames + '/' + file, target_size=(252, 252), color_mode='grayscale')
    #         image = tf.keras.preprocessing.image.img_to_array(image).astype('float32') / 255
    #         noisy_frames.append(image)
    #
    # noisy_frames = np.array(noisy_frames)
    #
    # # Import Clean Images
    # good_frames = 'imgData/good_frames'
    # clean_frames = []
    # for file in sorted(os.listdir(good_frames)):
    #     if any(extension in file for extension in ['.jpg', 'jpeg', '.png']):
    #         image = tf.keras.preprocessing.image.load_img(good_frames + '/' + file, target_size=(252, 252), color_mode='grayscale')
    #         image = tf.keras.preprocessing.image.img_to_array(image).astype('float32') / 255
    #         clean_frames.append(image)
    #
    # clean_frames = np.array(clean_frames)
    #
    # # Split Images Into Training & Test
    # round(len(noisy_frames) * 0.8)
    #
    # # 80% of images to training set
    # noisy_train = noisy_frames[0:round(len(noisy_frames) * 0.8)]
    # noisy_test = noisy_frames[round(len(noisy_frames) * 0.8):]
    # clean_train = clean_frames[0:round(len(clean_frames) * 0.8)]
    # clean_test = clean_frames[round(len(clean_frames) * 0.8):]
    #
    # # 超參數設定
    # batch_size = 128
    # max_epochs = 50
    # filters = [32,32,16]
    # model = Autoencoder(filters)
    #
    # model.compile(loss='binary_crossentropy', optimizer='adam')
    # loss = model.fit(noisy_train,
    #                  clean_train,
    #                  validation_data=(noisy_test, clean_test),
    #                  epochs=max_epochs,
    #                  batch_size=batch_size)
    #
    # plt.plot(range(max_epochs), loss.history['loss'])
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.show()
    #
    # number = 10  # how many digits we will display
    # plt.figure(figsize=(20, 4))
    # for index in range(number):
    #     # display original
    #     ax = plt.subplot(2, number, index + 1)
    #     plt.imshow(noisy_test[index].reshape(252, 252), cmap='gray')
    #     ax.get_xaxis().set_visible(False)
    #     ax.get_yaxis().set_visible(False)
    #
    #     # display reconstruction
    #     ax = plt.subplot(2, number, index + 1 + number)
    #     plt.imshow(tf.reshape(model(noisy_test)[index], (252, 252)), cmap='gray')
    #     ax.get_xaxis().set_visible(False)
    #     ax.get_yaxis().set_visible(False)
    # plt.show()













    # Import Noisy Images
    bad_frames = 'imgData/bad_frames'
    noisy_frames = []
    for file in sorted(os.listdir(bad_frames)):
        if any(extension in file for extension in ['.jpg', 'jpeg', '.png']):
            image = tf.keras.preprocessing.image.load_img(bad_frames + '/' + file, target_size=(128, 128))
            image = tf.keras.preprocessing.image.img_to_array(image).astype('float32') / 255
            noisy_frames.append(image)

    noisy_frames = np.array(noisy_frames)

    # Import Clean Images
    good_frames = 'imgData/good_frames'
    clean_frames = []
    for file in sorted(os.listdir(good_frames)):
        if any(extension in file for extension in ['.jpg', 'jpeg', '.png']):
            image = tf.keras.preprocessing.image.load_img(good_frames + '/' + file, target_size=(128, 128))
            image = tf.keras.preprocessing.image.img_to_array(image).astype('float32') / 255
            clean_frames.append(image)

    clean_frames = np.array(clean_frames)

    # Split Images Into Training & Test
    round(len(noisy_frames) * 0.8)

    # 80% of images to training set
    noisy_train = noisy_frames[0:round(len(noisy_frames) * 0.8)]
    noisy_test = noisy_frames[round(len(noisy_frames) * 0.8):]
    clean_train = clean_frames[0:round(len(clean_frames) * 0.8)]
    clean_test = clean_frames[round(len(clean_frames) * 0.8):]

    # Autoencoder
    autoencoder = tf.keras.models.Sequential()
    # Layer 1
    autoencoder.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', input_shape=(128, 128, 3)))
    autoencoder.add(tf.keras.layers.LeakyReLU(alpha=0.01))
    # Layer 3
    autoencoder.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same'))
    autoencoder.add(tf.keras.layers.BatchNormalization())
    autoencoder.add(tf.keras.layers.LeakyReLU(alpha=0.01))
    # Layer 4
    autoencoder.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same'))
    autoencoder.add(tf.keras.layers.BatchNormalization())
    autoencoder.add(tf.keras.layers.LeakyReLU(alpha=0.01))
    # Layer 5
    autoencoder.add(tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3), padding='same'))
    autoencoder.add(tf.keras.layers.BatchNormalization())
    autoencoder.add(tf.keras.layers.LeakyReLU(alpha=0.01))
    # Layer 6
    autoencoder.add(tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3), padding='same'))
    autoencoder.add(tf.keras.layers.BatchNormalization())
    autoencoder.add(tf.keras.layers.LeakyReLU(alpha=0.01))
    # Layer 7
    autoencoder.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same'))
    autoencoder.add(tf.keras.layers.BatchNormalization())
    autoencoder.add(tf.keras.layers.LeakyReLU(alpha=0.01))
    # Layer 8
    autoencoder.add(tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=(3, 3), padding='same'))
    autoencoder.add(tf.keras.layers.BatchNormalization())
    autoencoder.add(tf.keras.layers.LeakyReLU(alpha=0.01))
    # Layer 9
    autoencoder.add(tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), padding='same'))
    autoencoder.add(tf.keras.layers.BatchNormalization())
    autoencoder.add(tf.keras.layers.LeakyReLU(alpha=0.01))
    # Layer 11
    autoencoder.add(
        tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=(3, 3), activation='sigmoid', padding='same'))

    autoencoder.summary()
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    history = autoencoder.fit(noisy_train, clean_train, epochs=50, validation_data=(noisy_test, clean_test))

    #  Model History
    # plt.figure(figsize=(12, 8))
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.legend(['Train', 'Test'])
    # plt.title('Binary Crossentropy vs. Epoch (Noise  = 0.1)')
    # plt.xlabel('Epoch')
    # plt.ylabel('Binary Crossentropy')
    # plt.xticks(ticks=history.epoch, labels=history.epoch)
    # plt.show()

    # Test Autoencoder
    results = autoencoder.predict(noisy_test)
    image = np.random.randint(0, len(noisy_test))
    plt.imshow(noisy_test[image])
    plt.show()

    plt.imshow(results[image])
    plt.show()

    plt.imshow(clean_test[image])
    plt.show()
