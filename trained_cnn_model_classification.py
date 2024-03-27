"""

@anilturkmen

Before diving into the project, it's important to read the README section. 
The README contains project details and explanations. If you have any questions, please feel free to contact me.

Thank you!

"""
import tensorflow

# VGG16 stands out as a pre-trained cornerstone in convolutional neural network technology.
conv_base = tensorflow.keras.applications.VGG16(weights='imagenet',
                                                include_top=False,
                                                input_shape=(224, 224, 3)
                                                )

# Displaying the intricate layers of convolutions
conv_base.summary()

# Determining the layers to be trained and set as unalterable. Freezing layers until 'block5_conv1'
conv_base.trainable = True
set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

# An empty blueprint for the model is established.
model = tensorflow.keras.models.Sequential()

# Integrating VGG16 as a foundational convolutional layer.
model.add(conv_base)

# Transforming layers from matrices into a vector format.
model.add(tensorflow.keras.layers.Flatten())

# Our neural network layer is incorporated.
model.add(tensorflow.keras.layers.Dense(256, activation='relu'))
model.add(tensorflow.keras.layers.Dense(5, activation='softmax'))

model.compile(loss='binary_crossentropy',
              optimizer=tensorflow.keras.optimizers.RMSprop(learning_rate=1e-5),
              metrics=['acc'])

# Displaying the finalized model architecture.
model.summary()

# Specifying the directories containing the data.
EGITIM = '/Users/anilturkmen/Desktop/Bitirme_Tezi_Projem/veriseti/EGITIM'
GECERLEME_YOLU = '/Users/anilturkmen/Desktop/Bitirme_Tezi_Projem/veriseti/GECERLEME'
TEST = '/Users/anilturkmen/Desktop/Bitirme_Tezi_Projem/veriseti/TEST'

# To mitigate overfitting, data augmentation techniques are applied.
train_datagen = tensorflow.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,  # piksel değerleri 0-255'den 0-1 arasına getiriliyor.
    rotation_range=40,  # istenilen artırma işlemleri yapılabilir.
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    EGITIM,
    target_size=(224, 224),
    batch_size=16,
)

# For validating the training process, augmented images are not utilized.
validation_datagen = tensorflow.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255
)

validation_generator = validation_datagen.flow_from_directory(
    GECERLEME_YOLU,
    target_size=(224, 224),
    batch_size=16,
)

# Training process commences for the model.

EGITIM_TAKIP = model.fit(
    train_generator,
    steps_per_epoch=10,
    epochs=2,
    validation_data=validation_generator,
    validation_steps=1)

# Saving the trained model to the working directory
model.save('yuzde_doksan12_bessinif.keras')

# For testing the trained model, augmented images are unnecessary.
test_datagen = tensorflow.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255
)

test_generator = test_datagen.flow_from_directory(
    TEST,
    target_size=(224, 224),
    batch_size=16,
)

# Printing out the test results.
test_loss, test_acc = model.evaluate(test_generator, steps=25)
print('test acc:', test_acc)
