


import tensorflow

# VGG16 is a pre-trained CNN model.
conv_base = tensorflow.keras.applications.VGG16(weights='imagenet',
                                                include_top=False,
                                                input_shape=(224, 224, 3)
                                                )

# Showing the convolutional layers.
conv_base.summary()

# Deciding which layers are trained and frozen.
# Until 'block5_conv1' are frozen.
conv_base.trainable = True
set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

# An empyty model is created.
model = tensorflow.keras.models.Sequential()

# VGG16 is added as convolutional layer.
model.add(conv_base)

# Layers are converted from matrices to a vector.
model.add(tensorflow.keras.layers.Flatten())

# Our neural layer is added.
model.add(tensorflow.keras.layers.Dense(256, activation='relu'))
model.add(tensorflow.keras.layers.Dense(5, activation='softmax'))

model.compile(loss='binary_crossentropy',
              optimizer=tensorflow.keras.optimizers.RMSprop(learning_rate=1e-5),
              metrics=['acc'])

# Showing the created model.
model.summary()

# Defining the directories that data are in.
EGITIM = '/Users/anilturkmen/Desktop/Bitirme_Tezi_Projem/veriseti/EGITIM'
GECERLEME_YOLU = '/Users/anilturkmen/Desktop/Bitirme_Tezi_Projem/veriseti/GECERLEME'
TEST = '/Users/anilturkmen/Desktop/Bitirme_Tezi_Projem/veriseti/TEST'

# We need to apply data augmentation methods to prevent overfitting.
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

# To validate the training process, we do not need augmented images.
validation_datagen = tensorflow.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255
)

validation_generator = validation_datagen.flow_from_directory(
    GECERLEME_YOLU,
    target_size=(224, 224),
    batch_size=16,
)

# Training the model.
# Training the model.
EGITIM_TAKIP = model.fit(
    train_generator,
    steps_per_epoch=10,
    epochs=2,
    validation_data=validation_generator,
    validation_steps=1)

# Saving the trained model to working directory.
model.save('yuzde_doksan12_bessinif.keras')

# To test the trained model, we do not need augmented images.
test_datagen = tensorflow.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255
)

test_generator = test_datagen.flow_from_directory(
    TEST,
    target_size=(224, 224),
    batch_size=16,
)

# Printing the test results.
test_loss, test_acc = model.evaluate(test_generator, steps=25)
print('test acc:', test_acc)
