from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Convert labels to categorical (one-hot encoding)
y_train_cat = to_categorical(y_train, num_classes=2)
y_test_cat = to_categorical(y_test, num_classes=2)

# Load the pre-trained VGG16 model without the top layers
vgg = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))

# Freeze VGG16 layers
for layer in vgg.layers:
    layer.trainable = False

# Build the model
model = Sequential()

# Add the VGG16 base model
model.add(vgg)

# Add custom layers on top
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))  # 2 classes: Glaucoma, Normal

# Compile the model
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Model summary
model.summary()
