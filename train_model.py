import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Konfigurasi Path dan Hyperparameter
TRAIN_DIR = 'dataset/Training'
TEST_DIR = 'dataset/Testing'
MODEL_SAVE_PATH = 'models/model_vgg16.h5'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# Pastikan folder models/ ada
os.makedirs('models', exist_ok=True)

# 1. Preprocessing & Augmentasi Data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.1,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False # Jangan di-shuffle untuk confusion matrix
)

# Ambil daftar nama kelas
class_names = list(train_generator.class_indices.keys())
print("Kelas terdeteksi:", class_names)

# 2. Membangun Model VGG16 (Transfer Learning)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Bekukan layer bawaan VGG16
for layer in base_model.layers:
    layer.trainable = False

model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(class_names), activation='softmax') # Multi-kelas (4 kategori tumor)
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# 3. Proses Training
print("Memulai proses training...")
history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=EPOCHS
)

# Simpan Model
model.save(MODEL_SAVE_PATH)
print(f"Model berhasil disimpan di: {MODEL_SAVE_PATH}")

# 4. Evaluasi Visualisasi (Figure 1 & Figure 2)
# Figure 1: Akurasi dan Loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title('Akurasi Model')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Loss Model')
plt.savefig('Figure_1.png')
plt.close()
print("Figure_1.png (Grafik Akurasi) berhasil diekspor.")

# Figure 2: Confusion Matrix
Y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(Y_pred, axis=1)
y_true = test_generator.classes

cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Prediksi')
plt.ylabel('Aktual')
plt.title('Confusion Matrix - VGG16 Brain Tumor')
plt.savefig('Figure_2.png')
plt.close()
print("Figure_2.png (Confusion Matrix) berhasil diekspor.")

# Laporan Klasifikasi
print("\nClassification Report:\n", classification_report(y_true, y_pred_classes, target_names=class_names))