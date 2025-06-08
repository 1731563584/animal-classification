import tensorflow as tf
from keras.models import Sequential
from keras.src.ops import Flip
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from keras.applications import EfficientNetV2B0
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# 数据集路径
train_data_dir = '../traindata'
img_width, img_height = 224, 224
batch_size = 32

# ----------------------:1：优化数据增强策略 ----------------------
from albumentations import Compose, RandomRotate90, Transpose, ShiftScaleRotate, Blur, OpticalDistortion, \
    GridDistortion, HueSaturationValue


# 定义猫类专用增强策略（保护关键特征）
def cat_augmentation(image):
    transform = Compose([
        RandomRotate90(p=0.5),  # 随机旋转
        Flip(p=0.5),  # 随机翻转
        Transpose(p=0.5),  # 随机转置
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.8),  # 缩放和旋转，限制角度
        Blur(blur_limit=2, p=0.2),  # 轻微模糊
        HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5)  # 调整色彩
    ])
    return transform(image=image)['image']


# 定义狗类专用增强策略（保持纹理）
def dog_augmentation(image):
    transform = Compose([
        RandomRotate90(p=0.5),  # 随机旋转
        Flip(p=0.5),  # 随机翻转
        Transpose(p=0.5),  # 随机转置
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.8),  # 缩放和旋转，限制角度
        OpticalDistortion(p=0.2),  # 光学畸变，轻微扭曲
        GridDistortion(p=0.2),  # 网格畸变，轻微变形
        HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=5, p=0.5)  # 轻微调整色彩
    ])
    return transform(image=image)['image']


# 自定义数据生成器，支持类别特定增强
class CustomImageDataGenerator(ImageDataGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cat_class_index = None
        self.dog_class_index = None

    def set_class_indices(self, cat_class_index, dog_class_index):
        self.cat_class_index = cat_class_index
        self.dog_class_index = dog_class_index

    def flow_from_directory(self, directory, *args, **kwargs):
        generator = super().flow_from_directory(directory, *args, **kwargs)
        self.set_class_indices(generator.class_indices['cat'], generator.class_indices['dog'])
        return generator

    def __next__(self):
        X_batch, y_batch = super().__next__()
        for i in range(len(y_batch)):
            if y_batch[i][self.cat_class_index] == 1:  # 猫类增强
                X_batch[i] = cat_augmentation(X_batch[i])
            elif y_batch[i][self.dog_class_index] == 1:  # 狗类增强
                X_batch[i] = dog_augmentation(X_batch[i])
        return X_batch, y_batch


# 使用自定义数据生成器
train_datagen = CustomImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
    rotation_range=15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.9, 1.1],
    validation_split=0.2,
    fill_mode='reflect'
)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    interpolation='bilinear'
)

validation_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    interpolation='bilinear'
)

# ---------------------- 2：添加类别权重平衡 ----------------------
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights = dict(enumerate(class_weights))

cat_class_index = train_generator.class_indices['cat']
dog_class_index = train_generator.class_indices['dog']

print("Modified Class weights:", class_weights)

# ----------------------:3：增强模型结构 ----------------------
base_model = EfficientNetV2B0(
    weights='imagenet',
    include_top=False,
    input_shape=(img_width, img_height, 3)
)

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.5),
    Dense(512, activation='swish'),
    BatchNormalization(),
    Dropout(0.3),  # 调整dropout率
    Dense(256, activation='swish'),
    Dense(10, activation='softmax')
])

# 第一阶段：冻结基础模型训练
base_model.trainable = False

# ---------------------- 4：优化训练配置 ----------------------
# 猫狗专属监控指标
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(class_id=cat_class_index, name='cat_recall'),
        tf.keras.metrics.Precision(class_id=cat_class_index, name='cat_precision'),
        tf.keras.metrics.Recall(class_id=dog_class_index, name='dog_recall'),
        tf.keras.metrics.Precision(class_id=dog_class_index, name='dog_precision')
    ]
)

# 第一阶段训练
print("\n--- Phase 1: Frozen Base Model Training ---")
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=10,
    validation_data=validation_generator,
    class_weight=class_weights,  # 应用动态调整的类别权重
    verbose=1
)

# ---------------------- 5：精细化微调策略 ----------------------
# 解冻最后15层（不包括BN层）
base_model.trainable = True
for layer in base_model.layers[:-15]:
    layer.trainable = False
for layer in base_model.layers[-15:]:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

# 配置回调函数
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_recall',  # 监控召回率
    factor=0.1,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

# 修改: 增加早停机制，重点关注猫类召回率
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_cat_recall',  # 监控猫类召回率
    patience=5,
    mode='max',
    restore_best_weights=True
)

# 第二阶段训练
print("\n--- Phase 2: Fine-tuning ---")
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=10,
    validation_data=validation_generator,
    callbacks=[reduce_lr, early_stopping],  # 早停机制
    class_weight=class_weights,
    verbose=1
)

# ---------------------- 6：模型保存与评估 ----------------------
model.save('model.keras')

# 可视化训练过程
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.plot(history.history['recall'], label='Train Recall')
plt.plot(history.history['val_recall'], label='Val Recall')
plt.title('Training Metrics')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Training Loss')
plt.legend()
plt.savefig('training_curves.png')
plt.show()
