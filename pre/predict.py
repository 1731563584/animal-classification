import os
import json
import tensorflow as tf
from keras.src.legacy.preprocessing.image import ImageDataGenerator

# 加载模型
model = tf.keras.models.load_model('model.keras')

# 测试集路径
test_data_dir = './testdata'
results_file = './predictions.json'
true_labels_file = './results.json'

# 图像尺寸
img_width, img_height = 224, 224
batch_size = 32

# 预处理测试集数据
test_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.efficientnet.preprocess_input)

# 过滤掉非图片文件
file_paths = []
for fname in os.listdir(test_data_dir):
    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
        file_paths.append(os.path.join(test_data_dir, fname))

# 打印测试集图片数量
print(f"Number of test images: {len(file_paths)}")

# 使用 flow_from_dataframe 方法加载数据
import pandas as pd
test_df = pd.DataFrame({'filename': file_paths})
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='filename',
    y_col=None,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode=None,
    shuffle=False
)

# 预测测试集
predictions = model.predict(test_generator)
predicted_classes = tf.argmax(predictions, axis=1)

# 确保类名与训练集一致
class_names = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', 'ragno', 'sheep', 'squirrel']

# 保存结果到文件
results = {}
for i, pred_class in enumerate(predicted_classes):
    image_id = os.path.splitext(os.path.basename(file_paths[i]))[0]
    results[image_id] = class_names[pred_class]

with open(results_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=4)

# 加载真实标签
with open(true_labels_file, 'r', encoding='utf-8') as f:
    true_labels_dict = json.load(f)

# 提取真实标签
true_labels = []
for file_path in file_paths:
    image_id = os.path.splitext(os.path.basename(file_path))[0]
    true_label = true_labels_dict.get(image_id, 'null')  # 将未知标签设置为 'null'
    true_labels.append(true_label)

# 转换真实标签为索引
class_indices = {name: idx for idx, name in enumerate(class_names)}
class_indices['null'] = -1  # 添加 'null' 的索引
true_labels_indices = [class_indices[label] if label in class_indices else -1 for label in true_labels]

# 过滤掉无效的标签和预测值
valid_indices = [i for i, idx in enumerate(true_labels_indices) if idx != -1]
true_labels_indices = [true_labels_indices[i] for i in valid_indices]
predicted_classes = [predicted_classes[i] for i in valid_indices]

# 计算混淆矩阵
confusion_matrix = tf.math.confusion_matrix(true_labels_indices, predicted_classes, num_classes=len(class_names))

# 新增：计算精度
precision = tf.linalg.diag_part(confusion_matrix) / tf.reduce_sum(confusion_matrix, axis=0)

# 打印精度，避免 nan 值
print("Precision per class:")
for i, p in enumerate(precision):
    # 检查分母是否为零
    if tf.reduce_sum(confusion_matrix[:, i]) == 0:
        print(f"{class_names[i]}: 0.0000 (No samples for this class)")
    else:
        print(f"{class_names[i]}: {p.numpy():.4f}")

# 新增：统计每个类别的数量
print("\nClass counts:")
class_counts = {class_name: 0 for class_name in class_names}
class_counts['unknown'] = 0

# 统计真实标签中的每个类别的数量
for true_label in true_labels:
    if true_label in class_counts:
        class_counts[true_label] += 1
    else:
        class_counts['unknown'] += 1

# 输出每个类别的数量
for class_name, count in class_counts.items():
    print(f"{class_name}: {count}")

# 新增：绘制混淆矩阵
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix.numpy(), annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()

# 新增：计算并输出其他指标
from sklearn.metrics import classification_report

# 将 TensorFlow 张量转换为 NumPy 数组
true_labels_array = tf.convert_to_tensor(true_labels_indices).numpy()
predicted_classes_array = tf.convert_to_tensor(predicted_classes).numpy()

# 输出分类报告（包含精确率、召回率、F1 分数等）
report = classification_report(true_labels_array, predicted_classes_array, target_names=class_names, zero_division=0)
print("\nClassification Report:")
print(report)
