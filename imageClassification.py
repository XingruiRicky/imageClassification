import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# 数据集的路径
train_dir = 'dataset/train'
val_dir = 'dataset/val'
test_dir = 'dataset/test'

# 图像的尺寸和批大小
img_size = 224
batch_size = 32

# 训练数据的增强
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   rotation_range=30,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

# 验证和测试数据不需要增强，只需要进行相同的预处理
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# 从目录读取数据
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(img_size, img_size),
                                                    batch_size=batch_size,
                                                    class_mode='categorical')

val_generator = val_datagen.flow_from_directory(val_dir,
                                                target_size=(img_size, img_size),
                                                batch_size=batch_size,
                                                class_mode='categorical')

test_generator = test_datagen.flow_from_directory(test_dir,
                                                  target_size=(img_size, img_size),
                                                  batch_size=batch_size,
                                                  class_mode='categorical')

# 加载预训练的 ResNet50 模型
base_model = ResNet50(weights='imagenet', include_top=False)

# 添加自定义层
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

# 构建模型
model = Model(inputs=base_model.input, outputs=predictions)

# 冻结预训练层
for layer in base_model.layers:
    layer.trainable = False

# 编译模型
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
history = model.fit(train_generator, epochs=10, validation_data=val_generator)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(1, len(acc) + 1)

plt.figure(figsize=(12, 6))
plt.plot(epochs, acc, 'b', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

model.save('model.h5')

# 评估模型
loss, accuracy = model.evaluate(test_generator)
print(f"Test accuracy: {accuracy*100}%")



# CNN模型的层结构：
# 输入层：接收原始数据（例如图像）。
# 卷积层：通过卷积操作提取特征。
# 池化层（可选）：通常跟在卷积层后面，用于降低特征图的空间尺寸，减少参数数量和计算量，同时保留重要特征。
# 全连接层：在网络的末端，用于将学习到的特征映射到最终的输出（如分类任务中的类别）。
# 输出层：产生最终的结果，例如分类任务中的类别概率。
# 隐藏层的概念：
# 严格来说，池化层可以被视为隐藏层的一种，因为它不直接与最终的输出相连，而是在内部处理特征表示。
# 通常，术语“隐藏层”用于指代那些不是输入层或输出层的层，因为它们“隐藏”在网络的内部，执行特征提取和变换。

# 前向传播和反向传播：
# 在前向传播过程中，数据通过网络的每一层进行传递，并产生最终输出。
# 如果输出与预期不符（例如，分类错误），损失函数将计算一个损失值，表示预测的错误程度。
# 然后，在反向传播过程中，根据损失函数相对于网络参数的梯度，调整网络权重，以减少未来预测的损失。

# 损失函数和优化器：
# 损失函数定义了模型预测和实际标签之间的差距。常见的损失函数包括交叉熵损失（用于分类任务）和均方误差损失（用于回归任务）。
# 优化器决定了如何根据损失函数计算的梯度来更新网络的权重。常见的优化器包括梯度下降、Adam 和 RMSprop。