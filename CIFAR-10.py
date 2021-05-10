# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow import keras


# %%
import matplotlib.pyplot as plt


# %%
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Conv2D, Dropout, MaxPooling2D
from tensorflow.keras.optimizers import SGD


# %%
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()


# %%
x_train_original = x_train
x_test_original = x_test
y_train_original = y_train
y_test_original = y_test


# %%
print(x_train)


# %%
#convert matrix values to float
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')


# %%
x_train = x_train/255
x_test = x_test/255


# %%
print(x_train)


# %%
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[y_train[i][0]])
plt.show()


# %%
print(y_train.shape)


# %%
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test,10)


# %%
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# %%
model = models.Sequential()
model.add(layers.Conv2D(32,(3,3), activation = 'relu', kernel_initializer='he_uniform', padding = 'same', 
                        input_shape = (32,32,3)))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(32,(3,3), activation = 'relu', kernel_initializer='he_uniform', padding = 'same'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Dropout(0.20))

model.add(layers.Conv2D(64,(3,3), activation = 'relu', kernel_initializer='he_uniform', padding = 'same'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(64,(3,3), activation = 'relu', kernel_initializer='he_uniform', padding = 'same'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(128,(3,3), activation = 'relu', kernel_initializer='he_uniform', padding = 'same'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(128,(3,3), activation = 'relu', kernel_initializer='he_uniform', padding = 'same'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Dropout(0.30))

model.add(layers.Flatten())
model.add(layers.Dense(512, activation = 'relu', kernel_initializer='he_uniform')) 
model.add(layers.Dropout(0.35))
model.add(layers.Dense(10, activation = 'softmax'))


# %%
model.summary()


# %%
opt = SGD(lr=0.001, momentum=0.9)


# %%
model.compile(optimizer = opt,
              loss = 'categorical_crossentropy',
              metrics = ['accuracy']
             )


# %%
history = model.fit(x_train, y_train, batch_size=32, epochs=15, validation_data=(x_test,y_test) )


# %%
batch_size = 32
data_generator = tf.keras.preprocessing.image.ImageDataGenerator(width_shift_range=0.1,
                                                                 height_shift_range=0.1,
                                                                 horizontal_flip=True)
train_data_gen = data_generator.flow(x_train, y_train, batch_size)
steps_per_epoch = x_train.shape[0] // batch_size
r = model.fit_generator(train_data_gen, validation_data=(x_test, y_test), steps_per_epoch=steps_per_epoch, epochs=15)


# %%
test_scores = model.evaluate(x_test, y_test, batch_size = 128, verbose = 0)
print('Test loss:', test_scores[0])
print('Test accuracy:', test_scores[1]*100.0)


# %%
import matplotlib.pyplot as plt

plt.plot(r.history['loss'], label='training')
plt.plot(r.history['val_loss'], label='testing')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()


# %%
# Plot accuracy per iteration
plt.plot(r.history['accuracy'], label='training')
plt.plot(r.history['val_accuracy'], label='testing')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()


# %%
model.save('my_model1')
del model
# Recreate the exact same model purely from the file:
model = keras.models.load_model('my_model1')


# %%
pred = model.predict(x_test, verbose=2)
y_predict = np.argmax(pred, axis=1)


# %%
#Confusion matrix
from sklearn.metrics import classification_report, confusion_matrix
 
for cm in range(10):
    print(cm, confusion_matrix(np.argmax(y_test,axis=1),y_predict)[cm].sum())
con_matrix = confusion_matrix(np.argmax(y_test,axis=1),y_predict)
print(con_matrix)
 
#Normalizing the values
con_matrix = con_matrix.astype('float') / con_matrix.sum(axis=1)[:, np.newaxis]

# Visualizing of confusion matrix
import seaborn as sn
import pandas  as pd
 
 
df_cm = pd.DataFrame(con_matrix, range(10),
                  range(10))
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)
sn.heatmap(df_cm, annot=True,annot_kws={"size": 12})
plt.show()


# %%
probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])


# %%
predictions = probability_model.predict(x_test)


# %%
predictions[0]


# %%
#The prediction for the image at index 1
np.argmax(predictions[0])


# %%
#The actual label value for the image at index 1
np.argmax(y_test[0])


# %%
#The corresponding label name at index 4 is:
print(labels[4])


# %%
from sklearn.metrics import classification_report
print(classification_report(np.argmax(y_test,axis=1),y_predict,target_names=class_names))


# %%
pred1 = model.predict(x_train, verbose=2)
y_predict1 = np.argmax(pred1, axis=1)


# %%
print(classification_report(np.argmax(y_train,axis=1),y_predict1,target_names=class_names))


# %%



# %%



# %%



# %%



