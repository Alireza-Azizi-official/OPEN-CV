import numpy as np
import pandas as pd
import obspy
import keras
import time
from keras_tqdm import TQDMNotebookCallback
from tqdm import tnrange, tqdm_notebook
import matplotlib.pyplot as plt
from keras.models import Sequential, Model, clone_model
from keras.layers import Conv2D, Dense, Activation, Flatten, Dropout, Input
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
import tensorflow as tf
from obspy.io.segy.segy import _read_segy
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import VGG16
from keras import backend as K

np.random.seed(42)
patch_size = 64  # for ResNet50 put 244
batch_size = 256
num_channels = 1
num_classes = 9
all_examples = 158812
num_examples = 7500
epochs = 20
steps = 450
sampler = list(range(all_examples))

opt = 'adam'
lossfkt = ['categorical_crossentropy']
metrica = ['mae', 'acc']
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
# It should say GPU here. Otherwise your model will run slow.
filename = 'data/Dutch Government_F3_entire_8bit seismic.segy'

t0 = time.time()
stream0 = _read_segy(filename, headonly=True)
print('--> data read in {:.1f} sec'.format(time.time()-t0))  

t0 = time.time()

labeled_data = np.stack(t.data for t in stream0.traces if t.header.for_3d_poststack_data_this_field_is_for_in_line_number == 339).T
inline_data = np.stack(t.data for t in stream0.traces if t.header.for_3d_poststack_data_this_field_is_for_in_line_number == 500).T
xline_data = np.stack(t.data for t in stream0.traces if t.header.for_3d_poststack_data_this_field_is_for_cross_line_number == 500).T

print('--> created slices in {:.1f} sec'.format(time.time()-t0))

def patch_extractor2D(img, mid_x, mid_y, patch_size, dimensions=1):
    try:
        x, y, c = img.shape
    except ValueError:
        x, y = img.shape
        c = 1
    patch = np.pad(img, patch_size // 2, 'constant', constant_values=0)[mid_y:mid_y + patch_size, mid_x:mid_x + patch_size]
    if c != dimensions:
        tmp_patch = np.zeros((patch_size, patch_size, dimensions))
        for uia in range(dimensions):
            tmp_patch[:, :, uia] = patch
        return tmp_patch
    return patch

image = np.random.rand(10, 10) // .1
print(image)

patch_extractor2D(image, 10, 10, 4, 1)

def acc_assess(data, loss=['categorical_crossentropy'], metrics=['acc']):
    if not isinstance(loss, list):
        try:
            loss = [loss]
        except:
            raise("Loss must be list.")
    if not isinstance(metrics, list):
        try:
            metrics = [metrics]
        except:
            raise("Metrics must be list.")
    out = 'The test loss is {:.3f}\n'.format(data[0])
    for i, metric in enumerate(metrics):
        if metric in 'mae':
            out += "The total mean error on the test is {:.3f}\n".format(data[i + 1])
        if metric in 'accuracy':
            out += "The test accuracy is {:.1f}%\n".format(data[i + 1] * 100)
    return out

print(acc_assess([1, 2, 3], 'bla', ["acc", "mae"]))
labels = pd.read_csv('data/classification.ixz', delimiter=" ", names=["Inline", "Xline", "Time", "Class"])
labels.describe()

labels["Xline"] -= 300 - 1
labels["Time"] = labels["Time"] // 4
labels.describe()
labeled_data.shape

fig2 = plt.figure(figsize=(15.0, 10.0))
vml = np.percentile(labeled_data, 99)
img1 = plt.imshow(labeled_data, cmap="Greys", vmin=-vml, vmax=vml, aspect='auto')
plt.yticks(np.arange(0, 462, 100), np.arange(0, 462 * 4, 400))
plt.xlabel('Trace Location')
plt.ylabel('Time [ms]')
plt.savefig('labeled_data.png', bbox_inches='tight')
plt.show()

fig2 = plt.figure(figsize=(15.0, 10.0))
vmx = np.percentile(xline_data, 99)
plt.imshow(xline_data, cmap="Greys", vmin=-vmx, vmax=vmx, aspect='auto')
plt.yticks(np.arange(0, 462, 100), np.arange(0, 462 * 4, 400))
plt.xlabel('Trace Location')
plt.ylabel('Time [ms]')
plt.savefig('xline_data.png', bbox_inches='tight')
plt.show()

fig2 = plt.figure(figsize=(15.0, 10.0))
vmy = np.percentile(inline_data, 99)
plt.imshow(inline_data, cmap="Greys", vmin=-vmy, vmax=vmy, aspect='auto')
plt.yticks(np.arange(0, 462, 100), np.arange(0, 462 * 4, 400))
plt.xlabel('Trace Location')
plt.ylabel('Time [ms]')
plt.savefig('inline_data.png', bbox_inches='tight')
plt.show()

fig2 = plt.figure(figsize=(15.0, 10.0))
img2 = plt.imshow(labeled_data, cmap="Greys", vmin=-vml, vmax=vml, aspect='auto')
img1 = plt.scatter(labels["Xline"], labels[["Time"]], c=labels[["Class"]], cmap='Dark2', alpha=0.03)
plt.yticks(np.arange(0, 462, 100), np.arange(0, 462 * 4, 400))
plt.xlabel('Trace Location')
plt.ylabel('Time [ms]')
plt.savefig('label.png', bbox_inches='tight')
plt.show()

train_data, test_data, train_samples, test_samples = train_test_split(
    labels, sampler, random_state=42)
print(train_data.shape, test_data.shape)

class SeismicSequence(keras.utils.Sequence):
    def __init__(self, img, x_set, t_set, y_set, patch_size, batch_size, dimensions):
        self.slice = img
        self.X, self.t = x_set, t_set
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.dimensions = dimensions
        self.label = y_set

    def __len__(self):
        return len(self.X) // self.batch_size

    def __getitem__(self, idx):
        sampler = np.random.permutation(len(self.X))
        samples = sampler[idx * self.batch_size:(idx + 1) * self.batch_size]
        labels = keras.utils.to_categorical(self.label[samples], num_classes=9)
        if self.dimensions == 1:
            return np.expand_dims(np.array([patch_extractor2D(self.slice, self.X[x], self.t[x], self.patch_size, self.dimensions) for x in samples]), axis=4), labels
        else:
            return np.array([patch_extractor2D(self.slice, self.X[x], self.t[x], self.patch_size, self.dimensions) for x in samples]), labels

earlystop1 = keras.callbacks.EarlyStopping(monitor='val_loss',
                                           min_delta=0,
                                           patience=3,
                                           verbose=0, mode='auto')

earlystop2 = keras.callbacks.EarlyStopping(monitor='val_acc',
                                           min_delta=0,
                                           patience=3,
                                           verbose=0, mode='auto')

checkpoint = keras.callbacks.ModelCheckpoint('tmp.h5',
                                             monitor='val_loss',
                                             verbose=0,
                                             save_best_only=False,
                                             save_weights_only=False,
                                             mode='auto',
                                             period=1)

callbacklist = [TQDMNotebookCallback(leave_inner=True, leave_outer=True), earlystop1, earlystop2, checkpoint]

tf.logging.set_verbosity(tf.logging.ERROR)

model_vanilla = Sequential()
model_vanilla.add(Conv2D(50, (5, 5), padding='same', input_shape=(patch_size, patch_size, 1), strides=(4, 4), data_format="channels_last", name='conv_layer1'))
model_vanilla.add(BatchNormalization())
model_vanilla.add(Activation('relu'))
model_vanilla.add(Conv2D(50, (3, 3), strides=(2, 2), padding='same', name='conv_layer2'))
model_vanilla.add(Dropout(0.5))
model_vanilla.add(BatchNormalization())
model_vanilla.add(Activation('relu'))
model_vanilla.add(Conv2D(50, (3, 3), strides=(2, 2), padding='same', name='conv_layer3'))
model_vanilla.add(Dropout(0.4))
model_vanilla.add(BatchNormalization())
model_vanilla.add(Activation('relu'))
model_vanilla.add(Conv2D(50, (3, 3), strides=(2, 2), padding='same', name='conv_layer4'))
model_vanilla.add(BatchNormalization())
model_vanilla.add(Activation('relu'))
model_vanilla.add(Conv2D(50, (3, 3), strides=(2, 2), padding='same', name='conv_layer5'))
model_vanilla.add(BatchNormalization())
model_vanilla.add(Activation('relu'))
model_vanilla.add(Flatten())
model_vanilla.add(Dense(1024, activation='relu'))
model_vanilla.add(Dropout(0.5))
model_vanilla.add(Dense(num_classes, activation='softmax'))
model_vanilla.compile(optimizer=opt,
                      loss=lossfkt[0],
                      metrics=metrica)
model_vanilla.summary()

model_vanilla.fit_generator(SeismicSequence(labeled_data, train_data["Xline"], train_data["Time"], train_data["Class"], patch_size, batch_size, 1),
                            epochs=epochs,
                            verbose=0,
                            steps_per_epoch=steps,
                            validation_data=SeismicSequence(labeled_data, test_data["Xline"], test_data["Time"], test_data["Class"], patch_size, batch_size, 1),
                            validation_steps=100,
                            use_multiprocessing=True,
                            workers=4,
                            max_queue_size=10,
                            callbacks=callbacklist)

results = model_vanilla.evaluate_generator(SeismicSequence(labeled_data, test_data["Xline"], test_data["Time"], test_data["Class"], patch_size, batch_size, 1),
                                           steps=100, use_multiprocessing=True, workers=4, max_queue_size=10)
print(acc_assess(results, lossfkt, metrica))

model_VGG16 = VGG16(include_top=True, weights=None, input_tensor=None, input_shape=(patch_size, patch_size, 1), pooling=None, classes=num_classes)

model_VGG16.compile(optimizer=opt,
                    loss=lossfkt[0],
                    metrics=metrica)
model_VGG16.summary()

model_VGG16.fit_generator(SeismicSequence(labeled_data, train_data["Xline"], train_data["Time"], train_data["Class"], patch_size, batch_size, 1),
                          epochs=epochs,
                          verbose=0,
                          steps_per_epoch=steps,
                          validation_data=SeismicSequence(labeled_data, test_data["Xline"], test_data["Time"], test_data["Class"], patch_size, batch_size, 1),
                          validation_steps=100,
                          use_multiprocessing=True,
                          workers=4,
                          max_queue_size=10,
                          callbacks=callbacklist)

results = model_VGG16.evaluate_generator(SeismicSequence(labeled_data, test_data["Xline"], test_data["Time"], test_data["Class"], patch_size, batch_size, 1),
                                         steps=100, use_multiprocessing=True, workers=4, max_queue_size=10)
print(acc_assess(results, lossfkt, metrica))

for layer in model_VGG16.layers:
    layer.trainable = False
weights = model_VGG16.get_weights()
print(weights)

inp = Input(shape=(patch_size, patch_size, 1), name='image_input')
last_layer = model_VGG16.get_layer('block5_pool').output
x = Flatten(name='flatten')(last_layer)
x = Dense(1024, activation='relu', name='fc1')(x)
x = Dropout(0.5)(x)
out = Dense(num_classes, activation='softmax', name='output')(x)
custom_vgg_model = Model(inp, out)

custom_vgg_model.layers[1].set_weights(weights)
custom_vgg_model.compile(optimizer=opt,
                         loss=lossfkt[0],
                         metrics=metrica)

custom_vgg_model.fit_generator(SeismicSequence(labeled_data, train_data["Xline"], train_data["Time"], train_data["Class"], patch_size, batch_size, 1),
                               epochs=epochs,
                               verbose=0,
                               steps_per_epoch=steps,
                               validation_data=SeismicSequence(labeled_data, test_data["Xline"], test_data["Time"], test_data["Class"], patch_size, batch_size, 1),
                               validation_steps=100,
                               use_multiprocessing=True,
                               workers=4,
                               max_queue_size=10,
                               callbacks=callbacklist)

results = custom_vgg_model.evaluate_generator(SeismicSequence(labeled_data, test_data["Xline"], test_data["Time"], test_data["Class"], patch_size, batch_size, 1),
                                              steps=100, use_multiprocessing=True, workers=4, max_queue_size=10)
print(acc_assess(results, lossfkt, metrica))

     

t_max, y_max = xline_data.shape

half_patch = patch_size//2

resnet_predx = np.full_like(xline_data,-1)

for space in tqdm_notebook(range(y_max),desc='Space'):
    for depth in tqdm_notebook(range(t_max),leave=False, desc='Time'):
        resnet_predx[depth,space] = np.argmax(resnet.predict(np.expand_dims(patch_extractor2D(xline_data,space,depth,patch_size,3), axis=0)))
     

np.save('resnet_predx.npy',resnet_predx,allow_pickle=False)
     

plt.imshow(resnet_predx)
     

fig2 = plt.figure(figsize=(15.0, 10.0))
img2 = plt.imshow(xline_data, cmap="Greys", vmin=-vmx, vmax=vmx, aspect='auto')
img1 = plt.imshow(resnet_predx, aspect='auto', cmap="Dark2", alpha=0.8)
plt.savefig('resnet_x.png', bbox_inches='tight')
plt.show()
     

t_max, y_max = inline_data.shape

half_patch = patch_size//2

resnet_predi = np.full_like(inline_data,-1)

for space in tqdm_notebook(range(y_max-400,y_max-300),desc='Space'):
    for depth in tqdm_notebook(range(t_max-400,t_max-300),leave=False, desc='Time'):
        resnet_predi[depth,space] = np.argmax(resnet.predict(np.expand_dims(patch_extractor2D(inline_data,space,depth,patch_size,3), axis=0)))
     

np.save('resnet_predi.npy',resnet_predi,allow_pickle=False)
     

plt.imshow(resnet_predi)
     

fig2 = plt.figure(figsize=(15.0, 10.0))
img2 = plt.imshow(inline_data, cmap="Greys", vmin=-vmy, vmax=vmy, aspect='auto')
img1 = plt.imshow(resnet_predi, aspect='auto', cmap="Dark2", alpha=0.8)
plt.savefig('resnet_i.png', bbox_inches='tight')
plt.show()
     

print(res_hist.history.keys())
plt.plot(res_hist.history['acc'])
plt.plot(res_hist.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
     

# summarize history for loss
plt.plot(res_hist.history['loss'])
plt.plot(res_hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
     

plot_model(resnet, to_file='model_resnet.png')
plot_model(resnet, to_file='model_resnet_shapes.png', show_shapes=True)
SVG(model_to_dot(resnet).create(prog='dot', format='svg'))