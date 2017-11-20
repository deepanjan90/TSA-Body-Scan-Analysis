import sklearn
import numpy as np
import os

from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Input, Maximum
from keras.optimizers import SGD

import exploratory_model.tsahelper as tsa


# In[4]:


# PARAMETERS
DATASET_PATH = "../../dataset/preprocessed-a3daps/"


# In[5]:


sublect_list = os.listdir(DATASET_PATH)
np.random.shuffle(sublect_list)


# In[113]:


# Load Keras' ResNet50 model that was pre-trained against the ImageNet database
INPUT_SHAPE = (660, 660, 3)
base_model = VGG16(include_top=False, input_shape=INPUT_SHAPE, pooling='None', weights='imagenet')


# In[114]:


layer_1 = base_model.layers[1]
block1_conv1_weight = layer_1.get_weights()


# In[116]:


FREEZE_LAYERS = 18

x = None
inputs = []
new_layer_1 = []
for i in range(4):
    x = Input(shape=INPUT_SHAPE, name=('input_' + str(i+1)))
    inputs.append(x)
    x = type(layer_1)(
            filters=layer_1.filters,
            strides=layer_1.strides,
            kernel_size=layer_1.kernel_size,
            activation=layer_1.activation,
            weights=block1_conv1_weight
        )(x)
    new_layer_1.append(x)


# In[117]:


x = Maximum()(new_layer_1)


# In[118]:


i = 2
for layer in base_model.layers[2:]:
    print("layer",i, end=';')
    i+=1
    if type(layer) == MaxPooling2D:
        print("MaxPooling2D", end=";")
        x = type(layer)(
                pool_size=layer.pool_size,
                strides=layer.strides,
                padding=layer.padding,
                data_format=layer.data_format
            )(x)
    elif type(layer) == Conv2D:
        print("Conv2D", end=";")
        x = type(layer)(
                filters=layer.filters,
                strides=layer.strides,
                kernel_size=layer.kernel_size,
                activation=layer.activation,
                weights=layer.get_weights()
            )(x)
    print(x)

x = Flatten()(x)

# # let's add a fully-connected layer
x = Dense(4096, activation='relu', kernel_initializer='truncated_normal', bias_initializer='random_uniform')(x)
x = Dense(4096, activation='relu', kernel_initializer='truncated_normal', bias_initializer='random_uniform')(x)

# # and a logistic layer, activation=sigmoid since we have multi label classification
predictions = Dense(17, activation='sigmoid', kernel_initializer='truncated_normal',
                    bias_initializer='random_uniform', name='output')(x)

# # this is the model we will train
model = Model(inputs=inputs, outputs=predictions)

# # Freeze first FREEZE_LAYERS in the model
for layer in model.layers[:FREEZE_LAYERS]:
    layer.trainable = False

# # Learning rate is changed to 0.001 = 1/10 of actual vgg16 since we are fine tuning
sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)

# # loss=binary_crossentropy for sigmoid units
model.compile(optimizer=sgd, loss='binary_crossentropy')


# In[119]:


def augment(image):
    X = []
    image[image<0]=0
    image = np.pad(image, ((0,0), (74,74)), 'edge')
    image = image.reshape((660,660,1))
    image = np.repeat(image, 3, axis=2)
    X.append(image)
    image = np.rot90(image)
    X.append(image)
    image = np.rot90(image)
    X.append(image)
    image = np.rot90(image)
    X.append(image)
    return X


# In[120]:


TRAIN_TEST_SPLIT = 0.8
sublect_list = os.listdir(DATASET_PATH)
file_list = sublect_list
number_of_train = int(len(file_list)*TRAIN_TEST_SPLIT)
TRAIN_SET_FILE_LIST = file_list[:number_of_train]
TEST_SET_FILE_LIST = file_list[number_of_train:]

def get_io_tensors(in_size):
    return [[] for i in range(in_size)], []

def data_generator(subject_file_list, in_size, batch_size):
    in_tensor, out_tensor = get_io_tensors(in_size)
    counter = 0
    for subject_image in subject_file_list:
        npset = np.load(os.path.join(DATASET_PATH, subject_image))
        images = npset[0]
        labels = [i[1] for i in npset[1]]

        for i in range(16):
            # Containes four angle each with 4 augmentations
            input_augments = []
            for j in range(4):
                input_augments.append(augment(images[i + j*16]))

            for j in range(4):
                for in_i in range(in_size):
                    in_tensor[in_i].append(input_augments[in_i][j])
                out_tensor.append(labels)

                counter += 1
                if counter%batch_size == 0:
                    for in_i in range(in_size):
                        in_tensor[in_i] = np.array(in_tensor[in_i])
                    out_tensor = np.array(out_tensor)
                    yield(in_tensor, out_tensor)
                    in_tensor, out_tensor = get_io_tensors(in_size)


step_per_epoch = 64
epochs = 917
# in_size = len(inputs)

# fig, ax = plt.subplots(4,4,figsize=(16,16))
# row = 0
# col = 0

# for in_tensors, out_tensors in data_generator(TRAIN_SET_FILE_LIST, in_size, step_per_epoch):
#     print('out', out_tensors.shape)
#     print(in_tensors[0].shape)

#     col = 0
#     for tensor in in_tensors:
#         ax[row, col].imshow(tensor[row])
#         col += 1
#     row += 1
#     if row == 4:
#         break


# In[ ]:


in_size = len(inputs)

model.fit_generator(
    data_generator(TRAIN_SET_FILE_LIST, in_size, step_per_epoch), step_per_epoch, epochs, verbose=1,
    use_multiprocessing=False, shuffle=True, initial_epoch=0)


# In[ ]:


# evaluate_generator(self, generator, steps, max_queue_size=10, workers=1, use_multiprocessing=False)
# model.fit(X_train, y_train, epochs=5, batch_size=2000)

# preds = model.predict(X_test)
# preds[preds>=0.5] = 1
# preds[preds<0.5] = 0
# score = compare preds and y_test

