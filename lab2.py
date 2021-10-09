import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import losses
import matplotlib.pyplot as plt
import random

from tensorflow.python.keras.layers.core import Dropout
from tensorflow.python.keras.layers.normalization.batch_normalization import BatchNormalization

import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context
random.seed(1618)
np.random.seed(1618)
#tf.set_random_seed(1618)   # Uncomment for TF1.
tf.random.set_seed(1618)

#tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#=========================<Classifier Functions>================================

def guesserClassifier(xTest, NUM_CLASSES):
    ans = []
    for entry in xTest:
        pred = [0] * NUM_CLASSES
        pred[random.randint(0, 9)] = 1
        ans.append(pred)
    return np.array(ans)


def buildTFNeuralNet(x, y, meta_data, eps = 6):
    DATASET,ALGORITHM,NUM_CLASSES,IH,IW,IZ,IS = meta_data
    checkpoint_path = "model_weights/"+DATASET+"/ann_cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    print("Saving checkpoints to: ", checkpoint_dir)

    # Create a callback that saves the model's weights
    cp_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        verbose=1
    )

    model = tf.keras.models.Sequential([
        layers.Dense(2*IS//3, activation='relu'),
        layers.Dropout(.2),
        layers.Dense(2**2*IS//(3**2), activation='relu'),
        layers.Dropout(.2),
        layers.Dense(NUM_CLASSES)
    ])
    
    if (os.path.exists(checkpoint_dir)):
        model.load_weights(checkpoint_path).expect_partial()
    else:
        loss = losses.MeanSquaredError()
        model.compile(optimizer='adam',loss=loss,metrics=['accuracy'])
        model.fit(x,y,epochs=eps,callbacks=[cp_cb])
    return model


def buildTFConvNet(x, y, meta_data,eps = 20, dropout = True, dropRate = 0.25):
    DATASET,ALGORITHM,NUM_CLASSES,IH,IW,IZ,IS = meta_data
    checkpoint_path = "model_weights/conv_"+DATASET+"/ann_cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    print("Saving checkpoints to: ", checkpoint_dir)

    # Create a callback that saves the model's weights
    cp_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        verbose=1
    )
    model = tf.keras.models.Sequential()
    if (DATASET == 'mnist_d' or DATASET == 'cifar_10' or DATASET == 'cifar_100_c'):
            
        model.add(layers.Conv2D(32,(3,3),input_shape=(IW,IH,IZ),padding='same', data_format='channels_last',activation='relu'))
        model.add(layers.MaxPool2D(pool_size=(2,2)))
        if dropout: model.add(layers.Dropout(dropRate))
        
        model.add(layers.Conv2D(64,(3,3),padding='same',activation='relu'))
        model.add(layers.MaxPool2D(pool_size=(2,2)))
        if dropout: model.add(layers.Dropout(dropRate))
        
        model.add(layers.Conv2D(128,(3,3),padding='same',activation='relu'))
        model.add(layers.MaxPool2D(pool_size=(2,2)))
        if dropout: model.add(layers.Dropout(dropRate))
        
        model.add(layers.Conv2D(256,(3,3),padding='same',activation='relu'))
        model.add(layers.MaxPool2D(pool_size=(2,2)))
        if dropout: model.add(layers.Dropout(dropRate))
        
        model.add(layers.Flatten())
        model.add(layers.Dense(128,activation='relu'))
        if dropout: model.add(layers.Dropout(dropRate))
        model.add(layers.Dense(64,activation='relu'))
        if dropout: model.add(layers.Dropout(dropRate))
        
        model.add(layers.Dense(NUM_CLASSES, activation='softmax'))
    elif (DATASET == 'mnist_f'):
        dropRate = .2
        model.add(layers.Conv2D(32,(3,3),input_shape=(IW,IH,IZ),padding='same', data_format='channels_last',activation='relu'))
        model.add(layers.MaxPool2D(pool_size=(2,2)))
        if dropout: model.add(layers.Dropout(dropRate))
        
        model.add(layers.Conv2D(64,(3,3),padding='same',activation='relu'))
        model.add(layers.MaxPool2D(pool_size=(2,2)))
        if dropout: model.add(layers.Dropout(dropRate))
        
        model.add(layers.Conv2D(128,(3,3),padding='same',activation='relu'))
        model.add(layers.MaxPool2D(pool_size=(2,2)))
        if dropout: model.add(layers.Dropout(dropRate))
        
        model.add(layers.Conv2D(256,(3,3),padding='same',activation='relu'))
        model.add(layers.MaxPool2D(pool_size=(2,2)))
        if dropout: model.add(layers.Dropout(dropRate))
        
        model.add(layers.Flatten())
        model.add(layers.Dense(128,activation='relu'))
        if dropout: model.add(layers.Dropout(dropRate))
        model.add(layers.Dense(64,activation='relu'))
        if dropout: model.add(layers.Dropout(dropRate))
        
        model.add(layers.Dense(NUM_CLASSES, activation='softmax'))
        
    else:
        dropRate = .25    
        eps = 30 
        model.add(layers.Conv2D(32,(3,3),input_shape=(IW,IH,IZ),padding='same', data_format='channels_last',activation='relu'))
        model.add(layers.MaxPool2D(pool_size=(2,2)))
        if dropout: model.add(layers.Dropout(dropRate))
        
        model.add(layers.Conv2D(64,(3,3),padding='same',activation='relu'))
        model.add(layers.MaxPool2D(pool_size=(2,2)))
        if dropout: model.add(layers.Dropout(dropRate))
        
        model.add(layers.Conv2D(128,(3,3),padding='same',activation='relu'))
        model.add(layers.MaxPool2D(pool_size=(2,2)))
        if dropout: model.add(layers.Dropout(dropRate))
        
        model.add(layers.Flatten())
        model.add(layers.Dense(256,activation='relu'))
        if dropout: model.add(layers.Dropout(dropRate))
        model.add(layers.Dense(128,activation='relu'))
        if dropout: model.add(layers.Dropout(dropRate))
        
        model.add(layers.Dense(NUM_CLASSES, activation='softmax'))
        
    if (os.path.exists(checkpoint_dir)):
        model.load_weights(checkpoint_path).expect_partial()
    else:
        loss = losses.CategoricalCrossentropy()
        model.compile(optimizer='adam',loss=loss,metrics=['accuracy'])
        model.fit(x,y,epochs=eps,callbacks=[cp_cb])
    return model

#=========================<Pipeline Functions>==================================

def getRawData(meta_data):
    DATASET,ALGORITHM,NUM_CLASSES,IH,IW,IZ,IS = meta_data
    if DATASET == "mnist_d":
        mnist = tf.keras.datasets.mnist
        (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    elif DATASET == "mnist_f":
        mnist = tf.keras.datasets.fashion_mnist
        (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    elif DATASET == "cifar_10":
        (xTrain, yTrain), (xTest, yTest) = keras.datasets.cifar10.load_data()
    elif DATASET == "cifar_100_f":
        (xTrain, yTrain), (xTest, yTest) = tf.keras.datasets.cifar100.load_data(label_mode="fine")
    elif DATASET == "cifar_100_c":
        (xTrain, yTrain), (xTest, yTest) = tf.keras.datasets.cifar100.load_data(label_mode="coarse")
    else:
        raise ValueError("Dataset not recognized.")
    print("Dataset: %s" % DATASET)
    print("Shape of xTrain dataset: %s." % str(xTrain.shape))
    print("Shape of yTrain dataset: %s." % str(yTrain.shape))
    print("Shape of xTest dataset: %s." % str(xTest.shape))
    print("Shape of yTest dataset: %s." % str(yTest.shape))
    return ((xTrain, yTrain), (xTest, yTest))



def preprocessData(raw,meta_data,random_crop=False):
    DATASET,ALGORITHM,NUM_CLASSES,IH,IW,IZ,IS = meta_data
    ((xTrain, yTrain), (xTest, yTest)) = raw

    xTest = (xTest-np.min(xTest))/(np.max(xTest)-np.min(xTest))  
    xTrain = (xTrain-np.min(xTrain))/(np.max(xTrain)-np.min(xTrain))  
    if ALGORITHM != "tf_conv":
        xTrainP = xTrain.reshape((xTrain.shape[0], IS))
        xTestP = xTest.reshape((xTest.shape[0], IS))
    else:
        xTrainP = xTrain.reshape((xTrain.shape[0], IH, IW, IZ))
        xTestP = xTest.reshape((xTest.shape[0], IH, IW, IZ))
    yTrainP = to_categorical(yTrain, NUM_CLASSES)
    yTestP = to_categorical(yTest, NUM_CLASSES)
    
    # if (random_crop and (DATASET != "mnist_d" or DATASET != "mnist_f")):
    if (random_crop == False):
        print("cropping")
        images = xTrainP.reshape((xTrainP.shape[0],IH,IW,IZ))
        xFlips = tf.image.random_flip_left_right(images)
        xCrops = tf.image.random_crop(images, (xTrainP.shape[0],7*IH//10,7*IW//10,IZ))
        xCrops = tf.image.resize_with_crop_or_pad(xCrops,IH,IW)
        cats = yTrainP
        xTrainP = xFlips
        xTrainP = tf.concat([xTrainP,xCrops],0)
        yTrainP = tf.concat([yTrainP,cats],0)
        
        

    print("New shape of xTrain dataset: %s." % str(xTrainP.shape))
    print("New shape of xTest dataset: %s." % str(xTestP.shape))
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    print("New shape of yTest dataset: %s." % str(yTestP.shape))
    return ((xTrainP, yTrainP), (xTestP, yTestP))



def trainModel(data,meta_data):
    DATASET,ALGORITHM,NUM_CLASSES,IH,IW,IZ,IS = meta_data
    xTrain, yTrain = data
    if ALGORITHM == "guesser":
        return None   # Guesser has no model, as it is just guessing.
    elif ALGORITHM == "tf_net":
        print("Building and training TF_NN.")
        return buildTFNeuralNet(xTrain, yTrain,meta_data)
    elif ALGORITHM == "tf_conv":
        print("Building and training TF_CNN.")
        return buildTFConvNet(xTrain, yTrain,meta_data=meta_data)
    else:
        raise ValueError("Algorithm not recognized.")



def runModel(data, model,meta_data):
    DATASET,ALGORITHM,NUM_CLASSES,IH,IW,IZ,IS = meta_data
    if ALGORITHM == "guesser":
        return guesserClassifier(data)
    elif ALGORITHM == "tf_net":
        print("Testing TF_NN.")
        preds = model.predict(data)
        for i in range(preds.shape[0]):
            oneHot = [0] * NUM_CLASSES
            oneHot[np.argmax(preds[i])] = 1
            preds[i] = oneHot
        return preds
    elif ALGORITHM == "tf_conv":
        print("Testing TF_CNN.")
        preds = model.predict(data)
        for i in range(preds.shape[0]):
            oneHot = [0] * NUM_CLASSES
            oneHot[np.argmax(preds[i])] = 1
            preds[i] = oneHot
        return preds
    else:
        raise ValueError("Algorithm not recognized.")



def evalResults(data, preds,meta_data):
    DATASET,ALGORITHM,NUM_CLASSES,IH,IW,IZ,IS = meta_data
    xTest, yTest = data
    acc = 0
    for i in range(preds.shape[0]):
        if np.array_equal(preds[i], yTest[i]):   acc = acc + 1
    accuracy = acc / preds.shape[0]
    print("Classifier algorithm: %s" % ALGORITHM)
    print("Classifier accuracy: %f%%" % (accuracy * 100))
    print()
    return (accuracy*100)



#=========================<Main>================================================

def main():
    
    mnist_ann = set_meta_data('tf_conv','mnist_d')
    mnist_f_ann = set_meta_data('tf_conv','mnist_f')
    cf_10_ann = set_meta_data('tf_conv','cifar_10')
    cf_100f_ann = set_meta_data('tf_conv','cifar_100_f')
    cf_100c_ann = set_meta_data('tf_conv','cifar_100_c')
    accuracies = []
    accuracies.append(run_nn(mnist_ann))
    accuracies.append(run_nn(mnist_f_ann))
    accuracies.append(run_nn(cf_10_ann))
    accuracies.append(run_nn(cf_100c_ann))
    accuracies.append(run_nn(cf_100f_ann))
    
    plot_bar(accuracies, ['MNIST_D','MNIST_F','CIFAR_10','CIFAR_100_C','CIFAR_100_F'])

    
def set_meta_data(alg,dataset):
    if dataset == "mnist_d":
        NUM_CLASSES = 10
        IH = 28
        IW = 28
        IZ = 1
    elif dataset == "mnist_f":
        NUM_CLASSES = 10
        IH = 28
        IW = 28
        IZ = 1
    elif dataset == "cifar_10":
        NUM_CLASSES = 10
        IH = 32
        IW = 32
        IZ = 3
    elif dataset == "cifar_100_f":
        NUM_CLASSES = 100
        IH = 32
        IW = 32
        IZ = 3
    elif dataset == "cifar_100_c":
        NUM_CLASSES = 20
        IH = 32
        IW = 32
        IZ = 3
    
    IS = IH*IW*IZ
    return [dataset,alg,NUM_CLASSES,IH,IW,IZ,IS]

def run_nn(meta_data):
    DATASET,alg,NUM_CLASSES,IH,IW,IZ,IS = meta_data
    raw = getRawData(meta_data)
    data = preprocessData(raw,meta_data,True)
    model = trainModel(data[0],meta_data)
    preds = runModel(data[1][0], model,meta_data)
    acc = evalResults(data[1], preds,meta_data)
    return acc

def plot_bar(x,x_labels):
    x_pos = np.arange(len(x))
    x_pos = [x for x in x_pos]
    print(x_labels)
    print(x)
    plt.xticks(x_pos, x_labels)
    plt.xlabel("Dataset")
    plt.title("Accuracy by Dataset")
    plt.ylabel("Accuracy")
    plt.bar(x_pos,x)

    plt.savefig('barplot.png')
if __name__ == '__main__':
    main()
