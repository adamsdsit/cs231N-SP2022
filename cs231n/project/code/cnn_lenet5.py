import utils
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

def convolutional_model(input_shape):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> DENSE -> DENSE -> DENSE

    Arguments:
    input_img -- input dataset, of shape (input_shape)

    Returns:
    model -- TF Keras model (object containing the information for the entire training process)
    """
    input_img = tf.keras.Input(shape=input_shape)
    # YOUR CODE STARTS HERE
    ## CONV2D: 6 filters 5x5
    Z1 = tf.keras.layers.Conv2D(6, (5, 5), strides=(1,1), padding='same', activation='tanh')(input_img)
    ## MAXPOOL: window 2x2
    P1 = tf.keras.layers.AvgPool2D(pool_size=(2, 2), strides=(2,2))(Z1)
    ## CONV2D: 16 filters 5x5
    Z2 = tf.keras.layers.Conv2D(16, (5, 5), activation='tanh', padding='valid')(P1)
    ## MAXPOOL: window 2x2
    P2 = tf.keras.layers.AvgPool2D(pool_size=(2, 2), strides=(2,2))(Z2)
    ## FLATTEN
    F = tf.keras.layers.Flatten()(P2)
    ## Dense layer
    D1 = tf.keras.layers.Dense(120, activation='tanh')(F)
    ## Dense layer
    D2 = tf.keras.layers.Dense(84, activation='tanh')(D1)
    ## Dense layer
    outputs = tf.keras.layers.Dense(7, activation='softmax')(D2)

    # YOUR CODE ENDS HERE
    model = tf.keras.Model(inputs=input_img, outputs=outputs)
    return model

def process_images(image, label):
    # Normalize images to have a mean of 0 and standard deviation of 1
    image = tf.image.per_image_standardization(image)
    # Resize images from 500x500 to 256x256
    image = tf.image.resize(image, (256,256))
    return image, label

def main(plot=True):
    orig_x, orig_y = utils.read_data()
    orig_y = utils.classification(orig_y, orig_y)
    orig_y = utils.one_hot_labels(orig_y)
    orig_x = orig_x / 255.
    # Create the training sample
    train_x, val_test_x, train_y, val_test_y = train_test_split(orig_x, orig_y, test_size=0.3, random_state=1)
    # Split the remaining observations into validation and test
    val_x, test_x, val_y, test_y = train_test_split(val_test_x, val_test_y, test_size=0.33, random_state=1)

    # Example of an image from the dataset
    index = 9
    plt.imshow(orig_x[index])

    print("number of training examples = " + str(train_x.shape[0]))
    print("number of validation examples = " + str(val_x.shape[0]))
    print("X_train shape: " + str(train_x.shape))
    print("Y_train shape: " + str(train_y.shape))
    # print("X_test shape: " + str(val_x.shape))
    # print("Y_test shape: " + str(val_y.shape))

    conv_model = convolutional_model((256, 256, 3))
    lr_schedule = tf.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=640,
        decay_rate=0.9
    )
    adam = tf.optimizers.Adam(learning_rate=lr_schedule)
    sgd = tf.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)
    conv_model.compile(optimizer=sgd,
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])
    conv_model.summary()

    datagen = ImageDataGenerator(featurewise_std_normalization=True)
    train_batches = datagen.flow(train_x, train_y, batch_size=64)
    train_crops = utils.crop_generator(train_batches, 256)
    val_batches = datagen.flow(val_x, val_y, batch_size=64)
    val_crops = utils.crop_generator(val_batches, 256)
    test_batches = datagen.flow(test_x, test_y, batch_size=1, shuffle=False)
    test_crops = utils.crop_generator(test_batches, 256)

    # history = conv_model.fit(train_ds, epochs=100, validation_data=test_ds)
    history = conv_model.fit_generator(train_crops, epochs=400, steps_per_epoch=64,
                                       validation_data=val_crops, validation_steps=64)

    print(history.history)
    df_loss_acc = pd.DataFrame(history.history)
    df_loss = df_loss_acc[['loss', 'val_loss']]
    # df_loss.rename(columns={'loss': 'train', 'val_loss': 'validation'}, inplace=True)
    df_acc = df_loss_acc[['accuracy', 'val_accuracy']]
    # df_acc.rename(columns={'accuracy': 'train', 'val_accuracy': 'validation'}, inplace=True)
    df_loss.plot(title='Model loss', figsize=(12, 8)).set(xlabel='Epoch', ylabel='Loss')
    df_acc.plot(title='Model Accuracy', figsize=(12, 8)).set(xlabel='Epoch', ylabel='Accuracy')
    plt.show()

    # Pred_y = conv_model.predict(test_crops, steps=1000)
    # pred_y = np.argmax(Pred_y, axis=1)
    # print('test_crops')
    temp_x, y = [], []
    x = np.empty((1000, 256, 256, 3))
    for i in range(1000):
        a, b = test_crops.__next__()
        temp_x.extend(a)
        x[i,] = a
        y.extend(b)
    print(x.shape)
    Pred_y = conv_model.predict(x)
    pred_y = np.argmax(Pred_y, axis=1)
    Test_y = np.array(y)
    test_y = np.argmax(Test_y, axis=1)
    print(test_y)
    print('pred_y')
    print(pred_y)
    # Calculate accuracy, precision, recall and confusion matrix
    print('Test Accuracy: ', accuracy_score(test_y, pred_y))
    print('Test Precision: ', precision_score(test_y, pred_y, zero_division=0, average='weighted'))
    print('Test Recall: ', recall_score(test_y, pred_y, average='weighted'))
    print('Test f1', f1_score(test_y, pred_y, zero_division=0, average='weighted', labels=np.unique(pred_y)))
    cm = confusion_matrix(test_y, pred_y)
    print('Confustion Matrix', cm)
    cmd = ConfusionMatrixDisplay(confusion_matrix=cm)
    cmd.plot()
    plt.savefig('LeNet_Confusion', dpi=300)
    plt.clf()

    scores = conv_model.evaluate(test_crops, steps=1000)
    print(scores)

if __name__ == '__main__':
    main()