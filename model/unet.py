
from keras import Input
from keras.engine import Model
from keras.layers import ZeroPadding2D, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose, Cropping2D, Activation
from keras.layers.normalization import BatchNormalization

from hugin.tools.loss import keras_dice_loss
from hugin.tools.utils import custom_objects, f1, dice_coef

@custom_objects({'f1': f1, 'dice_coef': dice_coef,'keras_dice_loss':keras_dice_loss})
def build_unet(input_width=256,
            input_height=256,
            n_channels=4,
            kernel=3,
            stride=1,
            activation='relu',
            nr_classes=8,
            kinit='RandomUniform',
            batch_norm=True,
            padding='same',
            axis=3,
            crop=0,
            mpadd=0,trainable=False,
            ):
    inputs = Input((input_height, input_width, n_channels))



    conv1 = ZeroPadding2D((crop, crop))(inputs)

    conv1 = Conv2D(32, kernel_size=kernel, strides=stride, kernel_initializer=kinit, padding=padding,trainable=trainable)(conv1)
    conv1 = BatchNormalization()(conv1) if batch_norm else conv1
    conv1 = Activation(activation)(conv1)
    conv1 = Conv2D(32, kernel_size=kernel, strides=stride,  kernel_initializer=kinit, padding=padding,trainable=trainable)(conv1)
    conv1 = BatchNormalization()(conv1) if batch_norm else conv1
    conv1 = Activation(activation)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, kernel_size=kernel, strides=stride,  kernel_initializer=kinit, padding=padding,trainable=trainable)(pool1)
    conv2 = BatchNormalization()(conv2)if batch_norm else conv2
    conv2 = Activation(activation)(conv2)
    conv2 = Conv2D(64, kernel_size=kernel, strides=stride,  kernel_initializer=kinit, padding=padding,trainable=trainable)(conv2)
    conv2 = BatchNormalization()(conv2) if batch_norm else conv2
    conv2 = Activation(activation)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, kernel_size=kernel, strides=stride, kernel_initializer=kinit, padding=padding,trainable=trainable)(pool2)
    conv3 = BatchNormalization()(conv3) if batch_norm else conv3
    conv3 = Activation(activation)(conv3)
    conv3 = Conv2D(128, kernel_size=kernel, strides=stride,  kernel_initializer=kinit, padding=padding,trainable=trainable)(conv3)
    conv3 = BatchNormalization()(conv3) if batch_norm else conv3
    conv3 = Activation(activation)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, kernel_size=kernel, strides=stride,  kernel_initializer=kinit, padding=padding,trainable=trainable)(pool3)
    conv4 = BatchNormalization()(conv4) if batch_norm else conv4
    conv4 = Activation(activation)(conv4)
    conv4 = Conv2D(256, kernel_size=kernel, strides=stride,  kernel_initializer=kinit, padding=padding,trainable=trainable)(conv4)
    conv4 = BatchNormalization()(conv4) if batch_norm else conv4
    conv4 = Activation(activation)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, kernel_size=kernel, strides=stride,  kernel_initializer=kinit, padding=padding,trainable=trainable)(pool4)
    conv5 = BatchNormalization()(conv5) if batch_norm else conv5
    conv5 = Activation(activation)(conv5)
    conv5 = Conv2D(512, kernel_size=kernel, strides=stride,  kernel_initializer=kinit, padding=padding,trainable=trainable)(conv5)
    conv5 = BatchNormalization()(conv5) if batch_norm else conv5
    conv5 = Activation(activation)(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding=padding)(conv5), conv4], axis=axis)
    conv6 = Conv2D(256, kernel_size=kernel, strides=stride,  kernel_initializer=kinit, padding=padding)(up6)
    conv6 = BatchNormalization()(conv6) if batch_norm else conv6
    conv6 = Activation(activation)(conv6)
    conv6 = Conv2D(256, kernel_size=kernel, strides=stride,  kernel_initializer=kinit, padding=padding)(conv6)
    conv6 = BatchNormalization()(conv6) if batch_norm else conv6
    conv6 = Activation(activation)(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding=padding)(conv6), conv3], axis=axis)
    conv7 = Conv2D(128, kernel_size=kernel, strides=stride,  kernel_initializer=kinit, padding=padding)(up7)
    conv7 = BatchNormalization()(conv7) if batch_norm else conv7
    conv7 = Activation(activation)(conv7)
    conv7 = Conv2D(128, kernel_size=kernel, strides=stride,  kernel_initializer=kinit, padding=padding)(conv7)
    conv7 = BatchNormalization()(conv7) if batch_norm else conv7
    conv7 = Activation(activation)(conv7)


    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding=padding)(conv7), conv2], axis=axis)
    conv8 = Conv2D(64, kernel_size=kernel, strides=stride,  kernel_initializer=kinit, padding=padding)(up8)
    conv8 = BatchNormalization()(conv8) if batch_norm else conv8
    conv8 = Activation(activation)(conv8)
    conv8 = Conv2D(64, kernel_size=kernel, strides=stride,  kernel_initializer=kinit, padding=padding)(conv8)
    conv8 = BatchNormalization()(conv8) if batch_norm else conv8
    conv8 = Activation(activation)(conv8)


    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding=padding)(conv8), conv1], axis=axis)
    conv9 = Conv2D(32, kernel_size=kernel, strides=stride,  kernel_initializer=kinit, padding=padding)(up9)
    conv9 = BatchNormalization()(conv9) if batch_norm else conv9
    conv9 = Activation(activation)(conv9)


    conv9 = Conv2D(32, kernel_size=kernel, strides=stride,  kernel_initializer=kinit, padding=padding)(conv9)
    conv9 = BatchNormalization()(conv9) if batch_norm else conv9
    conv9 = Activation(activation)(conv9)



    conv9 = Cropping2D((mpadd, mpadd))(conv9)

    conv10 = Conv2D(nr_classes, (1, 1), activation='softmax')(conv9)
    model = Model(inputs=[inputs], outputs=[conv10])

    return model

