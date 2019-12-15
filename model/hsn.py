from keras import Input, Model, layers
from keras.layers import Conv2D, Activation, concatenate, add, MaxPooling2D, Conv2DTranspose, Dropout, \
    BatchNormalization

from hugin.tools.utils import custom_objects, f1, dice_coef


def inception(input, filters, drop=0.2, drp=False,activation='elu'):

    conv1_1 = Conv2D(filters[0],(1,1), padding="same")(input)
    conv1_1 = Activation(activation)(conv1_1)

    conv1_2 = Conv2D(filters[1],(3,3), padding="same")(conv1_1)
    conv1_2 = Activation(activation)(conv1_2)


    conv2_1 = Conv2D(filters[2], (1, 1), padding="same")(input)
    conv2_1 = Activation(activation)(conv2_1)


    conv2_2 = Conv2D(filters[3], (5, 5), padding="same")(conv2_1)
    conv2_2 = Activation(activation)(conv2_2)


    conv3_1 = Conv2D(filters[4], (1, 1), padding="same")(input)
    conv3_1 = Activation(activation)(conv3_1)


    conv3_2 = Conv2D(filters[5], (7, 7), padding="same")(conv3_1)
    conv3_3 = Activation(activation)(conv3_2)


    conv4 = Conv2D(filters[6], (1, 1), padding="same")(input)
    conv4 = Activation(activation)(conv4)



    y = layers.concatenate([conv1_2,conv2_2,conv3_2,conv4])

    return y





def residual (input, drop=0.2, drp=True,activation='elu'):

    x = Conv2D(128, (1,1), padding="same")(input)
    x = Activation(activation)(x)

    x = Conv2D(128, (3,3), padding="same")(x)

    shortcut = Conv2D(128,(1,1), padding="same")(input)

    y = layers.add([x, shortcut])
    y = Activation('elu')(y)

    return y

@custom_objects({'f1': f1, 'dice_coef': dice_coef})
def build_hsn(input_width=256,
         input_height=256,
         n_channels=4, drop=0.5, drp=True,
           nr_classes=8):


    #entry flow
    img_input = Input(shape=(input_height, input_width, n_channels))

    # A layers
    x = Conv2D(64,(3,3), padding="same")(img_input)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((2,2), padding="same")(x)


    #B layers
    x = Conv2D(128, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    input_g1 = x
    x = MaxPooling2D((2,2), padding="same")(x)


    #C layers
    x = inception(x,[128,128,64,32,32,32,64],drop=drop)
    x = inception(x, [128, 128, 64, 32, 32, 32, 64],drop=drop)
    input_g2 = x
    x = MaxPooling2D((2,2), padding="same")(x)


    #D layers
    x = inception(x, [256, 384, 64, 32, 32, 32, 64],drop=drop)
    x = inception(x, [256, 384, 64, 32, 32, 32, 64],drop=drop)
    x = inception(x, [256, 384, 64, 32, 32, 32, 64],drop=drop)

    #G layers
    g1 = residual (input_g1)
    g2 = residual (input_g2)


    #F layer
    x = Conv2DTranspose(512,(2,2), strides=(2,2))(x)

    x = layers.concatenate([x,g2])
    x = Dropout(drop)(x) if drp else x

    x = inception(x, [128, 128, 64, 32, 32, 32, 64],drop=drop)
    x = inception(x, [128, 128, 64, 32, 32, 32, 64],drop=drop)


    x = Conv2DTranspose(512,(2,2), strides=(2,2))(x)

    x = layers.concatenate([x,g1])
    x = Dropout(drop)(x) if drp else x

    x = Conv2D(128, (3, 3), padding="same")(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (3, 3), padding="same")(x)
    x = Activation('elu')(x)


    x = Conv2DTranspose(512,(2,2), strides=(2,2))(x)

    x = Conv2D(nr_classes, (1,1))(x)
    x = Activation('softmax')(x)



    # Create model.
    inputs = img_input
    model = Model(inputs, x, name='hsn')

    return model


