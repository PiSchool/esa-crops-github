
import sys
from keras import Input
from keras.engine import Model
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Conv2DTranspose,concatenate, add, Dropout
from keras.utils import plot_model

from hugin.tools.utils import custom_objects, f1, dice_coef




@custom_objects({'f1': f1, 'dice_coef': dice_coef})
def build_wnet(input_width=256,
         input_height=256,
         n_channels=4, drop=0.5, drp=True,nr_classes=8):

    # entry flow
    img_input = Input(shape=(input_height, input_width, n_channels))

    conv1_1 = Conv2D(32, (3, 3), padding="same")(img_input)
    conv1_1 = Activation('elu')(conv1_1)

    conv1_2 = Conv2D(32, (3, 3),padding="same")(conv1_1)
    conv1_2 = Activation('elu')(conv1_2)
    max1 = MaxPooling2D(pool_size=(2, 2))(conv1_2)

    conv2_1 = Conv2D(64, (3, 3),padding="same")(max1)
    conv2_1 = Activation('elu')(conv2_1)
    conv2_2 = Conv2D(64, (3, 3),padding="same")(conv2_1)
    conv2_2 = Activation('elu')(conv2_2)
    max2 = MaxPooling2D(pool_size=(2, 2))(conv2_2)

    conv3_1 = Conv2D(128, (3, 3), padding="same")(max2)
    conv3_1 = Activation('elu')(conv3_1)

    conv3_2 = Conv2D(128, (3, 3), padding="same")(conv3_1)
    conv3_2 = Activation('elu')(conv3_2)

    max3 = MaxPooling2D(pool_size=(2, 2))(conv3_2)

    conv4_1 = Conv2D(256, (3, 3), padding="same")(max3)
    conv4_1 = Activation('relu')(conv4_1)
    conv4_1 = BatchNormalization()(conv4_1)


    #conv4_1 = Dropout(drop)(conv4_1) if drp else conv4_1
    conv4_2 = Conv2D(256, (3, 3), padding="same")(conv4_1)
    conv4_2 = BatchNormalization()(conv4_2)
    conv4_2 = Activation('relu')(conv4_2)


    max4 = MaxPooling2D(pool_size=(2, 2))(conv4_2)

    conv5_1 = Conv2D(512, (3, 3), padding="same")(max4)
    conv5_1 = BatchNormalization()(conv5_1)
    conv5_1 = Activation('relu')(conv5_1)


    conv5_2 = Conv2D(512, (3, 3), padding="same")(conv5_1)
    conv5_2 = BatchNormalization()(conv5_2)
    conv5_2 = Activation('relu')(conv5_2)




    up6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv5_2)
    concat6 = concatenate([conv4_2,up6])

    conv6_1 = Conv2D(512, (3, 3), padding="same")(concat6)
    conv6_1 = Activation('elu')(conv6_1)

    conv6_2 = Conv2D(512, (3, 3), padding="same")(conv6_1)
    conv6_2 = Activation('elu')(conv6_2)


    up7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv6_2)
    concat7 = concatenate([conv3_2,up7])

    conv7_1 = Conv2D(256, (3, 3), padding="same")(concat7)
    conv7_1 = Activation('elu')(conv7_1)

    conv7_2 = Conv2D(256, (3, 3), padding="same")(conv7_1)
    conv7_2 = Activation('elu')(conv7_2)


    up8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv7_2)
    concat8 = concatenate([conv2_2, up8])

    conv8_1 = Conv2D(128, (3, 3), padding="same")(concat8)
    conv8_1 = Activation('elu')(conv8_1)

    conv8_2 = Conv2D(128, (3, 3), padding="same")(conv8_1)
    conv8_2 = Activation('elu')(conv8_2)


    up9 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8_2)
    concat9 = concatenate([conv1_2, up9])

    conv9_1 = Conv2D(64, (3, 3), padding="same")(concat9)
    conv9_1 = Activation('elu')(conv9_1)

    conv9_2 = Conv2D(64, (3, 3), padding="same")(conv9_1)
    conv9_2 = Activation('elu')(conv9_2)


    conv10_1 = Conv2D(32, (3, 3), padding="same")(conv9_2)
    conv10_1 = Activation('elu')(conv10_1)

    conv10_2 = Conv2D(32, (3, 3), padding="same")(conv10_1)
    conv10_2 = Activation('elu')(conv10_2)

    max10 = MaxPooling2D(pool_size=(2, 2))(conv10_2)


    concat11 = concatenate([max10,conv8_2])
    conv11_1 = Conv2D(64, (3, 3), padding="same")(concat11)
    conv11_1 = Activation('elu')(conv11_1)

    conv11_2 = Conv2D(64, (3, 3), padding="same")(conv11_1)
    conv11_2 = Activation('elu')(conv11_2)

    max11 = MaxPooling2D(pool_size=(2, 2))(conv11_2)


    concat12 = concatenate([max11, conv7_2])
    conv12_1 = Conv2D(128, (3, 3), padding="same")(concat12)
    conv12_1 = Activation('elu')(conv12_1)

    conv12_2 = Conv2D(128, (3, 3), padding="same")(conv12_1)
    conv12_2 = Activation('elu')(conv12_2)

    max12 = MaxPooling2D(pool_size=(2, 2))(conv12_2)


    concat13 = concatenate([max12, conv6_2])
    conv13_1 = Conv2D(256, (3, 3), padding="same")(concat13)
    conv13_1 = BatchNormalization()(conv13_1)
    conv13_1 = Activation('relu')(conv13_1)


    conv13_2 = Conv2D(256, (3, 3), padding="same")(conv13_1)
    conv13_2 = BatchNormalization()(conv13_2)
    conv13_2 = Activation('relu')(conv13_2)


    max13 = MaxPooling2D(pool_size=(2, 2))(conv13_2)

    concat14 = concatenate([max13, conv5_2])
    conv14_1 = Conv2D(512, (3, 3), padding="same")(concat14)
    conv14_1 = BatchNormalization()(conv14_1)
    conv14_1 = Activation('relu')(conv14_1)


    conv14_2 = Conv2D(512, (3, 3), padding="same")(conv14_1)
    conv14_2 = BatchNormalization()(conv14_2)
    conv14_2 = Activation('relu')(conv14_2)



    up15 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv14_2)
    skip4_13_15 = add([conv4_2,conv13_2])
    concat15 = concatenate([skip4_13_15, up15])
    concat15 = Dropout(drop)(concat15) if drp else concat15

    conv15_1 = Conv2D(512, (3, 3), padding="same")(concat15)
    conv15_1 = Activation('elu')(conv15_1)

    conv15_2 = Conv2D(512, (3, 3), padding="same")(conv15_1)
    conv15_2 = Activation('elu')(conv15_2)


    up16 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv15_2)
    skip3_12_16 = add([conv3_2, conv12_2])
    concat16 = concatenate([skip3_12_16, up16])
    concat16 = Dropout(drop)(concat16) if drp else concat16

    conv16_1 = Conv2D(256, (3, 3), padding="same")(concat16)
    conv16_1 = Activation('elu')(conv16_1)

    conv16_2 = Conv2D(256, (3, 3), padding="same", activation="elu")(conv16_1)

    up17 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv16_2)
    skip2_11_17 = add([conv2_2, conv11_2])
    concat17 = concatenate([skip2_11_17, up17])
    concat17 = Dropout(drop)(concat17) if drp else concat17

    conv17_1 = Conv2D(128, (3, 3), padding="same")(concat17)
    conv17_1 = Activation('elu')(conv17_1)

    conv17_2 = Conv2D(128, (3, 3), padding="same")(conv17_1)
    conv17_2 = Activation('elu')(conv17_2)


    up18 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv17_2)
    skip1_10_18 = add([conv1_2, conv10_2])
    concat18 = concatenate([skip1_10_18, up18])
    concat18 = Dropout(drop)(concat18) if drp else concat18
    conv18_1 = Conv2D(64, (3, 3), padding="same")(concat18)
    conv18_1 = Activation('elu')(conv18_1)

    conv18_2 = Conv2D(64, (3, 3), padding="same")(conv18_1)
    conv18_2 = Activation('elu')(conv18_2)


    x = Conv2D(nr_classes, (1, 1))(conv18_2)
    x = Activation('softmax')(x)

    # Create model.
    inputs = img_input
    model = Model(inputs, x, name='wnet')


    return model