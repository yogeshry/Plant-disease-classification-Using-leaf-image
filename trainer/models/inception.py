from keras import layers

def coreCNN(img_input):        

        def inception1(input):
            branch_0 = layers.Conv2D(16, 1,padding='same', activation='relu')(input)
            branch_1 = layers.Conv2D(16, 1,padding='same', activation='relu')(input)
            branch_1 = layers.Conv2D(16, 3,padding='same', activation='relu')(branch_1)
            branch_2 = layers.Conv2D(16, 1,padding='same', activation='relu')(input)
            branch_2 = layers.Conv2D(16, 3,padding='same', activation='relu')(branch_2)
            branch_2 = layers.Conv2D(16, 3,padding='same', activation='relu')(branch_2)
            branch_3 = layers.Conv2D(16, 1,padding='same', activation='relu')(input)
            branch_3 = layers.MaxPooling2D((2,2), strides=(1,1), padding='same')(branch_3)
            output   = layers.concatenate([branch_0, branch_1, branch_2, branch_3], axis = 3)
        #     branch_0 = layers.Conv2D(32, 1,padding='same', activation='relu')(input)

            return output
        def inception2(input):
            branch_0 = layers.Conv2D(16, 1,padding='same', activation='relu')(input)
            branch_1 = layers.Conv2D(16, 1,padding='same', activation='relu')(input)
            branch_1 = layers.Conv2D(16, 3,padding='same', activation='relu')(branch_1)
            branch_3 = layers.Conv2D(16, 1,padding='same', activation='relu')(input)
            branch_3 = layers.MaxPooling2D((2,2), strides=(1,1), padding='same')(branch_3)
            output   = layers.concatenate([branch_0, branch_1, branch_3], axis = 3)
        #     branch_0 = layers.Conv2D(32, 1,padding='same', activation='relu')(input)

            return output
        def inception3(input):
            branch_0 = layers.Conv2D(16, 1,padding='same', activation='relu')(input)
            branch_1 = layers.Conv2D(16, 1,padding='same', activation='relu')(input)
            branch_1 = layers.Conv2D(16, 3,padding='same', activation='relu')(branch_1)
            output   = layers.concatenate([branch_0, branch_1], axis = 3)
        #     branch_0 = layers.Conv2D(32, 1,padding='same', activation='relu')(input)

            return output

        x = layers.Conv2D(64,3, activation = 'relu')(img_input)
        x = layers.Conv2D(32,3, activation = 'relu')(x)
        x = layers.Conv2D(32,3, activation = 'relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation='relu')(x)
        x = layers.Conv2D(64,1, activation = 'relu')(x)
        y = layers.MaxPooling2D(3)(x)

        x = inception1(y)

        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation='relu')(x)
        x = layers.Conv2D(48,1, activation = 'relu')(x)
        y = layers.MaxPooling2D(2)(x)

        x = inception2(y)


        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation='relu')(x)
        x = layers.Conv2D(32,1, activation = 'relu')(x)
        y = layers.MaxPooling2D(2)(x)

        x = inception3(y)



        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation='relu')(x)

        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(0.5)(x)
        # Flatten the output layer to 1 dimension

        return x
