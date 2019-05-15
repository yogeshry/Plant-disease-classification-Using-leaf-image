from keras import layers

def coreCNN(img_input):        
        def se_block1(in_block, ch=256, ratio=16):
          x = layers.GlobalAveragePooling2D()(in_block)
          x = layers.Dense(64, activation='relu')(x)
          x = layers.Dense(ch, activation='sigmoid')(x)
          return layers.multiply([in_block, x])
        def se_block2(in_block, ch=288, ratio=16):
          x = layers.GlobalAveragePooling2D()(in_block)
          x = layers.Dense(72, activation='relu')(x)
          x = layers.Dense(ch, activation='sigmoid')(x)
          return layers.multiply([in_block, x])
        def se_block3(in_block, ch=256, ratio=16):
          x = layers.GlobalAveragePooling2D()(in_block)
          x = layers.Dense(64, activation='relu')(x)
          x = layers.Dense(ch, activation='sigmoid')(x)
          return layers.multiply([in_block, x])
        
        def inception1(input):
            branch_0 = layers.Conv2D(64, 1,padding='same', activation='relu')(input)
            branch_1 = layers.Conv2D(64, 1,padding='same', activation='relu')(input)
            branch_1 = layers.Conv2D(64, 3,padding='same', activation='relu')(branch_1)
            branch_2 = layers.Conv2D(64, 1,padding='same', activation='relu')(input)
            branch_2 = layers.Conv2D(64, 3,padding='same', activation='relu')(branch_2)
            branch_2 = layers.Conv2D(64, 3,padding='same', activation='relu')(branch_2)
            branch_3 = layers.Conv2D(64, 1,padding='same', activation='relu')(input)
            branch_3 = layers.MaxPooling2D((2,2), strides=(1,1), padding='same')(branch_3)
            output   = layers.concatenate([branch_0, branch_1, branch_2, branch_3], axis = 3)
        #     branch_0 = layers.Conv2D(32, 1,padding='same', activation='relu')(input)

            return output
        def inception2(input):
            branch_0 = layers.Conv2D(96, 1,padding='same', activation='relu')(input)
            branch_1 = layers.Conv2D(96, 1,padding='same', activation='relu')(input)
            branch_1 = layers.Conv2D(96, 3,padding='same', activation='relu')(branch_1)
            branch_3 = layers.Conv2D(96, 1,padding='same', activation='relu')(input)
            branch_3 = layers.MaxPooling2D((2,2), strides=(1,1), padding='same')(branch_3)
            output   = layers.concatenate([branch_0, branch_1, branch_3], axis = 3)
        #     branch_0 = layers.Conv2D(32, 1,padding='same', activation='relu')(input)

            return output
        def inception3(input):
            branch_0 = layers.Conv2D(128, 1,padding='same', activation='relu')(input)
            branch_1 = layers.Conv2D(128, 1,padding='same', activation='relu')(input)
            branch_1 = layers.Conv2D(128, 3,padding='same', activation='relu')(branch_1)
            output   = layers.concatenate([branch_0, branch_1], axis = 3)
        #     branch_0 = layers.Conv2D(32, 1,padding='same', activation='relu')(input)

            return output

        x = layers.Conv2D(64,3, activation = 'relu')(img_input)
        x = layers.Conv2D(64,3, activation = 'relu')(x)
        x = layers.Conv2D(64,3, activation = 'relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation='relu')(x)
        x = layers.Conv2D(256,1, activation = 'relu')(x)


        y = layers.MaxPooling2D(3)(x)


        x = inception1(y)
        #x = se_block1(x)
        x = layers.add([x,y])

        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation='relu')(x)
        x = layers.Conv2D(288,1, activation = 'relu')(x)
        y = layers.MaxPooling2D(2)(x)

        x = inception2(y)
        #x = se_block2(x)
        x = layers.add([x,y])

        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation='relu')(x)
        x = layers.Conv2D(256,1, activation = 'relu')(x)
        y = layers.MaxPooling2D(2)(x)

        x = inception3(y)
        #x= se_block3(x)
        x = layers.add([x,y])



        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation='relu')(x)

        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(0.5)(x)
        # Flatten the output layer to 1 dimension

        return x
