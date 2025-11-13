import tensorflow as tf

def build_model(input_shape=(224,224,3), base_name='EfficientNetB0',freeze_until=100):

    if base_name == 'EfficientNetB0':
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=input_shape
        )
        preprocess_input = tf.keras.applications.efficientnet.preprocess_input

    elif base_name == 'ResNet50':
        base_model= tf.keras.applications.ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=input_shape
        )
        preprocess_input = tf.keras.applications.resnet.preprocess_input

    else:
        raise ValueError(f"Unsupported base model: {base_name}")

    # freeze base model layers
    base_model.trainable = True
    if freeze_until is not None:
        for layer in base_model.layers[:freeze_until]:
            layer.trainable = False 

    inputs = tf.keras.Input(shape=input_shape)
    x = preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs, outputs)
    return model
