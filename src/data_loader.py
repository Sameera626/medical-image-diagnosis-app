import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE

def get_datasets(data_dir="data/train",
                 img_size=(224,224),
                 batch_size=32,
                 validation_split=0.2,
                 seed=1337):
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
        directory=data_dir,
        labels='inferred',
        label_mode='binary',
        validation_split=validation_split,
        subset='training',
        seed=seed,
        image_size=img_size,
        batch_size=batch_size
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        directory=data_dir,
        labels='inferred',
        label_mode='binary',
        validation_split=validation_split,
        subset='validation',
        seed=seed,
        image_size=img_size,
        batch_size=batch_size
    )
    
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    return train_ds, val_ds

def get_test_dataset(test_dir="data/test", img_size=(224,224), batch_size=32):

    test_ds = tf.keras.utils.image_dataset_from_directory(
        directory=test_dir,
        labels='inferred',
        label_mode='binary',
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False
    )
    return test_ds
    
        