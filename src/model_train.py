import tensorflow as tf
from model_builder import build_model
from data_loader import get_datasets


def main():

    train_ds, val_ds = get_datasets()
    print('Datasets loaded successfully.', train_ds, val_ds)

    model = build_model()

    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss = 'binary_crossentropy',
        metrics = ['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_auc', mode='max'),
        tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=4, mode='max', restore_best_weights=True),
        tf.keras.callbacks.TensorBoard(log_dir='./logs') 
    ]

    model.fit(
        train_ds,
        validation_data = val_ds,
        epochs = 12,
        callbacks = callbacks
    )
    
    model.save('medical_image_model.keras')
    print("Model saved as keras format")

if __name__ == "__main__":
    main()
