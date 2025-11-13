import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from data_loader import get_test_dataset


def main():

    model = tf.keras.models.load_model('medical_image_model.keras')
    test_ds = get_test_dataset()

    y_true = []
    y_pred_probs = []
    y_pred = []

    for images, labels in test_ds:
        probs = model.predict(images).ravel()
        y_pred_probs.extend(probs)
        y_true.extend(labels.numpy().tolist())
        
    threshold = 0.7

    for p in y_pred_probs:
        if p >= threshold:
            y_pred.append(1)
        else:
            y_pred.append(0)

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4))

    try:
        auc = roc_auc_score(y_true, y_pred_probs)
        print(f"\nROC AUC Score: {auc:.4f}")
    except Exception as e:
        print("\nROC AUC Score could not be computed.", e)


if __name__ == "__main__":
    main()


