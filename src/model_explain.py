import numpy as np
import tensorflow as tf
import cv2


def make_gradcam_heatmap(img_array, model, pred_index=None):
    
    base_model = model.get_layer("efficientnetb0")
    print(base_model.summary())
    
    grad_model = tf.keras.Model(
        inputs = [model.inputs],
        outputs = [base_model.get_layer("top_conv").output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array, training=False)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, cam_path='cam.jpg', alpha=0.4, img_size=(224, 224)):

    img = cv2.imread(img_path)
    img = cv2.resize(img, img_size)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(225 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * alpha + img
    cv2.imwrite(cam_path, superimposed_img)
    return cam_path

