import tensorflow as tf
from PIL import Image


def create_grad_cam_heatmap(model, inputs, pred_index, last_conv_layer, timestep):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer).output, model.output]
    )

    with tf.GradientTape() as tape:
        feature_maps, cnn_preds = grad_model(inputs)
        # if cnn_pred_index is None:
        #     cnn_pred_index = tf.argmax(cnn_preds[cnn_pred_index])
        class_channel = cnn_preds[:, pred_index]

    grads = tape.gradient(class_channel, feature_maps)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 2, 3))

    feature_maps = feature_maps[0]

    heatmap_res = tf.zeros([0, feature_maps.shape[1], feature_maps.shape[2]])

    for i in range(0, feature_maps.shape[0]):
        heatmap = feature_maps[i, ...] @ pooled_grads[i, ..., tf.newaxis]
        heatmap = tf.reshape(heatmap, (1, heatmap.shape[0], heatmap.shape[1]))
        if tf.math.reduce_max(heatmap) == 0:
            heatmap = tf.maximum(heatmap, 0)
        else:
            heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

        heatmap_res = tf.concat([heatmap_res, heatmap], axis=0)

    # heatmap_res = tf.maximum(heatmap_res, 0) / tf.math.reduce_max(heatmap_res)

    np_heatmap = heatmap_res.numpy()

    # t = 25
    image = inputs[0, timestep, ..., 0]
    image_heatmap = np_heatmap[timestep, ...]

    image_heatmap = Image.fromarray(image_heatmap)
    image_heatmap = image_heatmap.resize((image.shape[1], image.shape[0]))
    image_heatmap = tf.keras.preprocessing.image.img_to_array(image_heatmap)

    return image_heatmap


def create_grad_lstm_cam_heatmap(model, inputs, pred_index, last_conv_layer, last_lstm_layer, timestep,
                                 mixed_weighting, scale_across_time):

    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer).output, model.output]
    )

    with tf.GradientTape() as tape:
        feature_maps, cnn_preds = grad_model(inputs)
        # if cnn_pred_index is None:
        #     cnn_pred_index = tf.argmax(cnn_preds[cnn_pred_index])
        class_channel = cnn_preds[:, pred_index]

    grads = tape.gradient(class_channel, feature_maps)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 2, 3))

    feature_maps = feature_maps[0]

    lstm_grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_lstm_layer).output, model.output]
    )

    with tf.GradientTape() as lstm_tape:
        lstm_feature_maps, lstm_preds = lstm_grad_model(inputs)
        # if cnn_pred_index is None:
        #     cnn_pred_index = tf.argmax(cnn_preds[cnn_pred_index])
        lstm_class_channel = lstm_preds[:, pred_index]

    lstm_grads = lstm_tape.gradient(lstm_class_channel, lstm_feature_maps)
    lstm_grads = lstm_grads[0]

    heatmap_res = tf.zeros([0, feature_maps.shape[1], feature_maps.shape[2]])

    for i in range(0, feature_maps.shape[0]):
        if mixed_weighting:
            heatmap = feature_maps[i, ...] @ (pooled_grads[i, ..., tf.newaxis] * lstm_grads[i, ..., tf.newaxis])
        else:
            heatmap = feature_maps[i, ...] @ lstm_grads[i, ..., tf.newaxis]
        heatmap = tf.reshape(heatmap, (1, heatmap.shape[0], heatmap.shape[1]))

        if not scale_across_time:
            if tf.math.reduce_max(heatmap) == 0:
                heatmap = tf.maximum(heatmap, 0)
            else:
                heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

        heatmap_res = tf.concat([heatmap_res, heatmap], axis=0)

    if scale_across_time:
        heatmap_res = tf.maximum(heatmap_res, 0) / tf.math.reduce_max(heatmap_res)

    np_heatmap = heatmap_res.numpy()

    # t = 25
    image = inputs[0, timestep, ..., 0]
    image_heatmap = np_heatmap[timestep, ...]

    image_heatmap = Image.fromarray(image_heatmap)
    image_heatmap = image_heatmap.resize((image.shape[1], image.shape[0]))
    image_heatmap = tf.keras.preprocessing.image.img_to_array(image_heatmap)

    return image_heatmap
