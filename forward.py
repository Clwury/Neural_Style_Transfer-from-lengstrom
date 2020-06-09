from nst_utils import *
import functools

BATCH_SIZE = 1
STYLE_LAYERS = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
CONTENT_LAYER = 'conv4_2'
batch_shape = (BATCH_SIZE, CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH, CONFIG.COLOR_CHANNELS)

def _tensor_size(tensor):
    from operator import mul
    return functools.reduce(mul, (d for d in tensor.get_shape()[1:]), 1)


def compute_content_cost(vgg_content, vgg_target):
    content_size = _tensor_size(vgg_content[CONTENT_LAYER])*BATCH_SIZE
    assert _tensor_size(vgg_content[CONTENT_LAYER]) == _tensor_size(vgg_target[CONTENT_LAYER])
    content_loss = 2 * tf.nn.l2_loss(vgg_target[CONTENT_LAYER] - vgg_content[CONTENT_LAYER]) / tf.cast(content_size, tf.float32)
    
    return content_loss


def compute_style_cost(vgg_style, vgg_target):
    style_losses = []
    for style_layer in STYLE_LAYERS:
        features = vgg_style[style_layer]
        features = tf.reshape(features, (-1, features.shape[3]))
        gram = tf.matmul(tf.transpose(features), features) / tf.cast(tf.size(features), tf.float32)
        # style_features[style_layer] = gram

        layer = vgg_target[style_layer]
        bs, height, width, filters = map(lambda i:i.value,layer.get_shape())
        size = height * width * filters
        feats = tf.reshape(layer, (bs, height * width, filters))
        feats_T = tf.transpose(feats, perm=[0,2,1])
        grams = tf.matmul(feats_T, feats) / size
        
        style_losses.append(2 * tf.nn.l2_loss(grams - gram)/tf.cast(tf.size(gram), tf.float32))
    style_loss = functools.reduce(tf.add, style_losses) / BATCH_SIZE

    return style_loss

def total_variation_loss(target):
    # total variation denoising
    tv_y_size = _tensor_size(target[:,1:,:,:])
    tv_x_size = _tensor_size(target[:,:,1:,:])
    y_tv = tf.nn.l2_loss(target[:,1:,:,:] - target[:,:batch_shape[1]-1,:,:])
    x_tv = tf.nn.l2_loss(target[:,:,1:,:] - target[:,:,:batch_shape[2]-1,:])
    tv_loss = 2*(x_tv/tf.cast(tv_x_size, tf.float32) + y_tv/tf.cast(tv_y_size, tf.float32))/BATCH_SIZE

    return tv_loss