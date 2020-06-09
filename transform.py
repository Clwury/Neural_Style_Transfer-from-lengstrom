from nst_utils import *

# class TransformNet(object):
    # def __init__(self):
    #     # super().__init__()
    #     print(self._conv_layer(np.ones([1, 256, 256, 3], dtype=np.float32), 32, 9, stride=1))
    #     x = self._conv_layer(np.ones([1, 256, 256, 3], dtype=np.float32), 32, 9, stride=1)
    #     print(self._residual_block(x, 32, 32, 3))

def net(image):
    with tf.variable_scope('transfrom'):
        # image = image - np.array([123.68, 116.779, 103.939])  # 预处理输入图片
        conv1 = conv_layer(image, 32, 9, 1)
        print("//////////////")
        print(conv1)
        conv2 = conv_layer(conv1, 64, 3, 2)
        print(conv2)
        conv3 = conv_layer(conv2, 128, 3, 2)
        print(conv3)
        resid1 = residual_block(conv3, 128, 3, 1)
        print(resid1)
        resid2 = residual_block(resid1, 128, 3, 1)
        print(resid2)
        resid3 = residual_block(resid2, 128, 3, 1)
        print(resid3)
        resid4 = residual_block(resid3, 128, 3, 1)
        print(resid4)
        resid5 = residual_block(resid4, 128, 3, 1)
        print(resid5)
        conv_t1 = conv_transpose_layer(resid5, 64, 3, 2)
        print(conv_t1)
        conv_t2 = conv_transpose_layer(conv_t1, 32, 3, 2)
        print(conv_t2)
        conv_t3 = conv_layer(conv_t2, 3, 9, 1, relu=False)
        print(conv_t3)
        preds = tf.nn.sigmoid(conv_t3) * 255
        return preds

def relu_layer(conv2d_layer):
    """
    Return the RELU function wrapped over a TensorFlow layer. Expects a
    Conv2d layer input.
    """
    return tf.nn.relu(conv2d_layer)

def conv2d(prev_layer, base, kernel_size, stride, transpose=False):
    """
    Return the Conv2D layer using the weights, biases from the VGG
    model at 'layer'.
    """
    if not transpose:
        conv_filter_w = tf.Variable(tf.truncated_normal([kernel_size, kernel_size, prev_layer.shape[3], base], stddev=.1, seed=1), trainable=True)
        return tf.nn.conv2d(prev_layer, conv_filter_w, strides=[1, stride, stride, 1], padding='SAME')
    else:
        # 反卷积
        batch_size, height, width, channels = [i.value for i in prev_layer.get_shape()]
        conv_filter_w = tf.Variable(tf.truncated_normal([kernel_size, kernel_size, base, prev_layer.shape[3]], stddev=.1, seed=1), trainable=True)
        output_shape = [batch_size, int(height)*stride, int(width)*stride, base]
        return tf.nn.conv2d_transpose(prev_layer, conv_filter_w, output_shape, strides=[1, stride, stride, 1], padding='SAME')

# 卷积层
def conv_layer(prev_layer, base, kernel_size, stride=1, relu=True):
    if relu:
        return relu_layer(instance_norm(conv2d(prev_layer, base, kernel_size, stride)))
    else:
        return instance_norm(conv2d(prev_layer, base, kernel_size, stride))
# 残差块
def residual_block(prev_layer, base, kernel_size, stride=1):
    conv = conv_layer(prev_layer, base, kernel_size, stride)
    return prev_layer + conv_layer(conv, base, kernel_size, stride, relu=False)

# 反卷积层
def conv_transpose_layer(prev_layer, base, kernel_size, stride=1):
    return relu_layer(instance_norm(conv2d(prev_layer, base, kernel_size, stride, transpose=True)))

# 归一化
def instance_norm(prev_layer):
    epsilon = 1e-3
    mean, variance = tf.nn.moments(prev_layer, [1,2], keep_dims=True)
    return tf.math.divide(tf.subtract(prev_layer, mean), tf.sqrt(tf.add(variance, epsilon)))