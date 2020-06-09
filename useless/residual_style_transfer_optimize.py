from transform_1 import *

BATCH_SIZE = 4
# CHANNEL_NUM = 3
STYLE_LAYERS = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
# CONTENT_LAYER = 'conv4_2'
# LEARNING_RATE = 1e-3
# STYLE_WEIGHT = 5.0
# CONTENT_WEIGHT = 1.0
# TV_WEIGHT = 1e-6
# CHECK_POINT_PATH = 'C:/Users/28620_dfxjqq7/Desktop/logs'
# TRAIN_CHECK_POINT = 'model/model.ckpt'

def gram_matrix(A):
    """
    Argument:
    A -- matrix of shape (n_C, n_H*n_W)
    
    Returns:
    GA -- Gram matrix of A, of shape (n_C, n_C)
    """
    # batch_size, height, width, channels = [i.value for i in A.get_shape()]
    # size = height * width * channels
    # features = tf.reshape(A, (batch_size, height * width, channels))

    ### START CODE HERE ### (≈1 line)
    shape = tf.shape(A)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]
    channel_num = shape[3]
    filters = tf.reshape(A, tf.stack([batch_size, -1, channel_num]))
    GA = tf.matmul(filters, filters, transpose_a=True)/tf.to_float(height * width * channel_num)
    # GA = tf.matmul(A, A, transpose_a=True)
    # GA = tf.matmul(A,tf.transpose(A))         /tf.cast(size, tf.float32)
    ### END CODE HERE ###
    
    return GA

# GRADED FUNCTION: compute_content_cost
def compute_content_cost(a_C, a_G):
    """
    Computes the content cost
    
    Arguments:
    a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C 
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G
    
    Returns: 
    J_content -- scalar that you compute using equation 1 above.
    """
    
    ### START CODE HERE ###
    # Retrieve dimensions from a_G (≈1 line)
    # m, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    # Reshape a_C and a_G (≈2 lines)
    # a_C_unrolled = tf.reshape(a_C,(n_H * n_W,n_C))
    # a_G_unrolled = tf.reshape(a_G,(n_H * n_W,n_C))
    
    # compute the cost with tensorflow (≈1 line)
    # J_content = 1/(4*n_H*n_W*n_C) * tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled , a_G_unrolled)))
    content_loss = tf.nn.l2_loss(a_C['conv4_2'] - a_G['conv4_2']) * 2/tf.cast(tf.size(a_C['conv4_2']), dtype=tf.float32)
    return content_loss
    ### END CODE HERE ###  

# GRADED FUNCTION: compute_layer_style_cost
def compute_layer_style_cost(a_S, a_G):
    """
    Arguments:
    a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S 
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G
    
    Returns: 
    J_style_layer -- tensor representing a scalar value, style cost defined above by equation (2)
    """
    
    ### START CODE HERE ###
    # Retrieve dimensions from a_G (≈1 line)
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    # Reshape the images to have them of shape (n_C, n_H*n_W) (≈2 lines)
    # a_S = tf.reshape(a_S,(-1, n_H*n_W,n_C))
    # a_G = tf.reshape(a_G,(-1, n_H*n_W,n_C))

    # Computing gram_matrices for both images S and G (≈2 lines)
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    # J_style_layer = 1/(4 * (n_C **2) * ((n_H * n_W) **2)) * tf.reduce_sum(tf.square(tf.subtract(GS, GG)))    tf.cast(tf.size(GS), tf.float32)
    J_style_layer = tf.nn.l2_loss(GG - GS) * 2/(4 * (n_C **2) * ((n_H * n_W) **2))
    return J_style_layer

def compute_style_cost(style_img, target_img):
    loss = 0
    for layer_name in STYLE_LAYERS:
        # out = model[layer_name]
        # a_S = sess.run(out)
        # a_G = out
        style_layer = style_img[layer_name]
        style_gram = gram_matrix(style_layer)

        target_layer = target_img[layer_name]
        target_gram = gram_matrix(target_layer)

        loss += tf.nn.l2_loss(style_gram - target_gram) * 2 / tf.cast(tf.size(target_gram), dtype=tf.float32)
    return loss


def total_variation_loss(layer):
    shape = tf.shape(layer)
    height = shape[1]
    width = shape[2]
    y = tf.slice(layer, [0, 0, 0, 0], tf.stack([-1, height - 1, -1, -1])) - tf.slice(layer, [0, 1, 0, 0], [-1, -1, -1, -1])
    x = tf.slice(layer, [0, 0, 0, 0], tf.stack([-1, -1, width - 1, -1])) - tf.slice(layer, [0, 0, 1, 0], [-1, -1, -1, -1])
    loss = 2 * (tf.nn.l2_loss(x) / tf.to_float(tf.size(x)) + tf.nn.l2_loss(y) / tf.to_float(tf.size(y)))/BATCH_SIZE
    return loss

'''
with tf.Graph().as_default():
    batch_image = tf.placeholder(tf.float32, [BATCH_SIZE, CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH, CHANNEL_NUM])
    batch_style_image = reshape_and_normalize_image("style_images/starry_night.jpg")

    # batch_style_image = tf.placeholder(tf.float32, [BATCH_SIZE, CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH, CHANNEL_NUM])

    vgg_C = load_vgg_model('pretrained-model/imagenet-vgg-verydeep-19.mat', batch_image)
    # sess.run(vgg['input'].assign(batch_image))
    # vgg_C = sess.run(vgg)
    # 内容损失
    batch_image_tfrom = net(batch_image)
    # sess.run(vgg['input'].assign(batch_image_tfrom))
    # vgg_CT = sess.run(vgg)
    vgg_CT = load_vgg_model('pretrained-model/imagenet-vgg-verydeep-19.mat', batch_image_tfrom)
    content_loss = compute_content_cost(vgg_C[CONTENT_LAYER], vgg_CT[CONTENT_LAYER])
    # 风格损失
    vgg_S = load_vgg_model('pretrained-model/imagenet-vgg-verydeep-19.mat', batch_style_image)
    # sess.run(vgg['input'].assign(batch_style_image))
    # vgg_S = sess.run(vgg)
    style_loss = compute_style_cost(vgg_CT, vgg_S, STYLE_LAYERS)

    loss = CONTENT_WEIGHT * content_loss + STYLE_WEIGHT * style_loss/BATCH_SIZE + TV_WEIGHT * total_variation_loss(batch_image_tfrom)
    tf.summary.scalar('losses/content_loss', CONTENT_WEIGHT * content_loss)
    tf.summary.scalar('losses/style_loss', STYLE_WEIGHT * style_loss)
    tf.summary.scalar('losses/loss', loss)
    tf.summary.image('transform', batch_image_tfrom, max_outputs=4)
    tf.summary.image('origin', batch_image, max_outputs=4)    
    summary = tf.summary.merge_all()

    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

    var_to_restore = tf.trainable_variables()
    saver = tf.train.Saver(var_to_restore, max_to_keep=10)


        # if __name__ == '__main__':
        # model = load_vgg_model('pretrained-model/imagenet-vgg-verydeep-19.mat')
        # TransformNet()
        # print(type(reshape_and_normalize_image('C:/Users/28620_dfxjqq7/OneDrive - mail.hnust.edu.cn/Neural Style Transfer/content_images/sky.jpg')))
        
        
        # print(image_batch)   batch_style_image: style_image_batch
    config = tf.ConfigProto(allow_soft_placement=False, log_device_placement=False) 
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        train_writer = tf.summary.FileWriter(CHECK_POINT_PATH, sess.graph)
        image_batch = read_image('D:\\Carole下载\\train2014', batch_size=BATCH_SIZE)
        tf.reshape(image_batch, [4, 256, 256, 3])


        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()

        threads = tf.train.start_queue_runners(sess, coord=coord)
        batch_step = 0
        try:
            while not coord.should_stop():
                batch_step += 1
                print(batch_step)
                
                image_data = sess.run(image_batch)
                # print(image_data.shape)
                _, batch_ls, style_ls, content_ls, summary_str = sess.run([train_step, loss, content_loss, style_loss, summary], feed_dict={batch_image: image_data})
                if batch_step % 10 == 0:
                    train_writer.add_summary(summary_str, batch_step)
                if batch_step % 100 == 0:
                    saver.save(sess, TRAIN_CHECK_POINT, global_step=batch_step)

                # image_data += CONFIG.MEANS
                # image_resized = np.asarray(image_data, dtype='uint8')
                # plt.imshow(image_resized)
                # plt.show()
        except tf.errors.OutOfRangeError:
            print('Complete')
        finally:
            coord.request_stop()
        coord.join(threads)
'''

