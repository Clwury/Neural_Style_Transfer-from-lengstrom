from nst_utils import *

STYLE_LAYERS = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

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
    content_loss = tf.nn.l2_loss(a_G - a_C) * 2/tf.cast(tf.size(a_C), tf.float32)
    return content_loss
    ### END CODE HERE ###


def gram_matrix(A):
    """
    Argument:
    A -- matrix of shape (n_C, n_H*n_W)
    
    Returns:
    GA -- Gram matrix of A, of shape (n_C, n_C)
    """
    # batch_size, height, width, channels = [i for i in A.shape]
    # size = height * width * channels
    # features = tf.reshape(A, (batch_size, height * width, channels))
    # features_T = tf.transpose(features, perm=[0,2,1])
    # GA = tf.matmul(features_T, features)/tf.cast(size, tf.float32)
    ### START CODE HERE ### (≈1 line)
    # height = tf.shape(A)[1]
    # width = tf.shape(A)[2]
    # channels = tf.shape(A)[3]
    # print(height * width * channels)
    GA = tf.matmul(tf.transpose(A), A)
    ### END CODE HERE ###/tf.cast(height * width * channels, tf.float32)/tf.cast(tf.size(A), tf.float32)
    
    return GA

# def gram_matrix(A):
#     """
#     Argument:
#     A -- matrix of shape (n_C, n_H*n_W)
    
#     Returns:
#     GA -- Gram matrix of A, of shape (n_C, n_C)
#     """
    
#     ### START CODE HERE ### (≈1 line)
#     GA = tf.matmul(A,tf.transpose(A))
#     ### END CODE HERE ###
    
#     return GA

# def compute_layer_style_cost(a_S, a_G):
#     """
#     Arguments:
#     a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S 
#     a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G
    
#     Returns: 
#     J_style_layer -- tensor representing a scalar value, style cost defined above by equation (2)
#     """
    
#     ### START CODE HERE ###
#     # Retrieve dimensions from a_G (≈1 line)
#     m, n_H, n_W, n_C = a_G.get_shape().as_list()
    
#     # Reshape the images to have them of shape (n_C, n_H*n_W) (≈2 lines)
#     a_S = tf.reshape(a_S,(n_H*n_W,n_C))
#     a_G = tf.reshape(a_G,(n_H*n_W,n_C))

#     # Computing gram_matrices for both images S and G (≈2 lines)
#     GS = gram_matrix(tf.transpose(a_S))
#     GG = gram_matrix(tf.transpose(a_G))

#     J_style_layer = 1/(4 * (n_C **2) * ((n_H * n_W) **2)) * tf.reduce_sum(tf.square(tf.subtract(GS, GG)))

#     return J_style_layer

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
    a_S = tf.reshape(a_S,(n_H*n_W,n_C))
    a_G = tf.reshape(a_G,(n_H*n_W,n_C))

    # Computing gram_matrices for both images S and G (≈2 lines)
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    # J_style_layer = 1/(4 * (n_C **2) * ((n_H * n_W) **2)) * tf.reduce_sum(tf.square(tf.subtract(GS, GG)))     tf.cast(tf.size(GS), tf.float32)
    J_style_layer = tf.nn.l2_loss(GG - GS) * 2/(4 * (n_C **2) * ((n_H * n_W) **2))
    return J_style_layer

def compute_style_cost(model1, model2, STYLE_LAYERS):
    J_style = 0

    for layer_name in STYLE_LAYERS:
        # out = model[layer_name]
        # a_S = sess.run(out)
        # a_G = out
        a_G = model1[layer_name]
        a_S = model2[layer_name]
        J_style_layer = compute_layer_style_cost(a_S, a_G)
        J_style += 0.2 * J_style_layer
    return J_style




# # Creates a graph.
# a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
# b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
# c = tf.matmul(a, b)
# # Creates a session with log_device_placement set to True.
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# # Runs the op.
# print(sess.run(c))



# img = tf.Variable(tf.random_normal([16,256,256,64]))
# mean, variance = tf.nn.moments(img, [1,2], keep_dims=True)
# print(mean.shape, variance.shape)

# a = tf.constant([[1,2],[3,4]])
# b = tf.constant([1,1])
# print(sess.run(tf.reshape(a-b, [1, 2, 2])))


# # image = tf.zeros([4, 256, 256, 3])
# # vgg = load_vgg_model('pretrained-model/imagenet-vgg-verydeep-19.mat', image)
# # print(vgg['conv4_2'])

# # style_image_batch = []
# # for i in range(4):
# #     print(style.shape)
# #     style_image_batch.append(style)
# # image = np.array(style_image_batch).astype(np.float32)
# # print(image.shape)
# a = tf.constant([[[[1,2],[3,4]], [[1,2],[3,4]]], [[[1,2],[3,4]], [[1,2],[3,4]]]], dtype=tf.float32)
# print(a.shape)
# print(gram_matrix(a).eval(session=sess))


# style = reshape_and_normalize_image('output_images/160.png')
# style1 = reshape_and_normalize_image('output_images/160.png')
# print(style.shape)
# vgg1 = load_vgg_model('pretrained-model/imagenet-vgg-verydeep-19.mat', style)
# vgg2 = load_vgg_model('pretrained-model/imagenet-vgg-verydeep-19.mat', style1)
# style_loss = compute_content_cost(vgg1['conv4_2'], vgg2['conv4_2'])
# print(style_loss.eval(session=sess))
# # image_resized = np.asarray(vgg1['conv1_1'], dtype='uint8')
# # plt.imshow(image_resized)
# # plt.show()

# #####################################################################
'''
sess = tf.Session()
# content_image = imageio.imread('content_images/amber.jpg')
content_image = reshape_and_normalize_image('content_images/amber.jpg')

# style_image = imageio.imread("style_images/style_400x300.jpg")
style_image = reshape_and_normalize_image("style_images/starry_night.jpg")

generated_image = generate_noise_image(content_image)
# imshow(generated_image[0])

model = load_vgg_model('pretrained-model/imagenet-vgg-verydeep-19.mat')
# sess.run(model['input'].assign(content_image))
# 计算内容最小损失
sess.run(model['input'].assign(content_image))
a_C = sess.run(model['conv4_2'])
# a_C = model['conv4_2']
# model1 = load_vgg_model('pretrained-model/imagenet-vgg-verydeep-19.mat', generated_image)
# sess.run(model['input'].assign(generated_image))
# sess.run(model1)
a_G = model['conv4_2']

J_content = compute_content_cost(a_C, a_G)

# 计算样式最小损失

# model2 = load_vgg_model('pretrained-model/imagenet-vgg-verydeep-19.mat', style_image)
sess.run(model['input'].assign(style_image))
model2 = sess.run(model)
J_style = compute_style_cost(model, model2, STYLE_LAYERS)

# 总计最小损失
J = 1 * J_content + 4 * J_style
optimizer = tf.train.AdamOptimizer(2.0)
train_step = optimizer.minimize(J)

def model_nn(sess, input_image, num_iterations = 1000):
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    generated_image = sess.run(model['input'].assign(input_image))

    for i in range(num_iterations):
        sess.run(train_step)

        generated_image = sess.run(model['input'])

        if i%20 == 0:
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print('Iterstion' + str(i) + ':')
            print('total_cost = ' + str(Jt))
            print('content_cost = ' + str(J_content.eval(session=sess)))
            print('style_cost = ' + str(J_style.eval(session=sess)))

            save_image(CONFIG.OUTPUT_DIR + str(i) + '.png', generated_image)

        save_image(CONFIG.OUTPUT_DIR + 'gengeated_image.jpg', generated_image)

    return generated_image
'''
# model_nn(sess, generated_image)
image = tf.image.decode_jpeg('D:/Carole下载/train2014/COCO_train2014_000000000671.jpg')
image_resize = tf.image.resize_images(image, [256, 256])
image_resize.set_shape([256, 256, 3])
print(image_resize.shape)
print('/////////////////')



