import tensorflow as tf
import imageio
from PIL import Image
import numpy as np
from transform_1 import *

IMAGE_PATH = "./content_images/turtle.jpg"
image = imageio.imread(IMAGE_PATH)
t_shape = image.shape
image = np.reshape(image, ((1,) + image.shape))

test_image = tf.placeholder(tf.float32, [1, t_shape[0], t_shape[1], t_shape[2]])
transform_img = net(test_image/255.0)
# transform_img_ = net(test_image)

saver = tf.train.Saver()

config = tf.ConfigProto(allow_soft_placement=False, log_device_placement=False) 
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    # train_writer = tf.summary.FileWriter(CHECK_POINT_PATH, sess.graph)

    sess.run(tf.global_variables_initializer())

    ckpt = tf.train.get_checkpoint_state('./model/')       # 通过检查点文件锁定最新的模型
    if ckpt and ckpt.model_checkpoint_path:
        # saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path +'.meta') # 载入图结构，保存在.meta文件中
        saver.restore(sess,ckpt.model_checkpoint_path)      # 载入参数，参数保存在两个文件中，不过restore会自己寻找
        print('Restore Model Successfully')
    else:
        print('No Checkpoint Found')
    style_image = sess.run(transform_img, feed_dict={test_image: image})

    # saver.save(sess, TRAIN_CHECK_POINT)

    # style_image[0] += CONFIG.MEANS
    save_image("output_images/generated3.jpg", style_image)
    
    image = np.clip(style_image[0], 0, 255).astype('uint8')
    # image_ = np.clip(style_image_[0], 0, 255).astype('uint8')
    image_resized = np.asarray(image, dtype='uint8')
    # image_resized_ = np.asarray(image_, dtype='uint8')
    # plt.imshow(image_resized)
    # plt.show()
    # plt.close()
    exit(0)