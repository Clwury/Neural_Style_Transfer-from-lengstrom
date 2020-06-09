# from residual_style_transfer_optimize import *
from forward import *
from transform_1 import *

BATCH_SIZE = 1
CHANNEL_NUM = 3
MODEL_PATH = 'pretrained-model/imagenet-vgg-verydeep-19.mat'
CONTENT_WEIGHT = 7.5e0
STYLE_WEIGHT = 1e2
TV_WEIGHT = 2e2
LEARNING_RATE = 1e-3
CHECK_POINT_PATH = './logs'
TRAIN_CHECK_POINT = 'model/model.ckpt'

'''
def generate_content_tfrecord():
    """
    制作coco数据集的tfRecord文件
    """
    # 若文件夹不存在，则生成文件夹，并打印相关信息
    if not os.path.exists('./data'):
        os.makedirs('./data')
        print("the directory was created successful")
    else:
        print("directory already exists")
    write_content_tfrecord()


def write_content_tfrecord():
    # 定义writer对象
    writer = tf.python_io.TFRecordWriter(os.path.join('./data', 'coco_train.tfrecords'))
    num_pic = 0
    example_list = list()
    file_path_list = []
    # 读入coco原始数据集中文件路径集合
    for root, _, files in os.walk('D:/Carole下载/train2014'):
        for file in files:
            # 检查是否为图像文件
            if os.path.splitext(file)[1] not in ['.jpg', '.png', '.jpeg']:
                continue
            # 若是，则加入文件路径集合
            file_path = os.path.join(root, file)
            file_path_list.append(file_path)

    # 对路径集合进行打乱
    np.random.shuffle(file_path_list)
    for file_path in file_path_list:
        with Image.open(file_path) as img:
            # # 对coco数据集图片剪裁为正方形
            # img = center_crop_img(img)
            # # resize图片大小
            # img = img.resize((FLAGS.img_w, FLAGS.img_h))
            img = img.convert('RGB')
            img_raw = img.tobytes()
            # 为图像建Example
            example = tf.train.Example(features=tf.train.Features(feature={
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
            # 写入tfrecord文件
            num_pic += 1
            writer.write(example.SerializeToString())
            print('the number of picture:', num_pic)
            if num_pic == 100:
                break
    print('write tfrecord successful')


def read_content_tfrecord(path_tfrecord, image_size):
    # 创建文件队列
    filename_queue = tf.train.string_input_producer([path_tfrecord])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features={
        'img_raw': tf.FixedLenFeature([], tf.string)
    })
    # 解码图片数据
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    # 设置图片shape
    img.set_shape([image_size * image_size * 3])
    return img


def get_content_tfrecord(batch_size, path_tfrecord, image_size):
    """
    获取content_batch，用于训练
    :param batch_size:
    :param path_tfrecord: tfrecord存储路径
    :param image_size: 图片尺寸
    :return: content_batch op
    """
    img = read_content_tfrecord(path_tfrecord, image_size)
    img_batch = tf.train.shuffle_batch([img, ], batch_size=batch_size, num_threads=2, capacity=10, min_after_dequeue=1)
    return img_batch
'''
def backward(model_path):
    content = tf.placeholder(tf.float32, [BATCH_SIZE, CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH, CHANNEL_NUM])
    style = tf.placeholder(tf.float32, [1, CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH, CHANNEL_NUM])
    target = net(content/255.0)
    # target = net(content)

    vgg_target = load_vgg_model(MODEL_PATH, target)
    vgg_content = load_vgg_model(MODEL_PATH, content)
    vgg_style = load_vgg_model(MODEL_PATH, style)

    content_loss = compute_content_cost(vgg_content, vgg_target)
    style_loss = compute_style_cost(vgg_style, vgg_target)

    tv_loss = total_variation_loss(target)

    loss = content_loss * CONTENT_WEIGHT + style_loss * STYLE_WEIGHT + tv_loss * TV_WEIGHT

    tf.summary.scalar('losses/content_loss', CONTENT_WEIGHT * content_loss)
    tf.summary.scalar('losses/style_loss', STYLE_WEIGHT * style_loss)
    tf.summary.scalar('losses/tv_loss', tv_loss * TV_WEIGHT)
    tf.summary.scalar('losses/loss', loss)
    tf.summary.image('transform', target, max_outputs=1)  
    tf.summary.image('origin', content, max_outputs=1)    
    summary = tf.summary.merge_all()

    global_step = tf.Variable(0, trainable=False)
    opt = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss, global_step=global_step)

    content_batch = read_image('D:/Graduation_Project/other/train2014', batch_size=BATCH_SIZE)
    # 读取训练数据
    # content_batch = get_content_tfrecord(BATCH_SIZE, os.path.join("./data/", "coco_train.tfrecords"), CONFIG.IMAGE_HEIGHT)


    saver = tf.train.Saver(max_to_keep=10)

    config = tf.ConfigProto(allow_soft_placement=False, log_device_placement=False) 
    config.gpu_options.allow_growth = True
    
    batch_style = reshape_and_normalize_image('style_images/style_400x300.jpg')

    with tf.Session(config=config) as sess:
        train_writer = tf.summary.FileWriter(CHECK_POINT_PATH, sess.graph)
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        sess.run(tf.local_variables_initializer())

        # 在路径中查询有无checkpoint
        ckpt = tf.train.get_checkpoint_state(model_path)
        # 从checkpoint恢复模型
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Restore Model Successfully')
        else:
            print('No Checkpoint Found')

        # 开启多线程
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)


        # for i in range(10000):
        try:
            while not coord.should_stop():
                batch_content = sess.run(content_batch)
                batch_content = np.reshape(batch_content, [CONFIG.BATCH_SIZE, CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH, CHANNEL_NUM])


                sess.run(opt, feed_dict={content: batch_content, style: batch_style})
                step = sess.run(global_step)

                [loss_p, target_p, content_loss_res, style_loss_res, summary_str] = sess.run([loss, target, content_loss, style_loss, summary],  
                                                                                            feed_dict={content: batch_content, style: batch_style})
                
                print(step)

                if step % 100 == 0:
                    train_writer.add_summary(summary_str, step)
                    print('Save summary successful')

                    print("Iteration: %d, Loss: %e, Content_loss: %e, Style_loss: %e" %
                        (step, loss_p, content_loss_res, style_loss_res))
                if step % 500 == 0:
                    # target_p = target_p + CONFIG.MEANS   ##############################
                    # Image.fromarray(np.uint8(target_p[0, :, :, :])).save("output_images/" + str(step) + ".jpg")
                    save_image("output_images/" + str(step) + ".jpg", target_p)
                    print('Save image successful')
                if step % 200 == 0:
                    saver.save(sess, TRAIN_CHECK_POINT, global_step=step)
                    print('Save model successful')
        except tf.errors.OutOfRangeError:
            print('Complete')
        finally:
            coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    # generate_content_tfrecord()
    backward(model_path="./model/")