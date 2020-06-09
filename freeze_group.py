import tensorflow as tf
import imageio
from PIL import Image
import numpy as np
from nst_utils import *

def freeze_group(checkpoint_path, output_pb):
    output_node_name = 'add_37'
    saver = tf.train.import_meta_graph(checkpoint_path +'.meta')
    graph = tf.get_default_graph() # 获得默认的图
    input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图

    config = tf.ConfigProto(allow_soft_placement=False, log_device_placement=False) 
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        saver.restore(sess, checkpoint_path)
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess=sess,
            input_graph_def=input_graph_def,
            output_node_names=output_node_name.split(',')
        )
        with tf.gfile.GFile(output_pb, 'wb') as f:
            f.write(output_graph_def.SerializeToString())
        
        print("%d ops in the final graph." % len(output_graph_def.node)) #得到当前图有几个操作节点
        # for op in output_graph_def.get_operations():
        #     print(op.name, op.values())


def to_tflite(pb_path, tflite_path):

    graph_def_file = pb_path
    converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays=['Placeholder'], output_arrays=['add_37'])
    converter.post_training_quantize = True
    tflite_model = converter.convert()
    open(tflite_path, "wb").write(tflite_model)

def read_tflite(tflite_path):
    image = imageio.imread("content_images/chicago.jpg")
    
    image = np.array(Image.fromarray(image).resize((512, 512)))
    # image = reshape_and_normalize_image("content_images/stata.jpg")
    shape = image.shape
    print(shape)
    tflite = tf.lite.Interpreter(model_path=tflite_path)
    print('read successful')
    tflite.allocate_tensors()
    image_expend = np.expand_dims(image, axis=0)
    image_expend = image_expend.astype('float32')
    print(image_expend.shape)

    # tflite.resize_tensor_input(input_index=tflite.get_input_details()[0]['index'], tensor_size = (1, shape[0], shape[1], shape[2]))
    # tflite.resize_tensor_input(input_index=tflite.get_output_details()[0]['index'], tensor_size = (1, shape[0], shape[1], shape[2]))
    print(tflite.get_input_details())

    print(tflite.get_output_details())
    tflite.set_tensor(tflite.get_input_details()[0]['index'], image_expend/255.0)
    print('set successful')

    tflite.invoke()
    print('invoke successful')
    output = tflite.get_tensor(tflite.get_output_details()[0]['index'])
    print(output)

    save_image('../other/test_model/test_image/tflite_convert7.jpg', output)    

if __name__ == '__main__':
    model_path = "../other/test_model/batch_size=1poppy_field/model.ckpt-165400"
    pb_path = "../other/test_model/batch_size=1poppy_field/frozen.pb"
    tflite_path = "../other/test_model/batch_size=1poppy_field/converted_model.tflite"
    # tflite_path = "converted_model.tflite"
    # freeze_group(model_path, pb_path)
    print('finish')
    # to_tflite(pb_path, tflite_path)
    read_tflite(tflite_path)

    # model = tf.lite.Interpreter(model_path="converted_model.tfile")
    # model.allocate_tensors()
    # model.resize_tensor_input(input_index=model.get_input_details()[0]['index'], tensor_size=(1, 1000, 400, 3))
    # model.resize_tensor_input(input_index=model.get_output_details()[0]['index'], tensor_size=(1, 1000, 400, 3))
    # print(model.get_input_details())
    # print(model.get_output_details())
