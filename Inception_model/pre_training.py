"""
pre_training

    Load the images from the folder
    Extract the feature vector
    Create the training data as .csv files with
                                features, labels, file names

"""

import tensorflow as tf
import os
import numpy as np
import sys
import tarfile
import cv2
import csv
from six.moves import urllib

from node_lookup import NodeLookup


MODEL_DIR = './model'
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'

CONFIG = {
    'num_top_predictions': 5,
    'x_data_csv':   'total_x.csv',
    'fn_csv':       'total_fn.csv',
    'label_csv': 'total_y.csv'
}


def create_graph():

    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(os.path.join(
            MODEL_DIR, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def maybe_download_and_extract():
    """Download and extract model tar file."""
    dest_directory = MODEL_DIR
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (
                filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def run_inference_on_image(img_path):
    """Runs inference on an image.
        Args:
            img_path: Image file name.

    Returns:
        Nothing
    """
    if not tf.gfile.Exists(img_path):
        tf.logging.fatal('File does not exist %s', img_path)
    image_data = tf.gfile.FastGFile(img_path, 'rb').read()

    # Creates graph from saved GraphDef.
    create_graph()

    with tf.Session() as sess:
        ###
        # Some useful tensors:
        # 'softmax:0': A tensor containing the normalized prediction across
        #               1000 labels.

        # 'pool_3:0': A tensor containing the next-to-last layer containing
        #               2048 float description of the image.

        # 'DecodeJpeg/contents:0': A tensor containing a string providing
        #               JPEG encoding of the image.

        # Runs the softmax tensor by feeding the image_data as input to the graph.
        ###
        softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
        predictions = sess.run(softmax_tensor,
                               {'DecodeJpeg/contents:0': image_data})
        predictions = np.squeeze(predictions)

        # Creates node ID --> English string lookup.
        node_lookup = NodeLookup()

        top_k = predictions.argsort()[-CONFIG['num_top_predictions']:][::-1]
        for node_id in top_k:
            human_string = node_lookup.id_to_string(node_id)
            score = predictions[node_id]
            print('%s (score = %.5f)' % (human_string, score))


sess, softmax_tensor = None, None


def get_feature_from_image(img_path):
    """Runs extract the feature from the image.
        Args:
            img_path: Image file name.

    Returns:
        predictions:
            2048 * 1 feature vector
    """
    global sess, softmax_tensor

    if not tf.gfile.Exists(img_path):
        tf.logging.fatal('File does not exist %s', img_path)
    image_data = tf.gfile.FastGFile(img_path, 'rb').read()

    predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
    predictions = np.squeeze(predictions)
    return predictions


def create_feature_csv(data_path, direction_info, session, softmax):

    dir_path = os.path.join(data_path, direction_info)

    feature_list = []
    label_list = []
    fn_list = []

    # valid_ext = ['.jpg', '.png']
    for f in os.listdir(dir_path):

        fn, ext = os.path.splitext(f)

        if ext.lower() == '.png':
            # Convert image file to jpg
            im = cv2.imread(os.path.join(dir_path, f))

            out_fn = os.path.join(dir_path, fn + '_o' + '.jpg')
            cv2.imwrite(out_fn, im)
            os.remove(os.path.join(dir_path, f))
        elif ext.lower() == '.jpg':
            out_fn = os.path.join(dir_path, f)

        # Extract the feature vector per each image
        feature = get_feature_from_image(out_fn)
        feature_list.append(feature)

        base_fn = os.path.basename(out_fn)
        fn_list.append([base_fn])

        label_list.append([direction_info])
        print(out_fn)

    print("Write X data to a csv file.")
    with open(os.path.join(dir_path, 'x.csv'), 'w', newline='') as fp:
        wr = csv.writer(fp, delimiter=',')
        wr.writerows(feature_list)

    print("Write file names list data to a csv file.")
    with open(os.path.join(dir_path, 'fn.csv'), 'w', newline='') as fp:
        wr = csv.writer(fp, delimiter=',')
        wr.writerows(fn_list)

    print("Write the label list data to a csv file.")
    with open(os.path.join(dir_path, 'y.csv'), 'w', newline='') as f:
        wr = csv.writer(f, quoting=csv.QUOTE_ALL)
        wr.writerows(label_list)

    return feature_list, fn_list, label_list


def main():
    global sess, softmax_tensor

    maybe_download_and_extract()
    # fn_img = './data/sample-cars/rear/13967556961.jpg'
    # run_inference_on_image(fn_img)

    # Creates graph from saved GraphDef.
    create_graph()
    sess = tf.Session()
    softmax_tensor = sess.graph.get_tensor_by_name('pool_3:0')

    data_path = './data/sample-cars'
    total_features = []
    total_labels = []
    total_fns = []

    directions = ['front', 'front 3 quarter', 'rear', 'rear 3 quarter', 'side']
    for direct_info in directions:
        features, fns, labels = create_feature_csv(data_path, direct_info, sess, softmax_tensor)
        total_features.extend(features)
        total_fns.extend(fns)
        total_labels.extend(labels)

    with open(CONFIG['x_data_csv'], 'w', newline='') as fp:
        wr = csv.writer(fp, delimiter=',')
        wr.writerows(total_features)

    with open(CONFIG['fn_csv'], 'w', newline='') as fp:
        wr = csv.writer(fp, delimiter=',')
        wr.writerows(total_fns)

    with open(CONFIG['label_csv'], 'w', newline='') as f:
        wr = csv.writer(f, quoting=csv.QUOTE_ALL)
        wr.writerows(total_labels)

if __name__ == '__main__':
    main()
