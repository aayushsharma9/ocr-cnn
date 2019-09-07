"""
Usage:
  python generateTFRecords.py --csv_input=hand_labels.csv --image_dir=images --output_path=hand.record
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import io
import pandas as pd
import tensorflow as tf

from PIL import Image
from collections import namedtuple, OrderedDict
sys.path.append('/home/chaitanya_anand/Work/models/research/')
from object_detection.utils import dataset_util

flags = tf.app.flags
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('image_dir', '', 'Path to images')
FLAGS = flags.FLAGS


def class_text_to_int(row_label):
    switcher = {
        '0': 0,
        '1': 1,
        '2': 2,
        '3': 3,
        '4': 4,
        '5': 5,
        '6': 6,
        '7': 7,
        '8': 8,
        '9': 9,
        '0': 10,
        'a': 11,
        'b': 12,
        'c': 13,
        'd': 14,
        'e': 15,
        'f': 16,
        'g': 17,
        'h': 18,
        'i': 19,
        'j': 20,
        'k': 21,
        'l': 22,
        'm': 23,
        'n': 24,
        'o': 25,
        'p': 26,
        'q': 27,
        'r': 28,
        's': 29,
        't': 30,
        'u': 31,
        'v': 32,
        'w': 33,
        'x': 34,
        'y': 35,
        'z': 36,
        'A': 37,
        'B': 38,
        'C': 39,
        'D': 40,
        'E': 41,
        'F': 42,
        'G': 43,
        'H': 44,
        'I': 45,
        'J': 46,
        'K': 47,
        'L': 48,
        'M': 49,
        'N': 50,
        'O': 51,
        'P': 52,
        'Q': 53,
        'R': 54,
        'S': 55,
        'T': 56,
        'U': 57,
        'V': 58,
        'W': 59,
        'X': 60,
        'Y': 61,
        'Z': 62
    }
    return switcher.get(row_label, None)

def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.gfile.GFile('{}'.format(group.filename), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'png'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    path = os.path.join(FLAGS.image_dir)
    examples = pd.read_csv(FLAGS.csv_input)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    tf.app.run()