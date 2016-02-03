
from PIL import Image
import urllib, cStringIO

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile


INPUT_SIZE = 224
IMAGE_MEAN = 117;

with open("model/labels.txt", "r") as f:
	LABELS = [label.strip() for label in f.readlines()]
	print LABELS[0]

def create_graph():
	with gfile.FastGFile('model/model.pb', 'r') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
		_ = tf.import_graph_def(graph_def, name='')



create_graph()
sess = tf.Session()

def inference(imgDataList):

	imageData = np.array(imgDataList).reshape(1, INPUT_SIZE, INPUT_SIZE, 3).astype(np.float32) - IMAGE_MEAN

	y_tensor = sess.graph.get_tensor_by_name('output:0')

	y = sess.run(y_tensor, {'input:0': imageData})

	preds = np.squeeze(y)

	top3 = preds.argsort()[::-1][:3]

	res = ([ LABELS[i] for i in top3 ], [ preds[i] for i in top3 ])
	
	print res
	return res


def main():
	url = 'http://img.taopic.com/uploads/allimg/130228/240513-13022PR25222.jpg'
	res = inference(url)
	print res

if __name__ == "__main__":
	main()