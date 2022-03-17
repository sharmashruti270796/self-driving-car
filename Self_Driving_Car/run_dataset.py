#pip3 install opencv-python

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import scipy.misc
import CNN
import cv2
from subprocess import call
import math
import imageio
import numpy
from tensorflow import keras


sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, "save/CNN.ckpt")

img = cv2.imread('steering_wheel_image.jpg',0)
rows,cols = img.shape

smoothed_angle = 0


#read data.txt
xs = []
ys = []
with open("data.txt") as f:
    for line in f:
        xs.append("driving_dataset/" + line.split()[0])

        ys.append(float(line.split()[1]) * scipy.pi / 180)

num_images = len(xs)


i = math.ceil(num_images*0.8)
print("Starting frameofvideo:" +str(i))
output_degree = []
while(cv2.waitKey(10) != ord('q')):
    full_image = imageio.imread("C:/Users/91886/Desktop/Applied AI/selfdrivingcar/self-driving-car-project/driving_dataset/" + str(i) + ".jpg", pilmode="RGB")

    image = (numpy.resize(full_image,(66,200,3)))/255.0
    degrees = CNN.output.eval(feed_dict={CNN.x: [image], CNN.keep_prob: 1.0})[0][0] * 180.0 / scipy.pi
    output_degree.append(degrees)

    print("Steering angle: " + str(degrees) + " (pred)\t" + str(ys[i]*180/scipy.pi) + " (actual)")
    cv2.imshow("frame", cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR))

    smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)
    M = cv2.getRotationMatrix2D((cols/2,rows/2),-smoothed_angle,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    cv2.imshow("steering wheel", dst)
    i += 1


model = keras.models.Model(ys, output_degree)
model.compile('sgd', loss='mse', metrics=[tf.keras.metrics.Accuracy()])


cv2.destroyAllWindows()
