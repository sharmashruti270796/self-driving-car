import numpy
import scipy
import pandas as pd
import imageio
from PIL import Image

x = []
y = []


with open("data.txt") as f:
    for line in f:
        x.append("driving_dataset/" + line.split()[0]) 
        y.append(float(line.split()[1]) * scipy.pi / 180)

num_images = len(x)


x_train = x[:int(len(x) * 0.8)]
y_train = y[:int(len(x) * 0.8)]

x_test = x[-int(len(x) * 0.2):]
y_test = y[-int(len(x) * 0.2):]

train_len = len(x_train)
test_len = len(x_test)

train_pointer = 0
test_pointer = 0

def TrainBatches(size):
    global train_pointer
    x_out = []
    y_out = []
    for i in range(0, size):
        image_read = imageio.imread(x_train[(train_pointer + i) % train_len],pilmode ='RGB') 
        image_one = numpy.array(Image.fromarray(image_read))
        image_two = image_one[-150:]
        image_resize = (numpy.resize(image_two,(66,200,3)))/255.0
        
        x_out.append(image_resize)
    
        
        y_out.append([y_train[(train_pointer + i) % train_len]])
    train_pointer += size
    return x_out, y_out

def Batches(size):
    global test_pointer
    x_out = []
    y_out = []
    for i in range(0, size):
        image_read = imageio.imread(x_test[(train_pointer + i) % test_len],pilmode ='RGB') 
        image_one = numpy.array(Image.fromarray(image_read))
        image_two = image_one[-150:]
        image_resize = (numpy.resize(image_two,(66,200,3)))/255.0
        x_out.append(image_resize)
        y_out.append([y_test[(test_pointer + i) % test_len]])
    test_pointer += size
    return x_out, y_out

