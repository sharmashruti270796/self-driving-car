
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import os
import tensorflow.compat.v1 as tf
from tensorflow.core.protobuf import saver_pb2
import data
import CNN



tf.disable_v2_behavior()


LOGDIR = './save'




sess = tf.InteractiveSession()

l2normalization = 0.001

train_vars = tf.trainable_variables()

loss = tf.reduce_mean(tf.square(tf.subtract(CNN.y_, CNN.output))) + tf.add_n([tf.nn.l2_loss(v) for v in train_vars]) * l2normalization
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
sess.run(tf.initialize_all_variables())

tf.summary.scalar("loss", loss)
merged_summary_op =  tf.summary.merge_all()

saver = tf.train.Saver(write_version = saver_pb2.SaverDef.V1)

logs_path = './logs'
summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

epochs = 30
size = 100

each_epoch_loss = []
for epoch in range(epochs):
  count = 0
  for i in range(int(data.num_images/size)):
    x, y = data.TrainBatches(size)
    train_step.run(feed_dict={CNN.x: x, CNN.y_: y, CNN.keep_prob: 0.8})
    if i % 10 == 0:
      x, y = data.Batches(size)
      loss_value = loss.eval(feed_dict={CNN.x:x, CNN.y_: y, CNN.keep_prob: 1.0})
      print("Epoch: %d, Step: %d, Loss: %g" % (epoch, epoch * size + i, loss_value))
      count+=loss_value
    
    summary = merged_summary_op.eval(feed_dict={CNN.x:x, CNN.y_: y, CNN.keep_prob: 1.0})
    summary_writer.add_summary(summary, epoch * data.num_images/size + i)

    if i % size == 0:
      if not os.path.exists(LOGDIR):
        os.makedirs(LOGDIR)
      checkpoint_path = os.path.join(LOGDIR, "CNN.ckpt")
      filename = saver.save(sess, checkpoint_path)
  print("Model saved in file: %s" % filename)
  count = count/46
  each_epoch_loss.append(count)
  print(each_epoch_loss)
  plt.plot(epoch,each_epoch_loss)
  plt.show()
