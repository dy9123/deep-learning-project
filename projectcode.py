
from sklearn.preprocessing import normalize
import pandas as pd
import numpy as np
import tensorflow as tf
import random
df=pd.read_csv('RawDatamodified.csv',sep=';')
d = df.values
ntotal=len(df)


d = normalize(d, axis=0, norm='l2')


resultu = []
# Return 100 results (for instance)
for i in range(ntotal):
    res = random.random()
    if res < 0.1:
        resultu.append(1)
    elif res < 0.2 and res>=0.1:
        resultu.append(2)
    elif res < 0.3 and res>=0.2:
        resultu.append(3)
    elif res < 0.4 and res>=0.3:
        resultu.append(4)
    elif res < 0.5 and res>=0.4:
        resultu.append(5)
    elif res < 0.6 and res>=0.5:
        resultu.append(6)
    elif res < 0.7 and res>=0.6:
        resultu.append(7)
    elif res < 0.8 and res>=0.7:
        resultu.append(8)
    else:
        resultu.append(9)
resultu=np.array(resultu)
trainset=[]
for i in range(1,8):
    toinsert=d[resultu==i].astype(np.float32)
    trainset.append(toinsert)
validationset=d[resultu==8].astype(np.float32)
testset=d[resultu==9].astype(np.float32)


#x = data

# Following Hinton-Salakhutdinov Architecture

# 3 hidden layers for encoder
n_encoder_h_1 = 19
n_encoder_h_2 = 16
n_encoder_h_3 = 14

n_encoder_h_4 = 12
#n_encoder_h_5 = 10


# 3 hidden layers for decoder
#n_decoder_h_1 = 10
n_decoder_h_1 = 12
n_decoder_h_2 = 14
n_decoder_h_3 = 16
n_decoder_h_4 = 19


# Parameters
learning_rate = 0.01
training_epochs = 50
#batch_size = 7
display_step = 1

total_batch=7


def layer_batch_normalization(x, n_out, phase_train):
    """
    Defines the network layers
    input:
        - x: input vector of the layer
        - n_out: integer, depth of input maps - number of sample in the batch 
        - phase_train: boolean tf.Varialbe, true indicates training phase
    output:
        - batch-normalized maps   
    
    
    """

    beta_init = tf.constant_initializer(value=0.0, dtype=tf.float32)
    beta = tf.get_variable("beta", [n_out], initializer=beta_init)
    
    gamma_init = tf.constant_initializer(value=1.0, dtype=tf.float32)
    gamma = tf.get_variable("gamma", [n_out], initializer=gamma_init)

    #tf.nn.moment: https://www.tensorflow.org/api_docs/python/tf/nn/moments
    #calculate mean and variance of x
    batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')
    
    #tf.train.ExponentialMovingAverage:
    #https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    #Maintains moving averages of variables by employing an exponential decay.
    ema = tf.train.ExponentialMovingAverage(decay=0.9)
    ema_apply_op = ema.apply([batch_mean, batch_var])
    ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)
    
    def mean_var_with_update():
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)
        
    #tf.cond: https://www.tensorflow.org/api_docs/python/tf/cond
    #Return true_fn() if the predicate pred is true else false_fn()
    mean, var = tf.cond(phase_train, mean_var_with_update, lambda: (ema_mean, ema_var))
    
    
    #Here, we changesd the shape of x into [[[x1]],[[x2] ],....]

    reshaped_x = tf.reshape(x, [-1, 1, 1, n_out])
    normed = tf.nn.batch_norm_with_global_normalization(reshaped_x, mean, var, beta, gamma, 1e-3, True)
    
    return tf.reshape(normed, [-1, n_out])

def layer(x, weight_shape, bias_shape, phase_train):
    
    """
    Defines the network layers
    input:
        - x: input vector of the layer
        - weight_shape: shape the the weight maxtrix
        - bias_shape: shape of the bias vector
        - phase_train: boolean tf.Varialbe, true indicates training phase
    output:
        - output vector of the layer after the matrix multiplication and non linear transformation
    """
    
    #initialize weights
    weight_init = tf.random_normal_initializer(stddev=(1.0/weight_shape[0])**0.5)
    W = tf.get_variable("W", weight_shape, initializer=weight_init)
    
    bias_init = tf.constant_initializer(value=0)
    b = tf.get_variable("b", bias_shape, initializer=bias_init)

    logits = tf.matmul(x, W) + b
    
    #apply the non-linear function after the batch normalization
    return tf.nn.sigmoid(layer_batch_normalization(logits, weight_shape[1], phase_train))
def encoder(x, n_code, phase_train):
    
    """
    Defines the network encoder part
    input:
        - x: input vector of the encoder
        - n_code: number of neurons in the code layer (output of the encoder - input of the decoder)
        - phase_train: boolean tf.Varialbe, true indicates training phase
    output:
        - output vector: reduced dimension
    """
    
    with tf.variable_scope("encoder"):
        
        with tf.variable_scope("h_1"):
            h_1 = layer(x, [19, n_encoder_h_1], [n_encoder_h_1], phase_train)

        with tf.variable_scope("h_2"):
            h_2 = layer(h_1, [n_encoder_h_1, n_encoder_h_2], [n_encoder_h_2], phase_train)

        with tf.variable_scope("h_3"):
            h_3 = layer(h_2, [n_encoder_h_2, n_encoder_h_3], [n_encoder_h_3], phase_train)
        with tf.variable_scope("h_4"):
            h_4 = layer(h_3, [n_encoder_h_3, n_encoder_h_4], [n_encoder_h_4], phase_train)
      

        with tf.variable_scope("code"):
            output = layer(h_4, [n_encoder_h_4, n_code], [n_code], phase_train)

    return output

def decoder(x, n_code, phase_train):
    """
    Defines the network encoder part
    input:
        - x: input vector of the decoder - reduced dimension vector
        - n_code: number of neurons in the code layer (output of the encoder - input of the decoder) 
        - phase_train: boolean tf.Varialbe, true indicates training phase
    output:
        - output vector: reconstructed dimension of the initial vector
    """
    
    with tf.variable_scope("decoder"):
        
        with tf.variable_scope("h_1"):
            h_1 = layer(x, [n_code, n_decoder_h_1], [n_decoder_h_1], phase_train)

        with tf.variable_scope("h_2"):
            h_2 = layer(h_1, [n_decoder_h_1, n_decoder_h_2], [n_decoder_h_2], phase_train)

        with tf.variable_scope("h_3"):
            h_3 = layer(h_2, [n_decoder_h_2, n_decoder_h_3], [n_decoder_h_3], phase_train)
        with tf.variable_scope("h_4"):
            h_4 = layer(h_3, [n_decoder_h_3, n_decoder_h_4], [n_decoder_h_4], phase_train)

        with tf.variable_scope("output"):
            output = layer(h_4, [n_decoder_h_4, 19], [19], phase_train)

    return output
def loss(output, x):
    """
    Compute the loss of the auto-encoder
    
    intput:
        - output: the output of the decoder
        - x: true value of the sample batch - this is the input of the encoder
        
        the two have the same shape (batch_size * num_of_classes)
    output:
        - loss: loss of the corresponding batch (scalar tensor)
    
    """
    
    with tf.variable_scope("training"):
        
        l2 = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(output, x)), 1))
        train_loss = tf.reduce_mean(l2)
        train_summary_op = tf.summary.scalar("train_cost", train_loss)
        return train_loss, train_summary_op
def training(cost, global_step):
    """
    defines the necessary elements to train the network
    
    intput:
        - cost: the cost is the loss of the corresponding batch
        - global_step: number of batch seen so far, it is incremented by one 
        each time the .minimize() function is called
    """
    
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam')
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.005)
    train_op = optimizer.minimize(cost, global_step=global_step)
    return train_op
def evaluate(output, x):
    """
    evaluates the accuracy on the validation set 
    input:
        -output: prediction vector of the network for the validation set
        -x: true value for the validation set
    output:
        - val_loss: loss of the autoencoder
        - in_image_op: input image 
        - out_image_op:reconstructed image 
        - val_summary_op: summary of the loss
    """
    
    with tf.variable_scope("validation"):
        
        #in_image_op = image_summary("input_image", x)
        
        #out_image_op = image_summary("output_image", output)
        
        l2_norm = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(output, x, name="val_diff")), 1))
        
        val_loss = tf.reduce_mean(l2_norm)
        
        val_summary_op = tf.summary.scalar("val_cost", val_loss)
        
        #return val_loss, in_image_op, out_image_op, val_summary_op
        return val_loss,  val_summary_op
"""
def image_summary(label, tensor):
    #tf.summary.image: https://www.tensorflow.org/api_docs/python/tf/summary/image
    #Outputs a Summary protocol buffer with images.

    tensor_reshaped = tf.reshape(tensor, [-1, 19, 1, 1])
    return tf.summary.image(label, tensor_reshaped)
"""
if __name__ == '__main__':
    print('we begin')

    #if a python file, please use the 4 lines bellow and comment the "n_code = '2'"
    #parser = argparse.ArgumentParser(description='Autoencoder')
    #parser.add_argument('n_code', nargs=1, type=str)
    #args = parser.parse_args(['--help'])
    #n_code = args.n_code[0]
    
    #if a jupyter file, please comment the 4 above and use the one bellow
    n_code = '10'
    
    #feel free to change with your own 
    #log_files_path = r'C:\Users\yy2895\Desktop\pys'
    log_files_path = 'C:/Users/yy2895/Desktop/pys'
    with tf.Graph().as_default():

        with tf.variable_scope("autoencoder_model"):

            #the input variables are first define as placeholder 
            # a placeholder is a variable/data which will be assigned later 
            # image vector & label, phase_train is a boolean 
            x = tf.placeholder("float", [None, 19]) # MNIST data image of shape 28*28=784
            
            phase_train = tf.placeholder(tf.bool)
            
            #define the encoder 
            code = encoder(x, int(n_code), phase_train)
            
            #define the decoder
            output = decoder(code, int(n_code), phase_train)
            
            #compute the loss 
            cost, train_summary_op = loss(output, x)

            #initialize the value of the global_step variable 
            # recall: it is incremented by one each time the .minimise() is called
            global_step = tf.Variable(0, name='global_step', trainable=False)

            train_op = training(cost, global_step)

            #evaluate the accuracy of the network (done on a validation set)
            #eval_op, in_image_op, out_image_op, val_summary_op = evaluate(output, x)
            eval_op, val_summary_op = evaluate(output, x)

            summary_op = tf.summary.merge_all()

            #save and restore variables to and from checkpoints.
            #saver = tf.train.Saver(max_to_keep=200)
            saver = tf.train.Saver()

            #defines a session
            sess = tf.Session()

            # summary writer
            #https://www.tensorflow.org/api_docs/python/tf/summary/FileWriter
            train_writer = tf.summary.FileWriter('mnist_autoencoder_hidden_' + n_code + '_logs/', graph=sess.graph)

            val_writer   = tf.summary.FileWriter('mnist_autoencoder_hidden_' + n_code + '_logs/', graph=sess.graph)

            #initialization of the variables
            init_op = tf.global_variables_initializer()

            sess.run(init_op)
            
            # Training cycle
            for epoch in range(training_epochs):

                avg_cost = 0.
                #total_batch = int(numberofsamples/batch_size)
                
                # Loop over all batches
                for i in range(total_batch):
                    
                    minibatch_x= trainset[i]
                    
                    # Fit training using batch data
                    #the training is done using the training dataset
                    _, new_cost, train_summary = sess.run([train_op, cost, train_summary_op], feed_dict={x: minibatch_x, phase_train: True})
                    
                    train_writer.add_summary(train_summary, sess.run(global_step))
                    
                    # Compute average loss
                    avg_cost += new_cost/total_batch
                    
                
                # Display logs per epoch step
                if epoch % display_step == 0:
                    print("Epoch:", '%04d' % (epoch+1), "cost =", "{:0.9f}".format(avg_cost))

                    #the accuracy is evaluated using the validation dataset
                    train_writer.add_summary(train_summary, sess.run(global_step))

                    #validation_loss, in_image, out_image, val_summary = sess.run([eval_op, in_image_op, out_image_op, val_summary_op], feed_dict={x: validationset, phase_train: False})
                    validation_loss,  val_summary = sess.run([eval_op, val_summary_op], feed_dict={x: validationset, phase_train: False})
                    #val_writer.add_summary(in_image, sess.run(global_step))
                    #val_writer.add_summary(out_image, sess.run(global_step))
                    val_writer.add_summary(val_summary, sess.run(global_step))
                    print("Validation Loss:", validation_loss)

                    #save to use later
                    #https://www.tensorflow.org/api_docs/python/tf/train/Saver
                    #saver.save(sess, log_files_path+'model-checkpoint', global_step=global_step)
                    saver.save(sess, log_files_path+'mnist_autoencoder_hidden_' + n_code + '_logs/model-checkpoint-' + '%04d' % (epoch+1), global_step=global_step)


            print("Optimization Finished!")

            test_loss = sess.run(eval_op, feed_dict={x: testset, phase_train: False})
            #nps_loss = sess.run(eval_op, feed_dict={x: mnist.validation.images, phase_train: False})
            print("Test Loss:", test_loss)
        
