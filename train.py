import json
import numpy
import gzip
import tensorflow as tf

with gzip.open('tmp/data.json.gz', 'rb') as f:
    data = json.load(f)
    indicators = data['indicators']
    trainX = numpy.asarray(data['trainX'])
    trainy = [
        {
            int(k): v
            for k, v in item.items()
        }
        for item in data['trainy']
    ]

print(indicators)

print('Num training examples = {}'.format(len(trainy)))

print('Building network...')

# Parameters
learning_rate = 0.01
num_steps = 1000
batch_size = 128
display_step = 100

# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
num_inputs = len(indicators) * 2
num_outputs = len(indicators)


# Define the neural network
def neural_net(x):
    regularizer = tf.contrib.layers.l2_regularizer(0.1)
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.layers.dense(x, n_hidden_1, kernel_regularizer=regularizer)
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.layers.dense(layer_1, n_hidden_2, kernel_regularizer=regularizer)
    # Output fully connected layer with a neuron for each indicator
    return [
        tf.layers.dense(layer_2, 1, kernel_regularizer=regularizer)
        for indicator in indicators
    ]

means = numpy.mean(trainX, axis=0)
variances = numpy.var(trainX, axis=0)

print(means, variances)

# Build the neural network
x = tf.placeholder(tf.float32, [1, num_inputs])
y = tf.placeholder(tf.float32)

inputs = (x - means) / variances
outputs = neural_net(inputs)

losses = [
    tf.losses.mean_squared_error(output, y) +
    tf.losses.get_regularization_loss()
    for i, output in enumerate(outputs)
]

optimizer = tf.train.GradientDescentOptimizer(learning_rate)

trains = [
    optimizer.minimize(l)
    for l in losses
]

def print_errors(sum, num):
    for i, n in enumerate(num):
        if n > 0:
            print('  - {}: {}'.format(indicators[i], sum[i]/n))

print("Training...")

iterations = 20000

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for k in range(iterations):
        if k % 100 == 0:
            if k > 0:
                print('iteration {} of {} examples'.format(k, len(trainX)))
                print_errors(errors, error_counts)

            errors = numpy.zeros(len(indicators))
            error_counts = numpy.zeros(len(indicators))

        i = numpy.random.randint(0, len(trainy))

        yi = trainy[i]

        xi = numpy.resize(trainX[i], [1, num_inputs])

        for j, v in yi.items():
            train = trains[j]
            loss = losses[j]

            feed_dict = {
                x: xi,
                y: numpy.float32((v-means[j*2])/variances[j*2])
            }
            val = sess.run([train, loss], feed_dict=feed_dict)

            """
            print(j, val[1], feed_dict)
            if k > 10:
                exit()
            """

            errors[j] += float(val[1])
            error_counts[j] += 1

print_errors(errors, error_counts)
