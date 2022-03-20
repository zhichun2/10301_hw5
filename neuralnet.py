import numpy as np
import argparse
import logging

parser = argparse.ArgumentParser()
parser.add_argument('train_input', type=str,
                    help='path to training input .csv file')
parser.add_argument('validation_input', type=str,
                    help='path to validation input .csv file')
parser.add_argument('train_out', type=str,
                    help='path to store prediction on training data')
parser.add_argument('validation_out', type=str,
                    help='path to store prediction on validation data')
parser.add_argument('metrics_out', type=str,
                    help='path to store training and testing metrics')
parser.add_argument('num_epoch', type=int,
                    help='number of training epochs')
parser.add_argument('hidden_units', type=int,
                    help='number of hidden units')
parser.add_argument('init_flag', type=int, choices=[1, 2],
                    help='weight initialization functions, 1: random')
parser.add_argument('learning_rate', type=float,
                    help='learning rate')
parser.add_argument('--debug', type=bool, default=False,
                    help='set to True to show logging')


def args2data(parser):
    """
    Parse argument, create data and label.
    :return:
    X_tr: train data (numpy array)
    y_tr: train label (numpy array)
    X_te: test data (numpy array)
    y_te: test label (numpy array)
    out_tr: predicted output for train data (file)
    out_te: predicted output for test data (file)
    out_metrics: output for train and test error (file)
    n_epochs: number of train epochs
    n_hid: number of hidden units
    init_flag: weight initialize flag -- 1 means random, 2 means zero
    lr: learning rate
    """

    # # Get data from arguments
    out_tr = parser.train_out
    out_te = parser.validation_out
    out_metrics = parser.metrics_out
    n_epochs = parser.num_epoch
    n_hid = parser.hidden_units
    init_flag = parser.init_flag
    lr = parser.learning_rate

    X_tr = np.loadtxt(parser.train_input, delimiter=',')
    y_tr = X_tr[:, 0].astype(int)
    X_tr[:, 0] = 1.0 #add bias terms

    X_te = np.loadtxt(parser.validation_input, delimiter=',')
    y_te = X_te[:, 0].astype(int)
    X_te[:, 0]= 1.0 #add bias terms


    return (X_tr, y_tr, X_te, y_te, out_tr, out_te, out_metrics,
            n_epochs, n_hid, init_flag, lr)



def shuffle(X, y, epoch):
    """
    Permute the training data for SGD.
    :param X: The original input data in the order of the file.
    :param y: The original labels in the order of the file.
    :param epoch: The epoch number (0-indexed).
    :return: Permuted X and y training data for the epoch.
    """
    np.random.seed(epoch)
    N = len(y)
    ordering = np.random.permutation(N)
    return X[ordering], y[ordering]

# returns the weight with folded bias
def random_init(shape):
    """
    Randomly initialize a numpy array of the specified shape
    :param shape: list or tuple of shapes
    :return: initialized weights
    """
    # DO NOT CHANGE THIS
    np.random.seed(np.prod(shape))
    
    # Implement random initialization here
    # temp_shape = [shape[0], shape[1]-1]
    # temp = np.random.uniform(-0.1, 0.1, temp_shape)
    # temp_col = np.zeros((shape[0], 1))
    # weight = np.hstack((temp_col, temp))
    # return weight
    return np.random.uniform(-0.1, 0.1, shape)

# returns the weight with folded bias
def zero_init(shape):
    """
    Initialize a numpy array of the specified shape with zero
    :param shape: list or tuple of shapes
    :return: initialized weights
    """
    return np.zeros((shape[0], shape[1]))

def linear(m, v):
    return np.dot(m, v)
    # return np.sum(np.multiply(m, v), axis=1)

def sigmoid(x):
    e = np.exp(x)
    return e / (1 + e)

def softmax(b):
    return (np.exp(b) / np.sum(np.exp(b), axis=0))

def cross_entropy(y_hat, y):
    res = np.log(y_hat[y])*(-1)
    return res
    # check this!

def d_cross_entropy(y, y_hat):
    y_onehot = np.zeros((y.size, 10))
    y_onehot[np.arange(y.size),y] = 1
    res = (y_hat - y_onehot.T)
    return res

    
class NN(object):
    def __init__(self, lr, n_epoch, weight_init_fn, input_size, hidden_size, output_size):
        """
        Initialization
        :param lr: learning rate
        :param n_epoch: number of training epochs
        :param weight_init_fn: weight initialization function
        :param input_size: number of units in the input layer
        :param hidden_size: number of units in the hidden layer
        :param output_size: number of units in the output layer
        """
        self.lr = lr
        self.n_epoch = n_epoch
        self.weight_init_fn = weight_init_fn
        self.n_input = input_size
        self.n_hidden = hidden_size
        self.n_output = output_size
        self.z = None
        self.loss = None
        self.mean_loss = None
        # initialize weights and biases for the models
        # HINT: pay attention to bias here

        if self.weight_init_fn == 1:
            temp1 = random_init([hidden_size, input_size-1]) 
            # append bias
            temp1_col = np.zeros((hidden_size, 1))
            self.w1 = np.hstack((temp1_col, temp1)) 

            temp2 = random_init([output_size, hidden_size]) 
            # append bias
            temp2_col = np.zeros((output_size, 1))
            self.w2 = np.hstack((temp2_col, temp2)) 

        else:
            self.w1 = zero_init([hidden_size, input_size]) 
            self.w2 = zero_init([output_size, hidden_size+1]) 

        # initialize parameters for adagrad
        self.epsilon = 1 * 10**(-5)
        self.grad_sum_w1 = zero_init([hidden_size, input_size]) # ?? what size
        self.grad_sum_w2 = zero_init([output_size, hidden_size+1])

        # feel free to add additional attributes



def print_weights(nn):
    """
    An example of how to use logging to print out debugging infos.

    Note that we use the debug logging level -- if we use a higher logging
    level, we will log things with the default logging configuration,
    causing potential slowdowns.

    Note that we log NumPy matrices on separate lines -- if we do not do this,
    the arrays will be turned into strings even when our logging is set to
    ignore debug, causing potential massive slowdowns.
    :param nn: your model
    :return:
    """
    logging.debug(f"shape of w1: {nn.w1.shape}")
    logging.debug(nn.w1)
    logging.debug(f"shape of w2: {nn.w2.shape}")
    logging.debug(nn.w2)

# taking all inputs together
def forward(X, y_tr, nn):
    """
    Neural network forward computation.
    Follow the pseudocode!
    :param X: input data
    :param nn: neural network class
    :return: output probability
    """
    a = np.matmul(nn.w1, X.T)
    z_star = sigmoid(a)
    # append top row of 1 to z
    nn.z = np.hstack([np.array([1]), z_star])
    b = np.matmul(nn.w2, nn.z)
    y_hat = softmax(b)
    loss = cross_entropy(y_hat, y_tr)
    return y_hat, loss


# taking input one by one
def backward(X, y, y_hat, nn):
    """
    Neural network backward computation.
    Follow the pseudocode!
    :param X: input data
    :param y: label
    :param y_hat: prediction
    :param nn: neural network class
    :return:
    d_w1: gradients for w1
    d_w2: gradients for w2
    """
    y_onehot = np.zeros((10))
    y_onehot[y] = 1
    gb = np.array([y_hat - y_onehot]) # column vector
    z_reshape = nn.z.reshape((nn.z.shape[0], 1))
    z_reshape = np.array(z_reshape.T)
    gbeta = gb.T.dot(z_reshape)
    beta_star = nn.w2[:,1:]
    beta_star_t =np.array(beta_star.T)
    gz = np.dot(beta_star_t, gb.T)
    z_star = z_reshape[:,1:]
    z_star = np.array(z_star.T)
    temp = 1 - z_star
    ga_temp = np.multiply(gz, z_star)
    ga = np.multiply(ga_temp, temp)
    x_reshape = X.reshape((1, X.shape[0]))
    galpha = np.dot(ga, x_reshape)
    return (galpha, gbeta)


def test(X, y, nn):
    """
    Compute the label and error rate.
    :param X: input data
    :param y: label
    :param nn: neural network class
    :return:
    labels: predicted labels
    error_rate: prediction error rate
    """
    error_count = 0
    labels = np.array([]) # ???????????? want an empty 1d array
    for i in range(X.shape[0]): 
        y_hat, loss = forward(X[i], y[i], nn)
        prediction = np.argmax(y_hat) 
        labels = np.append(labels, np.array([prediction])) # syntax?????
        if prediction != y[i]:
            error_count += 1
    error_rate = error_count / X.shape[0]
    return (labels, error_rate)
    

def train(X_tr_o, y_tr_o, X_te, y_te, nn):
    """
    Train the network using SGD for some epochs.
    :param X_tr: train data
    :param y_tr: train label
    :param nn: neural network class
    """
    train_mean_loss_arr = np.array([])
    test_mean_loss_arr = np.array([])
    for e in range(nn.n_epoch):
        X_tr, y_tr = shuffle(X_tr_o, y_tr_o, e)
        for i in range(y_tr.shape[0]):
            y_hat, loss = forward(X_tr[i], y_tr[i], nn)
            galpha, gbeta = backward(X_tr[i], y_tr[i], y_hat, nn)
            nn.grad_sum_w1 = nn.grad_sum_w1 + (galpha * galpha)
            nn.grad_sum_w2 = nn.grad_sum_w2 + (gbeta * gbeta)
            temp_alpha = nn.lr/((nn.grad_sum_w1 + nn.epsilon)**(0.5))
            temp_beta = nn.lr/((nn.grad_sum_w2 + nn.epsilon)**(0.5))
            nn.w1 = nn.w1 - (temp_alpha * galpha)
            nn.w2 = nn.w2 - (temp_beta * gbeta)
        
        # computing mean train loss and appending it to test_mean_out_arr

        train_mean_loss = 0
        for i in range(y_tr.shape[0]):
            x, loss = forward(X_tr[i], y_tr[i], nn)
            train_mean_loss += loss
        train_mean_loss = train_mean_loss / (y_tr.shape[0])
        train_mean_loss_arr = np.append(train_mean_loss_arr, np.array([train_mean_loss]))

        # computing mean test loss and appending it to test_mean_out_arr
        test_mean_loss = 0
        for i in range(y_te.shape[0]):
            x, loss = forward(X_te[i], y_te[i], nn)
            test_mean_loss += loss
        test_mean_loss = test_mean_loss / (y_te.shape[0])
        test_mean_loss_arr = np.append(test_mean_loss_arr, test_mean_loss)
    return (nn.w1, nn.w2, train_mean_loss_arr, test_mean_loss_arr)


if __name__ == "__main__":

    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(format='[%(asctime)s] {%(pathname)s:%(funcName)s:%(lineno)04d} %(levelname)s - %(message)s',
                            datefmt="%H:%M:%S",
                            level=logging.DEBUG)
    logging.debug('*** Debugging Mode ***')
    # Note: You can access arguments like learning rate with args.learning_rate

    # initialize training / test data and labels
    data = args2data(args)
    X_tr = data[0]
    y_tr = data[1]
    X_te = data[2]
    y_te = data[3]
    out_tr = data[4]
    out_te = data[5]
    out_metrics = data[6]
    n_epochs = data[7]
    n_hid = data[8]
    init_flag = data[9]
    lr = data[10]
    
    # Build model
    train_input_size = X_tr.shape[1]
    output_size = 10
    train_nn = NN(lr, n_epochs, init_flag, train_input_size, n_hid, output_size)

    # train model
    train_alpha, train_beta, train_ce_arr, test_ce_arr = train(X_tr, y_tr, X_te, y_te, train_nn)

    # test model and get predicted labels and errors
    train_labels, train_error_rate = test(X_tr, y_tr, train_nn)
    test_labels, test_error_rate = test(X_te, y_te, train_nn)

    # write predicted label and error into file
    with open(out_metrics, 'w') as f_out:
        for i in range(train_ce_arr.shape[0]):   
            f_out.write("epoch=%(epoch)x crossentropy(train): %(tr_ce)f\nepoch=%(epoch)x crossentropy(validation): %(te_ce)f\n"
            %{'epoch':(i+1), 'tr_ce':train_ce_arr[i], 'te_ce':test_ce_arr[i]})
        f_out.write('error(train): ' + str(train_error_rate) + '\n')
        f_out.write('error(validation): ' + str(test_error_rate) + '\n')

    with open(out_tr, 'w') as f_out:
        for label in train_labels:
            f_out.write(str(label) + '\n')
    with open(out_te, 'w') as f_out:
        for label in test_labels:
            f_out.write(str(label) + '\n')
