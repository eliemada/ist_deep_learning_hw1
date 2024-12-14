#!/usr/bin/env python

# Deep Learning Homework 1

import argparse

import numpy as np
import matplotlib.pyplot as plt

import time
import utils


class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features))

    def update_weight(self, x_i, y_i, **kwargs):
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


class Perceptron(LinearModel):
    def update_weight(self, x_i, y_i, **kwargs):
        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """
        scores = np.dot(self.W, x_i)
        y_pred = np.argmax(scores)
        
        if y_pred != y_i:
            self.W[y_i] += x_i
            self.W[y_pred] -= x_i

        


class LogisticRegression(LinearModel):
    def update_weight(self, x_i, y_i, learning_rate=0.001, l2_penalty=0.0, **kwargs):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """
        
        #calculate probability for each class  
        scores = self.W.dot(x_i)
        probabilities  = (np.exp(scores) / np.sum(np.exp(scores))).reshape(-1, 1)

        #calculate gradient
        one_hot = np.zeros((np.size(self.W, 0),1))
        one_hot[y_i] = 1

        gradient = (probabilities - one_hot).dot(x_i.reshape(1, -1))
        factor = 1
        regularization_term_grad = factor*l2_penalty*self.W
        #question: half factor or not? 

        #update weights 
        self.W -= learning_rate*(gradient + regularization_term_grad)

        


class MLP(object):
    def __init__(self, n_classes, n_features, hidden_size):
        # Initialize an MLP with a single hidden layer.
        self.W1 = np.random.normal(loc = 0.1, scale = 0.1, size= (hidden_size,n_features))
        self.b1 = np.zeros(shape = (hidden_size,1))
        self.w_out = np.random.normal(loc = 0.1, scale = 0.1, size= (n_classes,hidden_size))
        self.b_out = np.zeros(shape = (n_classes,1))
        self.n_classes = n_classes
        self.hidden_size = hidden_size

    def predict(self, X):
        out = self.get_softmax_values(X)
        #print(f"out = {out}")
        prediction = out.argmax(axis = 0)
        print(f"prediction = {prediction}")
        return prediction

    def get_softmax_values(self,X): 
        z1 = np.dot(self.W1,X.T) + self.b1
        h1 = np.maximum(z1,0)
        out = np.dot(self.w_out, h1)+self.b_out
        #question: use relu also for output function??? --> use softmax!!
        #relu: out = np.maximum(0,out)
        #softmax: 
        out -= np.max(out,axis = 0)
        out = (1/ np.sum(np.exp(out), axis = 0))*np.exp(out)

        return out

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate()
        y_hat = self.predict(X)
        print(f"y_hat = {y_hat}")
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible

    def train_epoch(self, X, y, learning_rate=0.001, **kwargs):


        """
        Dont forget to return the loss of the epoch.
        """ 

        loss = 0
        for x_i,y_i in zip(X,y):
            
            x = x_i.reshape(1,np.size(x_i)) #reshape x to (n_feat,1)
            y_c = y_i 

            #forward-pass 
            z1 = np.dot(self.W1,x.T) + self.b1[0]
            #print(f"shape z1 = {z1.shape}")
            h1 = np.maximum(z1,0)

            zc = np.dot(self.w_out, h1)+self.b_out
            zc_stab = zc - np.max(zc)  # Numerical stability
            softmax_zc = np.exp(zc_stab) / np.sum(np.exp(zc_stab), axis=0, keepdims=True)


            #calculating gradients with backpropagation
            one_hot_c = np.zeros((self.n_classes,1))
            one_hot_c[y_c] = 1 
            g_der = np.zeros(shape=(self.hidden_size,1))  
            g_der[z1>0] = 1     #derrivative of relu function
            grad_zc = softmax_zc - one_hot_c

            der_w_out = grad_zc.dot(h1.T)
            der_b_out = grad_zc

            grad_h1 = (self.w_out.T.dot(grad_zc))

            der_W1 = ((self.w_out.T.dot(grad_zc))*g_der).dot(x)
            der_b1 = (self.w_out.T.dot(grad_zc))*g_der
 
            #update weights 
            self.W1 -= learning_rate*der_W1
            self.b1 -= learning_rate*der_b1
            self.w_out-= learning_rate*der_w_out
            self.b_out -= learning_rate*der_b_out

            #compute loss 

            #new forward pass
            z1 = np.dot(self.W1,x.T) + self.b1
            h1 = np.maximum(z1,0)
            zc = np.dot(self.w_out, h1)+self.b_out
            zc_stab = zc - np.max(zc)  # Numerical stability
            softmax_zc = np.exp(zc_stab) / np.sum(np.exp(zc_stab), axis=0, keepdims=True)
            softmax_zc_at_correct_class = softmax_zc[y_c]+ 1e-9
            #loss for this sample
            sample_loss = -np.log(softmax_zc_at_correct_class)
            loss += sample_loss.item()

        loss = loss / len(y)
        
        return loss


def plot(epochs, train_accs, val_accs, filename=None):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(epochs, train_accs, label='train')
    plt.plot(epochs, val_accs, label='validation')
    plt.legend()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()

def plot_loss(epochs, loss, filename=None):
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(epochs, loss, label='train')
    plt.legend()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()


def plot_w_norm(epochs, w_norms, filename=None):
    plt.xlabel('Epoch')
    plt.ylabel('W Norm')
    plt.plot(epochs, w_norms, label='train')
    plt.legend()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-hidden_size', type=int, default=100,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    parser.add_argument('-l2_penalty', type=float, default=0.0,)
    parser.add_argument('-data_path', type=str, default='intel_landscapes.v2.npz',)
    opt = parser.parse_args()

    utils.configure_seed(seed=42)
    add_bias = opt.model != "mlp"
    data = utils.load_dataset(data_path=opt.data_path, bias=add_bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]
    n_classes = np.unique(train_y).size
    n_feats = train_X.shape[1]

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size)
    epochs = np.arange(1, opt.epochs + 1)
    train_loss = []
    weight_norms = []
    valid_accs = []
    train_accs = []

    start = time.time()

    print('initial train acc: {:.4f} | initial val acc: {:.4f}'.format(
        model.evaluate(train_X, train_y), model.evaluate(dev_X, dev_y)
    ))
    
    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        if opt.model == 'mlp':
            loss = model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate
            )
        else:
            model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate,
                l2_penalty=opt.l2_penalty,
            )
        
        train_accs.append(model.evaluate(train_X, train_y))
        valid_accs.append(model.evaluate(dev_X, dev_y))
        if opt.model == 'mlp':
            print('loss: {:.4f} | train acc: {:.4f} | val acc: {:.4f}'.format(
                loss, train_accs[-1], valid_accs[-1],
            ))
            train_loss.append(loss)
        elif opt.model == "logistic_regression":
            weight_norm = np.linalg.norm(model.W)
            print('train acc: {:.4f} | val acc: {:.4f} | W norm: {:.4f}'.format(
                 train_accs[-1], valid_accs[-1], weight_norm,
            ))
            weight_norms.append(weight_norm)
        else:
            print('train acc: {:.4f} | val acc: {:.4f}'.format(
                 train_accs[-1], valid_accs[-1],
            ))
    elapsed_time = time.time() - start
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print('Training took {} minutes and {} seconds'.format(minutes, seconds))
    print('Final test acc: {:.4f}'.format(
        model.evaluate(test_X, test_y)
        ))

    # plot
    plot(epochs, train_accs, valid_accs, filename=f"Q1-{opt.model}-accs.pdf")
    if opt.model == 'mlp':
        plot_loss(epochs, train_loss, filename=f"Q1-{opt.model}-loss.pdf")
    elif opt.model == 'logistic_regression':
        plot_w_norm(epochs, weight_norms, filename=f"Q1-{opt.model}-w_norms.pdf")
    with open(f"Q1-{opt.model}-results.txt", "w") as f:
        f.write(f"Final test acc: {model.evaluate(test_X, test_y)}\n")
        f.write(f"Training time: {minutes} minutes and {seconds} seconds\n")


if __name__ == '__main__':
    main()
