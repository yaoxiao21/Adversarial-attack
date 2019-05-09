# Goal

The goal of this program is to implement the "fast gradient sign method" presented in [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572). This method is used to modify classical samples that a deep neural network trained classification will fail to classify properly.

![fgsm idea](http://karpathy.github.io/assets/break/breakconv.png "Fast Gradient Sign Method")

The idea of this method is to take a sample, ask the network to classify it, compute the gradient of the loss in function of the input pixels and update the picture by a small amount in the direction of the gradient. This direction is opposite to the one that would increase the score for the correct class.

# Cifar10 Dataset
Dataset of 50,000 32x32 color training images, labeled over 10 categories, and 10,000 test images.
- param train_start: index of first training set example
- param train_end: index of last training set example
- param test_start: index of first test set example
- param test_end: index of last test set example
- param nb_epochs: number of epochs to train model
- param batch_size: size of training batches
- param learning_rate: learning rate for training
- param clean_train: perform normal training on clean examples only before performing adversarial training.
- param testing: if true, complete an AccuracyReport for unit tests to verify that performance is adequate
- param backprop_through_attack: If True, backprop through adversarial example construction process during adversarial training.
- param label_smoothing: float, amount of label smoothing for cross entropy
- return: an AccuracyReport object
