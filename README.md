# CIFAR-10
CIFAR-10 has 60000 32x32 color images in 10 classes, with 6000 images per class.
There are 50000 train images and 10000 test images in this dataset.
In my observation, the model can be trained effectively and efficiently using the
following model architecture:
I’ve chosen Sequential model for this image classification. CIFAR-10 is comparatively
harder to train than compared to the MNIST data. Therefore, I had to add more layers to
help the model train better to give the best results. Here, I’ve used 2 Convolutional layers
of 32 filters, 2 Convolutional layers of 64 filters, 2 Convolutional layers of 128 filters,
along with Max pooling layers of pool size (2x2) and Dropout layers to avoid overfitting,
a Batch Normalization layer to increase the stability of the neural network(batch
normalization normalizes the output of a previous activation layer by subtracting the batch
mean and dividing by the batch standard deviation) and avoid overfitting, a Flatten layer
to convert the 2 dimensional matrix of features into a vector that can be passed into a
fully connected layer i.e., the Dense layer of 512 neurons followed by an output layer, a
Dense layer of 10 neurons to classify the 10 classes of images [airplane, automobile, bird,
cat, deer, dog, frog, horse, ship, truck].
