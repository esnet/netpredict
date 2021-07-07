import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle

def generate_image_adversary(model, image, label, eps=0.01):
    # cast the image
    image = tf.cast(image, tf.float32)

    # record our gradients
    with tf.GradientTape() as tape:
        # explicitly indicate that our image should be tacked for
        # gradient updates
        tape.watch(image)

        # use our model to make predictions on the input image and
        # then compute the loss
        pred = model(image)
        loss = tf.keras.losses.MeanSquaredError(np.float32(label), pred)

    # calculate the gradients of loss with respect to the image, then
    # compute the sign of the gradient
    gradient = tape.gradient(loss, image)
    signedGrad = tf.sign(gradient)

    # construct the image adversary
    adversary = (image + (signedGrad * eps)).numpy()

    # return the image adversary to the calling function
    return adversary


def generate_adversarial_batch(model, total, images, labels, dims, eps=0.01):
    # unpack the image dimensions into convenience variables
    (h, w, c) = dims

    # we're constructing a data generator here so we need to loop
    # indefinitely
    while True:
        # initialize our perturbed images and labels
        perturbImages = []
        perturbLabels = []

        # randomly sample indexes (without replacement) from the
        # input data
        idxs = np.random.choice(range(0, len(images)), size=total, replace=False)

        # loop over the indexes
        for i in idxs:
            # grab the current image and label
            image = images[i]
            label = labels[i]

            # generate an adversarial image
            adversary = generate_image_adversary(model, image.reshape(1, h, w, c), label, eps=eps)

            # update our perturbed images and labels lists
            perturbImages.append(adversary.reshape(h, w, c))
            perturbLabels.append(label)

        # yield the perturbed images and labels
        yield (np.array(perturbImages), np.array(perturbLabels))


def generate_mixed_adverserial_batch(model, total, images, labels, dims, eps=0.01, split=0.5):
    # unpack the image dimensions into convenience variables
    (h, w, c) = dims

    # compute the total number of training images to keep along with
    # the number of adversarial images to generate
    totalNormal = int(total * split)
    totalAdv = int(total * (1 - split))

    # we're constructing a data generator so we need to loop
    # indefinitely
    while True:
        # randomly sample indexes (without replacement) from the
        # input data and then use those indexes to sample our normal
        # images and labels
        idxs = np.random.choice(range(0, len(images)), size=totalNormal, replace=False)
        mixedImages = images[idxs]
        mixedLabels = labels[0].iloc[idxs]

        # again, randomly sample indexes from the input data, this
        # time to construct our adversarial images
        idxs = np.random.choice(range(0, len(images)), size=totalAdv, replace=False)

        # loop over the indexes
        for i in idxs:
            # grab the current image and label, then use that data to
            # generate the adversarial example
            image = images[i]
            label = labels[0].iloc[i]
            adversary = generate_image_adversary(model, image.reshape(1, w, c), np.expand_dims(np.expand_dims(label, 0), 1), eps=eps)

            # update the mixed images and labels lists
            mixedImages = np.vstack([mixedImages, adversary])
            mixedLabels = np.concatenate((mixedLabels, np.expand_dims(label, 0)), axis=0)

        # shuffle the images and labels together
        (mixedImages, mixedLabels) = shuffle(mixedImages, mixedLabels)

        # yield the mixed images and labels to the calling function
        yield (mixedImages, mixedLabels)


# adam = optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, amsgrad=False)  # 1e-6
# model5.compile(loss=tf.keras.losses.MeanSquaredError, optimizer=adam)
# # model5.compile(loss = mean_log_LaPlace_like, optimizer=adam)
#
# dataGen = generate_mixed_adverserial_batch(model5, batch_size,
#                                            np.expand_dims(x_train_scaled.transpose(), axis=2), y_train,
#                                            (1, 100, 1), eps=0.01, split=0.5)
#
# model5.fit(dataGen, steps_per_epoch=len(y_train) // batch_size, epochs=epochs, verbose=1,
#            callbacks=[reduce_lr5])