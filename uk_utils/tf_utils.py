from tensorflow.keras.models import Model
from matplotlib import pyplot as plt
import numpy as np
from .imshow import imshow

def plot_layer_output(model, data, layer_no=0, img_no=0):
    # Construct a model using the original models input tensor and the requested
    # layers output tensor
    vis_model = Model(inputs=model.input, outputs=model.layers[layer_no].output)

    # Take one batch of data
    data = data.take(1).get_single_element()

    # Make a prediction
    out = vis_model.predict(data[0])

    fig = plt.figure(figsize=(4, 4))
    imshow(data[0][img_no].numpy())
    plt.show()

    fig = plt.figure(figsize=(12, 4))
    nc = out.shape[3]
    for c in range(nc):
        plt.subplot(1, nc, c + 1)
        imshow(out[img_no, :, :, c], pyplot_rescale=True)
    plt.show()


def print_layer_weights_(model, layer):
    weights = model.layers[layer].weights
    if len(weights) == 0:
        print(f'\n\nLayer {layer:2d} ({model.layers[layer].name}) has no weights\n')
        return

    for w in weights:
        if 'conv2d' in w.name and 'kernel' in w.name:
            print(f'\n\nLayer {layer:2d}  ({w.name})\n')
            w_weights = w.numpy()
            for n_c in range(w_weights.shape[2]):
                print(f'Channel {n_c}\n')
                for n_w in range(w_weights.shape[3]):
                    print(w_weights[:, :, n_c, n_w], '\n')

        elif 'conv2d' in w.name and 'bias' in w.name:
            print(f'Bias weights: {w.numpy()}\n')
        elif 'dense' in w.name and 'kernel' in w.name:
            print(f'\n\nLayer {layer:2d}  ({w.name})\n')
            w_weights = w.numpy()
            print(w_weights, '\n')
        elif 'dense' in w.name and 'bias' in w.name:
            print(f'Bias weights: {w.numpy()}\n')


def print_weights(model, layer_no=None):
    old_po = np.get_printoptions()
    np.set_printoptions(precision=3)

    if layer_no is None:
        for nl in range(len(model.layers)):
            print_layer_weights_(model, nl)
    else:
        print_layer_weights_(model, layer_no)

    np.set_printoptions(precision=old_po['precision'])

