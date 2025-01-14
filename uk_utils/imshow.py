from matplotlib import pyplot as plt
import numpy as np

# ---------------------------------------------------
def imshow(img=None, reverse_channels=False, pyplot_rescale=False, debug=False):
    """
    Display an image using matplotlib.pyplot.

    This function handles various image formats and dimensions, including
    grayscale, RGB, and RGBA images. It can also handle batches of images,
    displaying only the first image in the batch.

    Parameters:
    img (numpy.ndarray, optional): The image to display. Can be 2D (grayscale),
                                   3D (RGB/RGBA), or 4D (batch of images).
                                   Defaults to None.
    reverse_channels (bool, optional): If True, reverse the order of color channels
                                       for RGB and RGBA images. Defaults to False.
    pyplot_rescale (bool, optional): If True, allow pyplot to rescale the image
                                     intensity values. Defaults to False.
    debug (bool, optional): If True, print debug information. Defaults to False.

    Returns:
    None

    Note:
    - For 4D inputs (batch of images), only the first image is displayed.
    - For 3D inputs with a single channel, the image is reshaped to 2D.
    - For 2D inputs (grayscale), the image is displayed without color mapping.
    - The function sets up the plot but does not call plt.show(). The caller
      is responsible for showing or saving the plot.
    """
    if img is None: return

    if debug: print('Image has shape ', img.shape, ' and data type ', img.dtype)

    if img.ndim == 4:  # batch of images
        if debug: print('WARNING: Received a batch of images. Only the first image will be plotted ')
        img = img[0]

    if img.ndim==3 and img.shape[2]==1:
        img = img.reshape(img.shape[0], img.shape[1])

    if img.ndim == 3:
        if img.shape[2] == 3 and reverse_channels:
            imgDisplay = img[:, :, ::-1]
        elif img.shape[2] == 3 and not reverse_channels:
            imgDisplay = img
        elif img.shape[2] == 4 and reverse_channels:
            imgDisplay = img[:, :, 0:3][:, :, ::-1]
        elif img.shape[2] == 4 and not reverse_channels:
            imgDisplay = img[:, :, :-1]
        else:
            imgDisplay = None

        if imgDisplay is not None:
                plt.imshow(imgDisplay)
                _ = plt.xticks([]), plt.yticks([])

    elif img.ndim == 2:
        # Prevent autoscaling in matplotlib.pyplot
        vmin = vmax = None
        if img.dtype==np.uint8:
            vmin = 0
            vmax = 255
            if debug: print('INFO: Image will not be rescaled by pyplot')

        if vmin is None or vmax is None or pyplot_rescale is True:
            plt.imshow(img, cmap = 'gray', interpolation = 'nearest')
        else:
            plt.imshow(img, cmap = 'gray', interpolation = 'nearest', vmin=vmin, vmax=vmax)

        _ = plt.xticks([]), plt.yticks([])


# ---------------------------------------------------
def imgshow(img=None):
    imshow(img)