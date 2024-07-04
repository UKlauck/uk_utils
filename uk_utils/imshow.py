from matplotlib import pyplot as plt
import numpy as np

# ---------------------------------------------------
def imshow(img=None, reverse_channels=False, pyplot_rescale=False, debug=False):

    if img is None: return

    if debug: print('Image has shape ', img.shape, ' and data type ', img.dtype)

    if img.ndim == 4:  # batch of images
        if debug: print('WARNING: Received a batch of images. Only the first image will be plotted ')
        img = img[0]

    if img.ndim==3 and img.shape[2]==1:
        img = img.reshape(img.shape[0], img.shape[1])

    '''  Old code still using opencv
    if img.ndim==3:
        if img.shape[2] == 3 and reverse_channels: imgDisplay = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif img.shape[2] == 3 and not reverse_channels: imgDisplay = img
        elif img.shape[2] == 4 and reverse_channels: imgDisplay = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        elif img.shape[2] == 4 and not reverse_channels: imgDisplay = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        else: imgDisplay = None
    '''

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