from os.path import isfile


def is_image_file(fn):
    '''
    Tries to guess from the file name, wether a file contains an image. All image file types
    supported by OpenCV are considered.

    Parameters
    ----------
    fn

    Returns
    -------

    '''
    img_formats = ['jpg', 'jpeg', 'tif', 'tiff',
                   'bmp', 'dib', 'jp2', 'png',
                   'webp', 'pbm', 'pgm', 'ppm', 'pxm', 'pnm',
                   'pfm', 'sr', 'ras', 'exr', 'hdr', 'pic']

    (left, sep, fn_ext) = fn.rpartition('.')
    if sep != '.':
        return False
    if fn_ext.lower() not in img_formats:
        return False
    else:
        return True