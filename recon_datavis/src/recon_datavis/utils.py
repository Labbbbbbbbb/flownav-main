import io
import itertools
import numpy as np
import os
import PIL, PIL.Image
import sys
import termios
import tty


def imresize(image, shape, resize_method=PIL.Image.LANCZOS):
    assert (len(shape) == 3)
    assert (shape[-1] == 1 or shape[-1] == 3)
    assert (image.shape[0] / image.shape[1] == shape[0] / shape[1]) # maintain aspect ratio
    height, width, channels = shape

    if len(image.shape) > 2 and image.shape[2] == 1:
        image = image[:,:,0]

    im = PIL.Image.fromarray(image)
    im = im.resize((width, height), resize_method)
    im = np.array(im)

    if len(im.shape) == 2:
        im = np.expand_dims(im, 2)

    assert (im.shape == tuple(shape))

    return im

def mean_angle(angles):
    return np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))

class Getch:
    @staticmethod
    def getch(block=True):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

def im2bytes(arrs, format='jpg'):
    if len(arrs.shape) == 4:
        return np.array([im2bytes(arr_i, format=format) for arr_i in arrs])
    elif len(arrs.shape) == 3:
        im = PIL.Image.fromarray(arrs.astype(np.uint8))
        with io.BytesIO() as output:
            im.save(output, format="jpeg")
            return output.getvalue()
    else:
        raise ValueError

def bytes2im(arrs):
    if len(arrs.shape) == 1:
        return np.array([bytes2im(arr_i) for arr_i in arrs])
    elif len(arrs.shape) == 0:
        return np.array(PIL.Image.open(io.BytesIO(arrs)))
    else:
        raise ValueError


def get_files_ending_with(folder_or_folders, ext):
    if isinstance(folder_or_folders, str):
        folder = folder_or_folders
        assert os.path.exists(folder)

        fnames = []
        for fname in os.listdir(folder):
            if fname.endswith(ext):
                fnames.append(os.path.join(folder, fname))
        return sorted(fnames)
    else:
        assert hasattr(folder_or_folders, '__iter__')
        return sorted(list(itertools.chain(*[get_files_ending_with(folder, ext) for folder in folder_or_folders])))
