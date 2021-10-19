import os
import numpy as np
from PIL import Image
from multiprocessing import Process, Value

counter = Value('i', 0)
SOURCE = '/home/arthur/Programs/datasets/ISIC_2017_RESIZED/ISIC_2017_TEST'
DESTINATION = '/home/arthur/Programs/datasets/ISIC_2017_RESIZED/ISIC_2017_TEST'


def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


def center_crop(img, new_width=None, new_height=None):
    width = img.shape[1]
    height = img.shape[0]

    if new_width is None:
        new_width = min(width, height)

    if new_height is None:
        new_height = min(width, height)

    left = int(np.ceil((width - new_width) / 2))
    right = width - int(np.floor((width - new_width) / 2))

    top = int(np.ceil((height - new_height) / 2))
    bottom = height - int(np.floor((height - new_height) / 2))

    if len(img.shape) == 2:
        center_cropped_img = img[top:bottom, left:right]
    else:
        center_cropped_img = img[top:bottom, left:right, ...]

    return center_cropped_img


def process_images(images, total_imgs):
    for image in images:
        original_img = Image.open(os.path.join(SOURCE, image))
        new_size = min(original_img.size[0], original_img.size[1])

        cropped_image = center_crop(np.array(original_img), new_size, new_size)
        cropped_image = Image.fromarray(cropped_image)

        final_image = cropped_image.resize((256, 256))
        final_image.save(os.path.join(DESTINATION, image))

        with counter.get_lock():
            counter.value += 1
            print('%d/%d' % (counter.value, total_imgs))


if __name__ == '__main__':
    processes = []

    list_images = os.listdir(SOURCE)
    total_images = len(list_images)
    list_images = chunkIt(list_images, 4)

    for i in range(4):
        p = Process(target=process_images, args=(list_images[i], total_images))
        processes.append(p)

    for i in range(4):
        processes[i].start()

    for i in range(4):
        processes[i].join()
