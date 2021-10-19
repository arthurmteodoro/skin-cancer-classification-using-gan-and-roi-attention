from skimage import io
from hair_removal import remove_and_inpaint
import os
from multiprocessing import Process, Value

FOLDER = '/home/arthur/Programs/datasets/ISIC_2017_RESIZED/ISIC_2017_TRAINING'
counter = Value('i', 0)


def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


def process_images(images_names, total_imgs):
    for image in images_names:
        image_path = os.path.join(FOLDER, image)
        image = io.imread(image_path)

        hairless_image, _ = remove_and_inpaint(image)

        io.imsave(image_path, hairless_image)

        with counter.get_lock():
            counter.value += 1
            print('%d/%d' % (counter.value, total_imgs))


if __name__ == '__main__':
    processes = []

    files = os.listdir(FOLDER)
    total_imgs = len(files)
    files = [file for file in files if 'segmentation' not in file]
    files = chunkIt(files, 4)

    for i in range(4):
        p = Process(target=process_images, args=(files[i], total_imgs))
        processes.append(p)

    for i in range(4):
        processes[i].start()

    for i in range(4):
        processes[i].join()
