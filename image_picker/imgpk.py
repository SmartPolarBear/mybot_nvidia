import cv2
import PIL.Image
import numpy as np
import glob
import sys
import getopt
import os


def get_x(path):
    """Gets the x value from the image filename"""
    return (float(int(path[3:6])) - 50.0) / 50.0


def get_y(path):
    """Gets the y value from the image filename"""
    return (float(int(path[7:10])) - 50.0) / 50.0


def draw_xy(image, x, y):
    x = int(x*224/2+122)
    y = int(y*244/2+122)
    image = cv2.circle(image, (x, y), 8, (0, 255, 0), 3)
    image = cv2.circle(image, (122, 224), 8, (0, 0, 255), 3)
    image = cv2.line(image, (x, y), (122, 224), (255, 0, 0), 3)
    return PIL.Image.fromarray(image)

# imgpk.py -d -i <src> -o <dst>
# imgpk.py -s -i <src> -o <dst>


def display(src, dst):
    image_paths = glob.glob(os.path.join(src, '*.jpg'))
    for path in image_paths:
        print("Working on {}\n".format(path))
        _, name = os.path.split(path)

        cv2img = cv2.imread(path)
        cv2img = cv2.cvtColor(cv2img, cv2.COLOR_BGR2RGB)

        image = draw_xy(cv2img, get_x(name), get_y(name))
        new_path = os.path.join(dst, name)
        image.save(new_path)


def sync(src, dst):
    src_paths = glob.glob(os.path.join(src, '*.jpg'))
    dst_paths = glob.glob(os.path.join(dst, '*.jpg'))

    src_names = (os.path.split(path)[-1] for path in src_paths)
    dst_names = (os.path.split(path)[-1] for path in dst_paths)

    deleted = [i for i in src_names if i not in dst_names]

    for name in src_names:
        print("{}\n".format(name))
    print("\n")
    for name in dst_names:
        print("{}\n".format(name))
    print("\n")
    for name in deleted:
        print("{}\n".format(name))


    for name in deleted:
        pathname = os.path.join(src, name)
        print("Deleted {}\n".format(pathname))
        os.remove(pathname)



def main(argv):
    cmd: str = ""
    src: str = ""
    dst: str = ""
    try:
        opts, args = getopt.getopt(
            argv, "dsi:o:", ["display", "sync", "idir=", "odir="])
    except getopt.GetoptError:
        print("imgpk.py -d -i <src> -o <dst>\n")
        print("imgpk.py -s -i <src> -o <dst>\n")
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-i", "--idir"):
            src = arg
        elif opt in ("-o", "--odir"):
            dst = arg
        elif opt in ("-d", "--display"):
            cmd = "d"
        elif opt in ("-s", "--sync"):
            cmd = "s"

    if cmd == "d":
        display(src, dst)
    else:
        sync(src, dst)


if __name__ == "__main__":
    main(sys.argv[1:])
