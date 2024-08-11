import numpy as np
from skimage.metrics import peak_signal_noise_ratio

#---- Default parameters ----
a = 1
b = 1
key = 1
#---

def arnoldTransform(image: np.ndarray, key: int, ) -> np.ndarray:
    s = image.shape
    x, y = np.meshgrid(range(s[0]), range(s[0]), indexing="ij")
    xmap = (a * b * x + x + a * y) % s[0]
    ymap = (b * x + y) % s[0]
    img = image
    for r in range(key):
        img = img[xmap, ymap]
    return img

def arnoldInverseTransform(image: np.ndarray, key: int) -> np.ndarray:
    s = image.shape
    x, y = np.meshgrid(range(s[0]), range(s[0]), indexing="ij")
    xmap = (x - a * y) % s[0]
    ymap = (-b * x + a * b * y + y) % s[0]
    img = image
    for r in range(key):
        img = img[xmap, ymap]
    return img

#----- Attacks-------#

def gaussian_attack(img, intensity):
    ret = img.copy()
    ret += np.random.normal(0, intensity, ret.shape)
    return ret


def salt_pepper_attack(image, ratio=0.01, salt=True):
    """
    image - array
    salt: True is salt or 255, False is pepper or 0
    """
    ret = image.copy()
    if image.max() != 1:
        val = 255 if salt else 0
    else:
        val = 1 if salt else 0
    n = np.int(ratio * image.shape[0] * image.shape[1])
    for k in range(n):
        i = int(np.random.random() * image.shape[0])
        j = int(np.random.random() * image.shape[1])
        ret[i, j, :] = val
    return

def jpeg_attack(img, block_size = 8):
    marked = img.copy()
    channel_size = 3    #channel_size = marked.shape[2]
    dim1 = (img.shape[0] // block_size)
    dim2 = (img.shape[1] // block_size)
    for x in range(dim1):
        for y in range(dim2):
            for channel in range(channel_size):
                coef = dct2(img[x*block_size:x*block_size+block_size,
                           y*block_size:y*block_size+block_size, channel])
                for i in range(6,block_size):
                    for j in range(max(0, block_size-1-i), block_size):
                        coef[i,j] = 0
                marked[x*block_size:x*block_size+block_size, 
                       y*block_size:y*block_size+block_size, channel] = idct2(coef)
    return marked

def hist_attack(img):
    ret = (img.copy()[:, :, (2, 1, 0)] * 256).astype(np.uint8)
    img_yuv = cv2.cvtColor(ret, cv2.COLOR_BGR2YCrCb)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YCrCb2BGR)
    return img_output[:, :, (2, 1, 0)] / 256

def color_attack(img, channel):
    ret = img.copy()
    ret[:, :, channel] = cv2.equalizeHist(
        (256 * ret[:, :, channel]).astype(np.uint8)) / 256
    return ret

#---- Measures-----
def norm_data(data):
    mead_data = np.mean(data)
    std_data = np.std(data, ddof=1)
    return (data-mead_data)/std_data
def ncc(data0, data1):
    return (1.0/(data0.size-1))*np.sum(norm_data(data0)*norm_data(data1))


def mse(A, B):
    return np.mean(np.power(A - B, 2))
def psnr(A, B):
    return 20 * np.log10(255 / np.sqrt(mse(A, B)))
