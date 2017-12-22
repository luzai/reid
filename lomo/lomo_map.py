import cv2
from lz import *

global DEBUG
DEBUG = False


def channel_hist(img_channel, bin_size):
    hist = cv2.calcHist([img_channel], [0], None, [bin_size], [0.0, 255.0])
    w, h = img_channel.shape
    return hist / (w * h)


def map_lomo_com(im, block_step, block_size, bin_size):
    h, w, c = im.shape
    map_height = h / block_step - 1
    map_width = w / block_step - 1
    # print([map_height,map_width])
    new_array = np.zeros((map_height, map_width, bin_size * c), dtype='float32')
    for i in range(map_height):
        for j in range(map_width):
            buff = im[i * block_step:i * block_step + block_size, j * block_step:j * block_step + block_size, :]
            for k in range(c):
                hist_buff = channel_hist(buff[:, :, k], bin_size)
                new_array[i, j, k * bin_size:(k + 1) * bin_size] = hist_buff[:, 0]
    return new_array.transpose(2, 0, 1) * 255


extract_feature = lambda x, dbg=None: FeaExtractor().extract_feature(x, dbg)


class FeaExtractor(object):
    def __init__(self, RGB_para=[True, 8], HSV_para=[True, 8], SILTP_para=[True, 16],
                 block_size=8, block_step=4, pad_size=2, tau=0.3, R=5, numPoints=4):
        self.block_size = block_size
        self.block_step = block_step
        self.pad_size = pad_size
        self.tau = tau
        self.R = R
        self.numPoints = numPoints
        self.RGB_para = RGB_para
        self.HSV_para = HSV_para
        self.SILTP_para = SILTP_para

    def channel_hist(self, img_channel, bin_size, range_para):
        w, h = img_channel.shape
        hist = cv2.calcHist([img_channel], [0], None, [bin_size], [0.0, range_para])
        return hist / (w * h)

    def map_histgram_com(self, im, bin_size, range_para=255.0):
        if len(im.shape) == 2:
            h, w = im.shape
            c = 1
        else:
            h, w, c = im.shape
        # print([h,w,c])
        map_height = h // self.block_step - 1
        map_width = w // self.block_step - 1

        # print([map_height,map_width])
        im = np.array(im, dtype='float32')
        new_array = np.zeros((map_height, map_width, bin_size * c), dtype='float32')
        for i in range(map_height):
            for j in range(map_width):
                if c > 1:
                    buff = im[i * self.block_step:i * self.block_step + self.block_size,
                           j * self.block_step:j * self.block_step + self.block_size, :]
                    for k in range(c):
                        hist_buff = self.channel_hist(buff[:, :, k], bin_size, range_para)
                        new_array[i, j, k * bin_size:(k + 1) * bin_size] = hist_buff[:, 0]
                else:
                    buff = im[i * self.block_step:i * self.block_step + self.block_size,
                           j * self.block_step:j * self.block_step + self.block_size]
                    # print(buff.dtype)
                    hist_buff = self.channel_hist(buff[:, :], bin_size, range_para)
                    new_array[i, j] = hist_buff[:, 0]
        return new_array.transpose(2, 0, 1) * 255

    def SILTP(self, im, encoder=0):
        h, w = im.shape
        R = self.R
        tau = self.tau
        if h < 2 * R + 1 or w < 2 * R + 1:
            print('error: too small image or too large R!')
        # put the image in a larger container
        im_norm = 1.0 * im / 255
        # if DEBUG:
        #     print(im_norm)
        I0 = np.zeros((h + 2 * R, w + 2 * R), dtype=np.float32)
        I0[R:h + R, R:w + R] = im_norm
        # replicate border image pixels to the outer area
        for i in range(R):
            I0[i, :] = I0[R, :]
            I0[h + R + i, :] = I0[h + R - 1, :]
            I0[:, w + R + i] = I0[:, w + R - 1]
            I0[:, i] = I0[:, R]
        if DEBUG:
            cv2.imshow('ori', im)
            cv2.imshow('I0', I0)
            cv2.waitKey(0)
        # copy image in specified directions
        I1 = I0[R:h + R, 2 * R:]
        I3 = I0[:h, R:w + R]
        I5 = I0[R:h + R, :w]
        I7 = I0[2 * R:, R:w + R]
        if DEBUG:
            cv2.imshow('I1', I1)
            cv2.imshow('I3', I3)
            cv2.imshow('I5', I5)
            cv2.imshow('I7', I7)
            print([I1.shape, I3.shape, I5.shape, I7.shape])
            cv2.waitKey(0)
        # compute the upper and lower range
        L = (1 - tau) * im_norm
        U = (1 + tau) * im_norm
        if DEBUG:
            print([L.shape, U.shape])
            print(L[0, :])
        # compute the scale invariant local ternary patterns
        if encoder == 0:
            J = (I1 < L) + (I1 > U) * 2 + ((I3 < L) + (I3 > U) * 2) * 3 \
                + ((I5 < L) + (I5 > U) * 2) * 9 + ((I7 < L) + (I7 > U) * 2) * 27;
        else:
            J = (I1 > U) + (I1 < L) * 2 + (I3 > U) * 2 ^ 2 + (I3 < L) * 2 ^ 3 \
                + (I5 > U) * 2 ^ 4 + (I5 < L) * 2 ^ 5 + (I7 > U) * 2 ^ 6 + (I7 < L) * 2 ^ 7
        return J

    # ***_para = [flage, bin_size]
    def extract_feature(self, imgpath, dbg=None):
        if isinstance(imgpath, str):
            im = cv2.imread(imgpath, cv2.IMREAD_COLOR)
            # im = cv2.imread(dbg, cv2.IMREAD_COLOR)
            im = cv2.resize(im, (128, 256), interpolation=cv2.INTER_LINEAR)
        else:
            im = to_numpy(imgpath)
            im = norm_np(im).astype(np.uint8)
            im = np.transpose(im, (2, 1, 0))
            # im.shape
        hist_RGB = np.zeros((1, 1))
        hist_HSV = np.zeros((1, 1))
        hist_SILTP = np.zeros((1, 1))
        # RGB hist
        if len(self.RGB_para) > 1:
            if self.RGB_para[0]:
                RGB_im = np.pad(im, ((self.pad_size, self.pad_size), (self.pad_size, self.pad_size), (0, 0)),
                                mode='constant',
                                constant_values=0)
                hist_RGB = self.map_histgram_com(RGB_im, self.RGB_para[1])
        # HSV hist
        if len(self.HSV_para) > 1:
            if self.HSV_para[0]:
                HSV_im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
                HSV_im = np.pad(HSV_im, ((self.pad_size, self.pad_size), (self.pad_size, self.pad_size), (0, 0)),
                                mode='constant',
                                constant_values=0)
                hist_HSV = self.map_histgram_com(HSV_im, self.HSV_para[1])
        # SILTP hist
        if len(self.SILTP_para) > 1:
            if self.SILTP_para[0]:
                grayim = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                siltp_im = self.SILTP(grayim, encoder=0)
                siltp_im = np.pad(siltp_im, ((self.pad_size, self.pad_size), (self.pad_size, self.pad_size))
                                  , mode='constant', constant_values=0)
                hist_SILTP = self.map_histgram_com(siltp_im, self.SILTP_para[1], 81.0)

        return np.concatenate((hist_RGB, hist_HSV, hist_SILTP))


if __name__ == '__main__':
    img_path = '/data1/xinglu/work/data/cuhk03/label/images'
    npy_path = '/data1/xinglu/work/data/cuhk03/label/npy'

    # img_path = '/data1/xinglu/work/data/market1501/images'
    # npy_path = '/data1/xinglu/work/data/market1501/npy'


    # mkdir_p(npy_path,delete=True)

    def func(img_name):
        npy_name = img_name.replace('images', 'npy').replace('.jpg', '.npy')
        # if osp.exists(npy_name): return
        npy = FeaExtractor().extract_feature(img_name)
        np.save(npy_name, npy)
        print(npy_name, npy.shape)


    pool = mp.Pool(processes=16)
    img_name_iter = glob.iglob(img_path + '/*.jpg')
    pool.map(func, img_name_iter)
    timer = cvb.Timer()
    # for img_name in img_name_iter:
    #     func(img_name)
    # break
    print(timer.since_start())
