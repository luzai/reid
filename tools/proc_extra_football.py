from lz import *
import bs4

annfs = glob.glob(work_path + 'extra_footballplayer/*.xml')
annfs.sort()
inds = np.zeros(100, dtype=int)
for annf in annfs:
    with open(annf, 'r') as f:
        a = bs4.BeautifulSoup(f, 'lxml')
    for obj in a.find_all('object'):
        pid = obj.find('name').text
        if pid == 'person': continue
        pid = int(pid)
        bbox = obj.bndbox
        bbox = list(map(lambda x: int(getattr(x, 'text')),
                        [bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax]))
        imgf = annf.replace('.xml', '.png')
        im = cvb.read_img(imgf)
        im = im[
             bbox[1]:bbox[3],
             bbox[0]: bbox[2],
             :]
        # print(im.shape)
        if im.shape[0] == 0 or im.shape[1] == 0:
            print('fk')
        cvb.write_img(im, work_path + f'/extra_train/{pid}/{inds[pid]}.png')
        inds[pid] += 1
