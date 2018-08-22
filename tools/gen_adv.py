from lz import *
from reid.models import resnet50
from torch.autograd import Variable
from reid.datasets import cuhk03

ds = cuhk03.CUHK03()
# for img, pid, cid in ds.query:
#     print(img, pid, cid)
for img, pid, cid in ds.trainval:
    print(img, pid, cid)
    break
model = resnet50(num_features=128, num_classes=767, )
state_dict = torch.load(work_path + 'reid/work/11.xent.cu03lbl.ft/model_best.pth')['state_dict']
basedir = work_path + 'data/cuhk03/'
model.load_state_dict(state_dict)
model = model.eval().cuda()
criterion = nn.CrossEntropyLoss().cuda()


def l2_normalize(x):
    # can only handle (128,2048) or (128,2048,8,4)
    shape = x.size()
    x1 = x.view(shape[0], -1)
    x2 = x1 / x1.norm(p=2, dim=1, keepdim=True)
    return x2.view(shape)


# imgl = glob.glob(basedir + 'test/*')
for imgp, pid, cid in ds.trainval:
    print(imgp, pid, cid)
# for imgp in imgl:
    orig = cv2.imread(imgp)
    orig = cv2.resize(orig, (128, 256))  # orig in BGR
    img = orig.copy()[..., ::-1].astype(np.float32)
    # perturbation = np.empty_like(orig)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    std_th = torch.Tensor(std).view(1, -1, 1, 1).cuda()
    mean_th = torch.Tensor(mean).view(1, -1, 1, 1).cuda()
    img /= 255.0
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)
    with torch.no_grad():
        inp = torch.from_numpy(img).cuda().float().unsqueeze(0)
        # print(inp)
        _, logits, _ = model(inp)
        pred = np.argmax(logits.data.cpu().numpy())
        print('Prediction before attack: %s' % (pred))
    # break
    for eps in np.arange(0.09, 100, .01):
        eps = float(eps)
        inp = Variable(torch.from_numpy(img).cuda().float().unsqueeze(0), requires_grad=True)
        # compute loss
        _, logits, _ = model(inp)
        loss = criterion(logits, Variable(torch.Tensor([float(pred)]).cuda().long()))

        # compute gradients
        loss.backward()
        inpadv = inp.detach() + ((eps / 255.0) * torch.sign(inp.grad.data.detach()))
        # inpadv = inp.detach() + ((eps / 255.0) * l2_normalize(inp.grad.data.detach()))
        # inpadv = inp.detach() + ((eps / 255.0) * (inp.grad.data.detach()))
        # print(torch.sign(inp.grad.data.detach()).max(), torch.sign(inp.grad.data.detach()).min())
        # inpadv = ((inpadv * std_th) + mean_th) * 255.
        # inpadv = inpadv.long().clamp(0, 255)
        # inpadv = inpadv.float()
        # inpadv = (inpadv / 255. - mean_th) / std_th
        _, logits, _ = model(inpadv)
        pred_adv = np.argmax(logits.data.cpu().numpy())
        if pred_adv != pred:
            adv = inpadv.data.cpu().numpy().copy()[0]
            # perturbation = (adv - img).transpose(1, 2, 0)
            # print('--', perturbation.mean(), perturbation.max(), perturbation.min())
            # perturbation = (perturbation - perturbation.min()) * 255.  # / perturbation.max()
            # perturbation = np.clip(perturbation, 0, 255).astype(np.uint8)
            # print('--', perturbation.mean(), perturbation.max(), perturbation.min())
            adv = adv.transpose(1, 2, 0)
            adv = (adv * std) + mean
            adv = adv * 255.0
            adv = adv[..., ::-1]  # RGB to BGR
            adv = np.clip(adv, 0, 255).astype(np.uint8)
            if (adv - orig).max() > 0. or (adv - orig).min() < 0.:
                break
        # raise ValueError('pls incr range')

    print("After attack: eps [%f] \t%s"
          % (eps, pred_adv), 'from', pred)
    perturbation = (adv.astype(float) - orig.astype(float))
    print('--', perturbation.mean(), perturbation.max(), perturbation.min())
    perturbation = np.abs(perturbation)
    perturbation = (perturbation - perturbation.min()) / perturbation.max() * 128.
    perturbation = perturbation.astype(np.uint8)
    # cvb.show_img(cvb.bgr2gray(perturbation, keepdim=False), 'pergray', wait_time=1)
    # cvb.show_img(orig, 'orig', wait_time=1)
    # cvb.show_img(adv, 'after', wait_time=1)
    # cvb.show_img(perturbation, 'pertub', wait_time=30*1000)
    # cv2.destroyAllWindows()
    imgpdst = basedir + 'fgsm/' + osp.basename(imgp)
    cvb.write_img(adv, imgpdst)
