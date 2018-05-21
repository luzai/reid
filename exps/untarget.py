import sys

sys.path.insert(0, '/data1/xinglu/prj/open-reid')

from lz import *
import lz
from torch.optim import Optimizer
from torch.backends import cudnn
from torch.utils.data import DataLoader
import reid
from reid import datasets
from reid import models
from reid.models import *
from reid.dist_metric import DistanceMetric
from reid.loss import *
from reid.trainers import *
from reid.evaluators import *
from reid.mining import *
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.data.sampler import *
from reid.utils.logging import Logger
from reid.utils.serialization import *
from reid.utils.dop import DopInfo

from tensorboardX import SummaryWriter


def run(_):
    cfgs = lz.load_cfg('./cfgs/single_ohnm.py')
    procs = []
    for args in cfgs.cfgs:
        if args.loss != 'tcx':
            print(f'skip {args.loss} {args.logs_dir}')
            continue
        # args.log_at = np.concatenate([
        #     args.log_at,
        #     range(args.epochs - 8, args.epochs, 1)
        # ])
        args.logs_dir = 'work/' + args.logs_dir
        if not args.gpu_fix:
            args.gpu = lz.get_dev(n=len(args.gpu),
                                  ok=args.gpu_range,
                                  mem=[0.12, 0.07], sleep=32.3)
        lz.logging.info(f'use gpu {args.gpu}')
        # args.batch_size = 16
        # args.gpu = (3, )
        # args.epochs = 1
        # args.logs_dir+='.bak'

        if isinstance(args.gpu, int):
            args.gpu = [args.gpu]
        if not args.evaluate:
            assert args.logs_dir != args.resume
            lz.mkdir_p(args.logs_dir, delete=True)
            lz.pickle_dump(args, args.logs_dir + '/conf.pkl')

        # main(args)
        proc = mp.Process(target=main, args=(args,))
        proc.start()
        lz.logging.info('next')
        time.sleep(random.randint(39, 90))
        procs.append(proc)

    for proc in procs:
        proc.join()


def get_data(args):
    (name, split_id,
     data_dir, height, width,
     batch_size, num_instances,
     workers, combine_trainval) = (
        args.dataset, args.split,
        args.data_dir, args.height, args.width,
        args.batch_size, args.num_instances,
        args.workers, args.combine_trainval,)
    pin_memory = args.pin_mem
    name_val = args.dataset_val
    npy = args.has_npy
    rand_ratio = args.random_ratio

    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root, split_id=split_id, mode=args.dataset_mode)

    root = osp.join(data_dir, name_val)
    dataset_val = datasets.create(name_val, root, split_id=split_id, mode=args.dataset_mode)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_set = dataset.trainval if combine_trainval else dataset.train
    num_classes = (dataset.num_trainval_ids if combine_trainval
                   else dataset.num_train_ids)

    train_transformer = T.Compose([
        T.RandomCropFlip(height, width, area=args.area),
        T.ToTensor(),
        normalizer,
    ])

    test_transformer = T.Compose([
        T.RectScale(height, width),
        T.ToTensor(),
        normalizer,
    ])
    dop_info = DopInfo(num_classes)
    print('dop info and its id are', dop_info)
    trainval_t = np.asarray(dataset.trainval, dtype=[('fname', object),
                                                     ('pid', int),
                                                     ('cid', int)])
    trainval_t = trainval_t.view(np.recarray)
    trainval_t = trainval_t[:np.where(trainval_t.pid == 10)[0].min()]

    trainval_test_loader = DataLoader(Preprocessor(
        # dataset.val,
        # dataset.query,
        # random.choices(trainval_t, k=1367 * 3),
        trainval_t.tolist(),
        root=dataset.images_dir,
        transform=test_transformer,
        has_npy=npy),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=pin_memory)
    train_loader = DataLoader(
        Preprocessor(train_set, root=dataset.images_dir,
                     transform=train_transformer,
                     has_npy=npy),
        batch_size=batch_size, num_workers=workers,
        sampler=RandomIdentityWeightedSampler(
            train_set, num_instances,
            batch_size=batch_size,
            rand_ratio=rand_ratio,
            dop_info=dop_info,
        ),
        # shuffle=True,
        pin_memory=pin_memory, drop_last=True)

    val_loader = DataLoader(
        Preprocessor(dataset_val.val, root=dataset_val.images_dir,
                     transform=test_transformer,
                     has_npy=npy),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=pin_memory)
    query_ga = np.concatenate([
        np.asarray(dataset_val.query).reshape(-1, 3),
        np.asarray(list(set(dataset_val.gallery) - set(dataset_val.query))).reshape(-1, 3)
    ])

    query_ga = np.rec.fromarrays((query_ga[:, 0], query_ga[:, 1].astype(int), query_ga[:, 2].astype(int)),
                                 names=['fnames', 'pids', 'cids'])
    if args.vis:
        pids_chs = np.unique(query_ga.pids)[:10]
        query_ga = query_ga[np.where(np.isin(query_ga.pids, pids_chs))[0]]

    query_ga = query_ga.tolist()
    test_loader = DataLoader(
        Preprocessor(query_ga,
                     root=dataset_val.images_dir,
                     transform=test_transformer,
                     has_npy=npy),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=False)
    dataset.val = dataset_val.val
    dataset.query = dataset_val.query
    dataset.gallery = dataset_val.gallery
    dataset.images_dir = dataset_val.images_dir
    if args.vis:
        query = np.asarray(dataset.query, dtype=[('fname', object),
                                                 ('pids', int),
                                                 ('cid', int)])
        query = query.view(np.recarray)
        query = query[np.where(np.isin(query.pids, pids_chs))[0]]

        dataset.query = query.tolist()

        gallery = np.asarray(dataset.gallery, dtype=[('fname', object),
                                                     ('pids', int),
                                                     ('cid', int)])
        gallery = gallery.view(np.recarray)
        gallery = gallery[np.where(np.isin(gallery.pids, pids_chs))[0]]

        dataset.gallery = gallery.tolist()

    # dataset.num_val_ids
    return dataset, num_classes, train_loader, val_loader, test_loader, dop_info, trainval_test_loader


def recreate_image(im_as_var):
    """
        Recreates images from a torch variable, sort of reverse preprocessing

    Args:
        im_as_var (torch variable): Image to recreate

    returns:
        recreated_im (numpy arr): Recreated image in array
    """
    reverse_mean = [-0.485, -0.456, -0.406]
    reverse_std = [1 / 0.229, 1 / 0.224, 1 / 0.225]
    recreated_im = copy.copy(im_as_var.data.numpy()[0])
    for c in range(3):
        recreated_im[c] /= reverse_std[c]
        recreated_im[c] -= reverse_mean[c]
    recreated_im[recreated_im > 1] = 1
    recreated_im[recreated_im < 0] = 0
    recreated_im = np.round(recreated_im * 255)

    recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)
    # Convert RBG to GBR
    # recreated_im = recreated_im[..., ::-1]
    return recreated_im


def main(args):
    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    sys.stderr = Logger(osp.join(args.logs_dir, 'err.txt'))
    lz.init_dev(args.gpu)
    print('config is {}'.format(vars(args)))
    args.seed = 16
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True

    # Create data loaders
    assert args.num_instances > 1, "num_instances should be greater than 1"
    assert args.batch_size % args.num_instances == 0, \
        'num_instances should divide batch_size'

    (dataset, num_classes,
     train_loader, val_loader, test_loader,
     dop_info, trainval_test_loader) = get_data(args)
    # Create model
    model = models.create(args.arch,
                          dropout=args.dropout,
                          pretrained=args.pretrained,
                          block_name=args.block_name,
                          block_name2=args.block_name2,
                          num_features=args.num_classes,
                          num_classes=num_classes,
                          num_deform=args.num_deform,
                          fusion=args.fusion,
                          )

    print(model)
    param_mb = sum(p.numel() for p in model.parameters()) / 1000000.0
    logging.info('    Total params: %.2fM' % (param_mb))

    # Load from checkpoint
    start_epoch = best_top1 = 0
    if args.resume:
        while not osp.exists(args.resume):
            lz.logging.warning(' no chkpoint {} '.format(args.resume))
            time.sleep(20)
        checkpoint = load_checkpoint(args.resume,
                                     map_location='cpu'
                                     )
        # model.load_state_dict(checkpoint['state_dict'])
        db_name = args.logs_dir + '/' + args.logs_dir.split('/')[-1] + '.h5'
        load_state_dict(model, checkpoint['state_dict'])
        with lz.Database(db_name) as db:
            if 'cent' in checkpoint:
                db['cent'] = to_numpy(checkpoint['cent'])
            db['xent'] = to_numpy(checkpoint['state_dict']['embed2.weight'])
        if args.restart:
            start_epoch_ = checkpoint['epoch']
            best_top1_ = checkpoint['best_top1']
            print("=> Start epoch {}  best top1 {:.1%}"
                  .format(start_epoch_, best_top1_))
        else:
            start_epoch = checkpoint['epoch']
            best_top1 = checkpoint['best_top1']
            print("=> Start epoch {}  best top1 {:.1%}"
                  .format(start_epoch, best_top1))
    # if args.gpu is None:
    # model = nn.DataParallel(model)
    # elif len(args.gpu) == 1:
    #     model = nn.DataParallel(model).cuda()
    # else:
    #     model = nn.DataParallel(model, device_ids=range(len(args.gpu))).cuda()

    test_transformer = T.Compose([
        T.RectScale(256, 128),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    from PIL import Image

    for fn in glob.glob(root_path + '/exps/fig/*.jpg'):
        # fn = glob.glob(root_path + '/exps/fig/*.jpg')[0]
        name = fn.split('/')[-1].split('.')[0]

        def preprocess_image(img):
            img = test_transformer(img)
            img = torch.tensor(img.view(1, img.shape[0], img.shape[1], img.shape[2]), requires_grad=True)
            return img

        img = Image.open(fn).convert('RGB')
        original_image = img.copy().resize((128, 256))
        img = preprocess_image(img)
        model.eval()

        # gbp = GuidedBackprop(model)
        # grads = gbp.generate_gradients(img, 0)
        #
        # pos_sal, neg_sal = get_positive_negative_saliency(grads)
        # save_gradient_images(pos_sal, name + '_pos_sal')
        # save_gradient_images(neg_sal, name + '_neg_sal')

        im_label = 0

        ce = nn.CrossEntropyLoss()
        for i in range(10):
            img.grad = None
            out = model(img)
            loss = ce(out, torch.ones(1, dtype=torch.int64) * 0)
            loss.backward()
            adv_noise = 0.01 * torch.sign(img.grad.data)
            img.data += adv_noise

            recreated_image = recreate_image(img)
            # Process confirmation image
            recreated_image = np.ascontiguousarray(recreated_image)
            prep_confirmation_image = preprocess_image(recreated_image)
            # Forward pass
            confirmation_out = model(prep_confirmation_image)
            # Get prediction
            _, confirmation_prediction = confirmation_out.data.max(1)
            # Get Probability
            confirmation_confidence = \
                nn.functional.softmax(confirmation_out)[0][confirmation_prediction].data.numpy()[0]
            # Convert tensor to int
            confirmation_prediction = confirmation_prediction.numpy()[0]
            if confirmation_prediction != im_label:
                print('Original image was predicted as:', im_label,
                      'with adversarial noise converted to:', confirmation_prediction,
                      'and predicted with confidence of:', confirmation_confidence)
                # Create the image for noise as: Original image - generated image
                noise_image = original_image - recreated_image
                from scipy.misc import imsave
                imsave(f'../gen/{name}_untargeted_adv_noise_from_' + str(im_label) + '_to_' +
                       str(confirmation_prediction) + '.jpg', noise_image)
                # Write image
                imsave(f'../gen/{name}_untargeted_adv_img_from_' + str(im_label) + '_to_' +
                       str(confirmation_prediction) + '.jpg', recreated_image)
                break


from torch.nn import ReLU


def save_gradient_images(gradient, file_name):
    """
        Exports the original gradient image

    Args:
        gradient (np arr): Numpy array of the gradient with shape (3, 224, 224)
        file_name (str): File name to be exported
    """
    if not os.path.exists('../results'):
        os.makedirs('../results')
    gradient = gradient - gradient.min()
    gradient /= gradient.max()
    gradient = np.uint8(gradient * 255).transpose(1, 2, 0)
    path_to_file = os.path.join('../results', file_name + '.jpg')
    # Convert RBG to GBR
    gradient = gradient[..., ::-1]
    from scipy.misc import imsave
    imsave(path_to_file, gradient)


def get_positive_negative_saliency(gradient):
    """
        Generates positive and negative saliency maps based on the gradient
    Args:
        gradient (numpy arr): Gradient of the operation to visualize

    returns:
        pos_saliency ( )
    """
    pos_saliency = (np.maximum(0, gradient) / gradient.max())
    neg_saliency = (np.maximum(0, -gradient) / -gradient.min())
    return pos_saliency, neg_saliency


class GuidedBackprop():
    """
       Produces gradients generated with guided back propagation from the given image
    """

    def __init__(self, model):
        self.model = model
        self.gradients = None
        # Put model in evaluation mode
        self.model.eval()
        self.update_relus()
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]

        # Register hook to the first layer
        first_layer = list(self.model._modules.items())[0][1]
        first_layer.register_backward_hook(hook_function)

    def update_relus(self):
        """
            Updates relu activation functions so that it only returns positive gradients
        """

        def relu_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, changes it to zero
            """
            if isinstance(module, ReLU):
                return (torch.clamp(grad_in[0], min=0.0),)

        def update(model):
            # Loop through layers, hook up ReLUs with relu_hook_function
            for pos, module in model._modules.items():
                # print(pos)
                if 'post' in pos or 'embed' in pos:
                    continue
                if isinstance(module, nn.Sequential):
                    # print(pos)
                    update(module)
                if isinstance(module, Bottleneck):
                    update(module)
                if isinstance(module, ReLU):
                    print('succ')
                    module.register_backward_hook(relu_hook_function)

        update(self.model)

    def generate_gradients(self, input_image, target_class):
        # Forward pass
        model_output = self.model(input_image)
        # Zero gradients
        self.model.zero_grad()
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Backward pass
        model_output.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = self.gradients.data.numpy()[0]
        return gradients_as_arr


def _concat(l):
    l0 = l[0]
    if l0.shape == ():
        return torch.stack(l)
    else:
        return torch.cat(l)


class TripletLoss(nn.Module):
    name = 'tri'

    def __init__(self, margin=0, mode='hard', args=None, **kwargs):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.mode = mode
        self.margin2 = args.margin2
        self.margin3 = args.margin3

    def forward(self, inputs, targets, dbg=False, cids=None):
        n = inputs.size(0)
        dist = calc_distmat(inputs, inputs)
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []

        for i in range(n):
            some_pos = dist[i][mask[i]]
            some_neg = dist[i][mask[i] == 0]

            neg = some_neg.min()
            pos = some_pos.max()

            dist_ap.append(pos)
            dist_an.append(neg)

        dist_ap = _concat(dist_ap) * self.margin2
        dist_an = _concat(dist_an) / self.margin3
        y = torch.ones(dist_an.size(), requires_grad=False).cuda()
        loss = self.ranking_loss(dist_an, dist_ap, y)
        prec = (dist_an.data > dist_ap.data).sum().type(
            torch.FloatTensor) / y.size(0)

        if not dbg:
            return loss, prec, dist
        else:
            return loss, prec, dist, dist_ap, dist_an


def clean():
    for fn in glob.glob(root_path + '/exps/fig/*.jpg'):
        name = fn
        name = fn.split('/')[-1].split('.')[0]
        from scipy.misc import imread


if __name__ == '__main__':
    import datetime

    tic = time.time()
    # run('')

    toc = clean()
    time.time()
    print('consume time ', toc - tic)
    # if toc - tic > 600:
    #     mail('tri center xent finish')
    print(datetime.datetime.now().strftime('%D-%H:%M:%S'))
