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
from reid.utils.serialization import *
from reid.utils.dop import DopInfo

from tensorboardX import SummaryWriter

base_path = work_path + '/reid/work.use/'
pathf = base_path + '/tri6.combine.2'
# pathf = base_path + '/tri9.combine.3'
args = pickle_load(pathf + '/conf.pkl')
args.evaluate = True
args.adv_eval = True
args.rerank = False
args.dataset = args.dataset_val = 'cu03lbl'
args.batch_size = 128
args.resume = pathf + '/model_best.pth'
args.gpu = (3,)
print('arg is ', args)

def get_data(args):
    height, width = args.height, args.width
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
        T.RectScale(height, width),
        T.ToTensor(),
        normalizer,
    ])

    query_loader = DataLoader(
        DirPreprocessor('/home/xinglu/work/body.query/', test_transformer),
        batch_size=128,
        num_workers=12,
        shuffle=False, pin_memory=False)

    gallery_loader = DataLoader(
        DirPreprocessor('/home/xinglu/work/body.test/', test_transformer),
        batch_size=128,
        num_workers=12,
        shuffle=False, pin_memory=False)

    return query_loader, gallery_loader

lz.init_dev(args.gpu)
print('config is {}'.format(vars(args)))
if args.seed is not None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
cudnn.benchmark = True

# Create data loaders
assert args.num_instances > 1, "num_instances should be greater than 1"
assert args.batch_size % args.num_instances == 0, \
    'num_instances should divide batch_size'

# Create model
model = models.create(args.arch,
                      dropout=args.dropout,
                      pretrained=args.pretrained,
                      block_name=args.block_name,
                      block_name2=args.block_name2,
                      num_features=args.num_classes,
                      num_classes=999,
                      num_deform=args.num_deform,
                      fusion=args.fusion,
                      last_conv_stride=args.last_conv_stride,
                      last_conv_dilation=args.last_conv_dilation,
                      )

print(model)
param_mb = sum(p.numel() for p in model.parameters()) / 1000000.0
print('    Total params: %.2fM' % (param_mb))

if args.gpu is None or len(args.gpu) == 0:
    model = nn.DataParallel(model)
elif len(args.gpu) == 1:
    model = nn.DataParallel(model).cuda()
else:
    model = nn.DataParallel(model, device_ids=range(len(args.gpu))).cuda()
query_loader, gallery_loader = get_data(args)
gss = []
qss = []
qp = []
gp = []
for imgs, imgps in gallery_loader:
    imgs = imgs.cuda()
    bs, c, h, w = imgs.size()
    imgs = imgs.view(bs, c, h, w)
    features = model(imgs)[0]
    features = features.view(bs, -1)
    features = features.data.cpu().numpy()
    gss.extend(features)
    gp.extend(imgps)
for imgs, imgps in query_loader:
    imgs = imgs.cuda()
    bs, c, h, w = imgs.size()
    imgs = imgs.view(bs, c, h, w)
    features = model(imgs)[0]
    features = features.view(bs, -1)
    features = features.data.cpu().numpy()
    qss.extend(features)
    qp.extend(imgps)

gss = np.vstack(gss)
qss = np.vstack(qss, )
print(qss.shape, gss.shape)
from scipy.spatial.distance import cdist
dist = cdist(qss, gss)
dist.argsort(axis=1)[0][0]
