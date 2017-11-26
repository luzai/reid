from torch import nn
from lz import Database
from lz import *


class SiameseNet(nn.Module):
    def __init__(self, base_model, embed_model):
        super(SiameseNet, self).__init__()
        self.base_model = base_model
        self.embed_model = embed_model

    def forward(self, x1, x2):
        x1, x2 = self.base_model(x1), self.base_model(x2)
        if self.embed_model is None:
            return x1, x2
        return self.embed_model(x1, x2)


class SiameseNet2(nn.Module):
    def __init__(self, base_model, tranform, embed_model):
        super(SiameseNet2, self).__init__()
        self.base_model = base_model
        self.transform = tranform
        self.embed_model = embed_model
        # self.iter = 0
        # self.fid = Database('dbg.h5','a')

    def forward(self, x1, y1, info=None):
        batch = self.base_model(x1)

        if info is not None:
            batch_np = batch.data.cpu().numpy()
            batch_np = batch_np.reshape((batch_np.shape[0], -1))
            batch_np = np.concatenate([batch_np, batch_np])
            info['features'] = batch_np.tolist()

        pair1, pair2, y2, info = self.transform(batch, y1, info)
        y2 = y2.type_as(y1.data)


        pred = self.embed_model(pair1, pair2)
        if info is not None:
            info['y2'] = y2.data.cpu().numpy().tolist()

            info['pred0'] = pred[:, 0].data.cpu().numpy().tolist()
            info['pred1'] = pred[:, 1].data.cpu().numpy().tolist()

        # def save(k,v):
        #     self.fid[k]=v.data.cpu().numpy()
        #
        # save('x1',x1)
        # save('batch',batch)
        # save('pair1',pair1)
        # save('pair2',pair2)
        # save('pred', pred)
        #
        # save('y1',y1)
        # save('y2',y2)
        # self.fid.close()
        # exit(0)

        return pred, y2, info
        # x.size(), y.size(), outputs.size()
        # pair1.size(), pred.size()
        # outputs[:,12,7,3]
        # pair1[:,12,7,3]
        # pair2[:,12,7,3]
        # y2


class TripletNet(nn.Module):
    def __init__(self, base_model, embed_model):
        super(TripletNet, self).__init__()
        self.base_model = base_model
        self.embed_model = embed_model

    def forward(self, xa, xp, xn):
        xa, xp, xn = \
            self.base_model(xa), self.base_model(xp), self.base_model(xn)
        if self.embed_model is None:
            return xa, xp, xn
        xap = self.embed_model(xa, xp)
        xan = self.embed_model(xa, xn)
        return xan, xap  # to corporate with nn.MarginRankingLoss
