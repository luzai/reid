from torch import nn
from lz import *
from torch.utils.data import DataLoader
from reid.utils.data.preprocessor import IndValuePreprocessor
from reid.utils import to_torch


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

        return pred, y2, info


def extract_cnn_embeddings(model, inputs, modules=None):

    model.eval()
    for ind, inp in enumerate(inputs):
        inputs[ind] = to_torch(inp)
    inputs = [Variable(x, volatile=True).cuda() for x in inputs]

    assert modules is None

    outputs = model(*inputs)
    outputs = outputs.data
    return outputs


class SiameseNet3(nn.Module):
    def __init__(self, base_model, tranform, embed_model):
        super(SiameseNet3, self).__init__()
        self.base_model = base_model
        self.transform = tranform
        self.embed_model = embed_model

    def forward(self, x1, y1, info=None):
        batch = self.base_model(x1)

        if info is not None:
            batch_np = batch.data.cpu().numpy()
            batch_np = batch_np.reshape((batch_np.shape[0], -1))
            batch_np = np.concatenate([batch_np, batch_np])
            info['features'] = batch_np.tolist()

        pair_samples = []
        for i in range(batch.size(0)):
            for j in range(batch.size(0)):
                pair_samples.append((i, j))

        data_loader = DataLoader(
            IndValuePreprocessor(batch.data.cpu()),
            sampler=pair_samples,
            batch_size=1024,
            num_workers=4, pin_memory=False
        )

        embeddings = []
        for i, inputs in enumerate(data_loader):
            outpus = extract_cnn_embeddings(self.embed_model, inputs)
            embeddings.append(outpus)

        embeddings = torch.cat(embeddings)
        embeddings = F.softmax(Variable(embeddings[:, 0],volatile=True), dim=0)
        embeddings = embeddings.view(batch.size(0), batch.size(0))
        pair1, pair2, y2, info = self.transform(batch, y1, info, embeddings)
        y2 = y2.type_as(y1.data)

        pred = self.embed_model(pair1, pair2)
        if info is not None:
            mypickle(embeddings.data.cpu().numpy(), 'dbg.hard.pkl')
            info['y2'] = y2.data.cpu().numpy().tolist()
            info['pred0'] = pred[:, 0].data.cpu().numpy().tolist()
            info['pred1'] = pred[:, 1].data.cpu().numpy().tolist()

        return pred, y2, info


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
