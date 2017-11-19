from torch import nn


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

    def forward(self, x, y):
        outputs = self.base_model(x)

        pair1, pair2, y2 = self.transform(outputs, y)
        pred = self.embed_model(pair1, pair2)

        return pred, y2.type_as(y.data)


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
