import torch
from torch import nn


class MFM(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.out_features = in_features // 2

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return torch.max(x1, x2)


class LCNN_Exact(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        def block(in_c, out_c, k, s, pool=True, bn=True):
            layers = [
                nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=k[0] // 2),
                MFM(out_c),
            ]
            if pool:
                layers.append(nn.MaxPool2d(2))
            if bn:
                layers.append(nn.BatchNorm2d(out_c // 2))
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            block(1, 64, (5, 5), (1, 1)),  # -> MFM -> MaxPool -> BN
            block(32, 64, (1, 1), (1, 1), pool=False),
            block(32, 96, (3, 3), (1, 1)),
            block(48, 96, (1, 1), (1, 1), pool=False),
            block(48, 128, (3, 3), (1, 1)),
            block(64, 128, (1, 1), (1, 1), pool=False),
            block(64, 64, (3, 3), (1, 1)),
        )

        # --- Вычисляем размер фичей динамически ---
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 64, 600)
            out = self.features(dummy)
            feat_size = out.numel()

        self.fc1 = nn.Linear(feat_size, 160)
        self.mfm_fc1 = MFM(160)
        self.bn_fc1 = nn.BatchNorm1d(80)
        self.dropout = nn.Dropout(0.75)
        self.fc2 = nn.Linear(80, num_classes)

    def forward(self, data_object, labels):
        data_object = self.features(data_object)
        data_object = data_object.view(data_object.size(0), -1)
        data_object = self.fc1(data_object)
        data_object = self.mfm_fc1(data_object)
        data_object = self.bn_fc1(data_object)
        data_object = self.dropout(data_object)
        return self.fc2(data_object)
