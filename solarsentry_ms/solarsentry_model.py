"""SolarSentry CNN in MindSpore (same architecture as the PyTorch prototype).

Layer names match the PyTorch version (b1.0.weight, b1.1.gamma, ..., fc.bias), so
convert_pt_to_ms.py can copy trained PyTorch weights straight across.
"""
import mindspore.nn as nn
import mindspore.ops as ops

CLASSES = ["No-Anomaly", "Cell", "Cell-Multi", "Cracking", "Hot-Spot", "Hot-Spot-Multi", "Shadowing",
           "Diode", "Diode-Multi", "Vegetation", "Soiling", "Offline-Module"]


def blk(i, o):
    # PyTorch defaults differ from MindSpore's, so they are set explicitly:
    #   conv: zero padding of 1 with a bias (MindSpore defaults: pad_mode="same", no bias)
    #   BN:   MindSpore momentum = 1 - PyTorch momentum (0.1), eps 1e-5 in both
    return nn.SequentialCell(
        nn.Conv2d(i, o, 3, pad_mode="pad", padding=1, has_bias=True),
        nn.BatchNorm2d(o, eps=1e-5, momentum=0.9), nn.ReLU(),
        nn.Conv2d(o, o, 3, pad_mode="pad", padding=1, has_bias=True),
        nn.BatchNorm2d(o, eps=1e-5, momentum=0.9), nn.ReLU())


class Net(nn.Cell):
    def __init__(self, n=12):
        super().__init__()
        self.b1, self.b2, self.b3 = blk(1, 32), blk(32, 64), blk(64, 128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop = nn.Dropout(p=0.3)
        self.fc = nn.Dense(128, n)

    def construct(self, x):
        x = self.pool(self.b1(x))
        x = self.pool(self.b2(x))
        x = self.b3(x)
        x = ops.mean(x, (2, 3))          # global average pool -> (N, 128)
        return self.fc(self.drop(x))


def predict_tta(net, x):
    """Softmax averaged over the image and its horizontal flip (test-time augmentation)."""
    return (ops.softmax(net(x), 1) + ops.softmax(net(ops.flip(x, (3,))), 1)) / 2
