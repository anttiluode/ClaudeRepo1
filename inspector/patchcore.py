"""Frozen backbone patch features and a PatchCore memory bank (Roth et al., CVPR 2022).

Features follow PatchCore: WideResNet-50 layer2 + layer3, 3x3 local averaging, layer3 upsampled to
layer2's grid, concatenated and projected to 1024 dims by adaptive average pooling. The memory bank is a
greedy coreset of the training patches; a patch's anomaly score is its distance to the nearest bank
entry and an image's score is its largest patch score. In the energy language of the README this is the
unary term only: it asks 'has this patch been seen?', never 'do these patches belong together?'."""
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class Backbone:
    def __init__(self, device="cuda", pretrained=True, dim=1024):
        w = torchvision.models.Wide_ResNet50_2_Weights.IMAGENET1K_V1 if pretrained else None
        self.net = torchvision.models.wide_resnet50_2(weights=w).eval().to(device)
        self.device, self.dim = device, dim

    @torch.no_grad()
    def __call__(self, images):
        """images: float array (B, H, W, 3) in [0, 1]  ->  (B, h*w, dim) float32 on CPU, and (h, w)."""
        x = torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2).to(self.device)
        x = (x - MEAN.to(self.device)) / STD.to(self.device)
        n = self.net
        x = n.maxpool(n.relu(n.bn1(n.conv1(x))))
        l2 = n.layer2(n.layer1(x))
        l3 = n.layer3(l2)
        l2 = F.avg_pool2d(l2, 3, 1, 1)
        l3 = F.interpolate(F.avg_pool2d(l3, 3, 1, 1), size=l2.shape[-2:], mode="bilinear", align_corners=False)
        f = torch.cat([l2, l3], 1)
        B, C, h, w = f.shape
        f = f.permute(0, 2, 3, 1).reshape(B, h * w, C)
        f = F.adaptive_avg_pool1d(f, self.dim)
        return f.float().cpu(), (h, w)


def extract(backbone, paths, size, load, batch=8):
    out, hw = [], None
    for i in range(0, len(paths), batch):
        imgs = np.stack([load(p, size) for p in paths[i:i + batch]])
        f, hw = backbone(imgs)
        out.append(f)
    return torch.cat(out), hw


def greedy_coreset(feats, fraction, device, proj=128, seed=0):
    """Approximate k-center coreset on a random projection (as in PatchCore)."""
    N = feats.shape[0]
    n = max(1, int(round(N * fraction)))
    g = torch.Generator().manual_seed(seed)
    P = torch.randn(feats.shape[1], proj, generator=g).to(device)
    z = torch.cat([(c.to(device) @ P) for c in feats.split(65536)])
    first = int(torch.randint(N, (1,), generator=g))
    idx = [first]
    mind = ((z - z[first]) ** 2).sum(1)
    for _ in range(n - 1):
        i = int(torch.argmax(mind))
        idx.append(i)
        mind = torch.minimum(mind, ((z - z[i]) ** 2).sum(1))
    return feats[torch.tensor(idx)]


class PatchCore:
    def __init__(self, device="cuda", coreset=0.1):
        self.device, self.fraction = device, coreset

    def fit(self, train_feats):
        flat = train_feats.reshape(-1, train_feats.shape[-1])
        self.bank = greedy_coreset(flat, self.fraction, self.device).to(self.device)
        return self

    @torch.no_grad()
    def patch_scores(self, feats):
        """feats (B, P, C) -> (B, P) distance of every patch to its nearest bank entry."""
        B, Pn, C = feats.shape
        flat = feats.reshape(-1, C)
        out = []
        for q in flat.split(4096):
            d = torch.cdist(q.to(self.device), self.bank)
            out.append(d.min(1).values.cpu())
        return torch.cat(out).reshape(B, Pn)

    def score(self, feats):
        s = self.patch_scores(feats)
        return s.max(1).values.numpy(), s.numpy()
