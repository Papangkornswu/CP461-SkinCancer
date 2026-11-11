"""Offline stand-in for `clip.load` (random weights, same architecture) so the code can be smoke-tested
without downloading OpenAI weights. Only used by tests."""
import clip
import torch
from clip.model import CLIP
from torchvision import transforms as T


def install(layers=12, width=768):
    torch.manual_seed(0)

    def fake_load(name, device="cpu", jit=False, **kw):
        m = CLIP(512, 224, layers, width, 32, 77, 49408, 512, 8, 12).float().eval()
        tf = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor(),
                        T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
        return m.to(device), tf
    clip.load = fake_load
    # deterministic weights across calls (checkpoint stores only tuned tensors)
    return fake_load
