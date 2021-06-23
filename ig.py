import torchvision.models import resnet50
from torchvision import transforms
import torch
from PIL import Image
from PIL.ImageFilter import GaussianBlur
from tqdm import tqdm


class IntegratedGradients:
    def __init__(self, model, n_steps):
        self.model = model
        self.n_steps = n_steps

    def heatmap(self, query, baseline):
        # normalized image tensors
        # image [1, c, h, w], image [1, c, h, w]
        interpolated = self._interpolate(query, baseline)
        gradients = self._compute_grads(interpolated)
        ig = self._integrate(gradients)
        attribution = (query - baseline) * ig
        heatmap = torch.abs(attribution.sum(0))
        # heatmap [1, c, h, w]
        return heatmap.cpu()

    def _interpolate(self, image, baseline):
        alphas = torch.linspace(0.0, 1.0, steps=self.n_steps + 1).cuda()
        alphas_ = alphas[..., None, None, None]
        baseline_ = baseline[None, ...]
        image_ = image[None, ...]
        delta = image_ - baseline_
        interpolated = baseline_ + alphas_ * delta
        return interpolated

    def _compute_grads(self, images):
        self.model.eval()  # user may have forgotten
        clones = images.clone()
        clones.requires_grad_()
        logit = self.model(clones)
        probs = torch.sigmoid(logit)
        probs.sum().backward()
        return clones.grad

    def _integrate(self, gradients):
        grads = (gradients[:-1] + gradients[1:]) / 2.
        ig = grads.mean(dim=0)
        return ig


class NoiseTunnel:
    def __init__(self, attributor, sigma, n_passes, squared=True):
        self.attributor = attributor
        self.sigma = sigma
        self.n_passes = n_passes
        self.squared = squared

    def heatmap(self, image, baseline):
        # normalized image tensors
        # image [1, c, h, w], baseline [1, c, h, w]
        hms = []
        for _ in tqdm(range(self.n_passes)):
            pimage = image + torch.randn(image.shape).cuda() * self.sigma
            hm = self.attributor.heatmap(pimage, baseline)
            hms.append(hm)
        hms = torch.stack(hms)
        # hms [1, c, h, w]
        if self.squared:
            hms = hms ** 2
        return hms.mean(0)


class ImageAttributor:
    def __init__(self, attributor, normalizer, kernel_width):
        self.attributor = attributor
        self.normalizer = normalizer
        self.kernel_width = kernel_width

    def heatmap(self, img: Image):
        inputs = self.normalizer(img).to(0)
        if self.kernel_width:
            blurred = img.filter(GaussianBlur(self.kernel_width))
            baseline = self.normalizer(blurred).to(0)
        else:
            white = Image.new("RGB", img.size, (255, 255, 255))
            baseline = self.normalizer(white).to(0)
        # sum over channels
        hm = self.attributor.heatmap(inputs, baseline)
        return self.to_img(hm)

    def to_img(self, hm):
        # scale to range [0., 1.]
        hm -= hm.min()
        hm /= hm.max()
        # clip outlier [0., 0.25] -> [0., 1.]
        hm = torch.clip(hm, max=0.25) * 4
        # flip high value == dark colour
        hm = ((1 - hm).numpy() * 255).astype("uint8")
        hm = np.repeat(hm[..., None], 3, axis=-1)
        hm = Image.fromarray(hm)
        return hm

