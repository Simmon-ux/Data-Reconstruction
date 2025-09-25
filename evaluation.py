# evaluation.py

"""
Quick visual eval for Antarctic SAT inpainting

What it does
- Picks a short consecutive slice from a Dataset and runs the model once.
- Builds a 5-row grid per sample: [input(image_masked), mask, output, output_comp, gt].
- Saves the grid image to disk (stays in z-score space; no denorm).

Signature
    evaluate(model, dataset, device, filename, n_samples=8, seed=None)

Args
- model    : PConv U-Net (or compatible) with forward(image, mask) → (output, _)
- dataset  : returns (image_masked, mask, gt) each shaped [1,H,W] (float32)
- device   : torch.device ('cuda' or 'cpu')
- filename : output path for the saved grid (e.g., './snapshots/antarctica/images/val_0001.jpg')
- n_samples: number of consecutive time steps to visualize
- seed     : optional RNG seed for reproducible start index

Output
- A single image file written via torchvision.save_image.

Notes
- No gradients; wraps inference in torch.no_grad().
- output_comp = mask*input + (1-mask)*output shows filled fields next to raw output.
- Works with any dataset length (chooses a valid window automatically).

Dependencies
- torch, torchvision
"""

import torch, random
from torchvision.utils import make_grid, save_image

def evaluate(model, dataset, device, filename, n_samples=8, seed=None):
    """
    Saves a quick preview grid using n_samples consecutive items from the dataset.
    Works for any dataset length; stays in z-score space (no RGB unnormalize).
    """
    rng = random.Random(seed)
    n = len(dataset)
    start = 0 if n <= n_samples else rng.randint(0, n - n_samples)

    batch = [dataset[i] for i in range(start, start + n_samples)]
    image, mask, gt = zip(*batch)
    image, mask, gt = torch.stack(image), torch.stack(mask), torch.stack(gt)

    with torch.no_grad():
        output, _ = model(image.to(device), mask.to(device))
    output = output.cpu()
    output_comp = mask * image + (1 - mask) * output

    grid = make_grid(torch.cat((image, mask, output, output_comp, gt), dim=0))
    save_image(grid, filename)
