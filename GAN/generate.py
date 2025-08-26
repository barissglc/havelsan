from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torchvision import utils as vutils

from egitim import Generator  


def load_checkpoint(ckpt_path: Path):
    ckpt = torch.load(ckpt_path.as_posix(), map_location="cpu")
    config = ckpt.get("config", {})
    state_dict = ckpt["netG"] if "netG" in ckpt else ckpt
    return state_dict, config


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def generate_images(
    ckpt_path: Path,
    out_dir: Path,
    n_images: int = 64,
    grid: bool = True,
) -> None:
    state_dict, config = load_checkpoint(ckpt_path)
    latent_dim = int(config.get("latent_dim", 128))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    netG = Generator(latent_dim=latent_dim).to(device)
    # EMA varsa onu tercih et
    if "netG_ema" in state_dict:
        netG.load_state_dict(state_dict["netG_ema"])  # type: ignore[index]
    else:
        netG.load_state_dict(state_dict)
    netG.eval()

    ensure_dir(out_dir)

    with torch.no_grad():
        noise = torch.randn(n_images, latent_dim, 1, 1, device=device)
        fake = netG(noise).cpu()

    if grid:
        # Kare ızgara olacak şekilde n'i üst kareye yuvarla
        nrow = int(math.ceil(math.sqrt(n_images)))
        grid_path = out_dir / "generated_grid.png"
        vutils.save_image(fake, grid_path.as_posix(), nrow=nrow, normalize=True, value_range=(-1, 1))
        print(f"Grid kaydedildi: {grid_path}")
    else:
        for i, img in enumerate(fake):
            img_path = out_dir / f"img_{i:04d}.png"
            vutils.save_image(img, img_path.as_posix(), normalize=True, value_range=(-1, 1))
        print(f"{n_images} adet görsel '{out_dir}' klasörüne kaydedildi.")


def parse_args():
    p = argparse.ArgumentParser(description="DCGAN checkpoint'ten görsel üretimi")
    p.add_argument("--ckpt", type=str, required=True, help="Checkpoint dosyası (dcgan_epoch_XXX.pt)")
    p.add_argument("--out_dir", type=str, default="generated", help="Üretim çıktılarının kaydedileceği klasör")
    p.add_argument("--n", type=int, default=64, help="Üretilecek görsel sayısı")
    p.add_argument("--grid", action="store_true", help="Görselleri tek bir grid olarak kaydet")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_images(Path(args.ckpt), Path(args.out_dir), n_images=args.n, grid=args.grid)


