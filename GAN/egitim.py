from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils as vutils
from tqdm import tqdm
import torch.nn.functional as F


# -----------------------------
# Veri Kümesi
# -----------------------------


class SimpleImageDataset(Dataset):
    """Alt klasör yapısı gerektirmeden bir klasördeki resimleri okur."""

    def __init__(self, directory: Path, image_size: int) -> None:
        self.directory = Path(directory)
        self.image_size = int(image_size)

        patterns = ["*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG"]
        image_paths = sorted({p for pattern in patterns for p in self.directory.glob(pattern)}, key=lambda p: p.name.lower())

        # Açılabilenleri filtrele
        self.files: List[Path] = []
        for p in image_paths:
            try:
                with Image.open(p) as img:
                    img.verify()
                self.files.append(p)
            except Exception:
                # Bozuk/okunamayan görseli atla
                continue

        if not self.files:
            raise RuntimeError(f"'{self.directory}' içinde uygun görsel bulunamadı.")

        class SquarePad:
            def __call__(self, img: Image.Image) -> Image.Image:
                w, h = img.size
                if w == h:
                    return img
                s = max(w, h)
                pad_left = (s - w) // 2
                pad_top = (s - h) // 2
                pad_right = s - w - pad_left
                pad_bottom = s - h - pad_top
                return transforms.functional.pad(img, [pad_left, pad_top, pad_right, pad_bottom], fill=0, padding_mode="edge")

        self.tx = transforms.Compose(
            [
                SquarePad(),
                transforms.Resize((self.image_size, self.image_size), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:  # type: ignore[override]
        path = self.files[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
            return self.tx(img)


# -----------------------------
# DCGAN Modelleri
# -----------------------------


class Generator(nn.Module):
    def __init__(self, latent_dim: int, feature_maps: int = 64, out_channels: int = 3) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.feature_maps = feature_maps
        self.fc = nn.Linear(latent_dim, feature_maps * 8 * 4 * 4)

        def block(in_ch: int, out_ch: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(True),
            )

        self.up1 = block(feature_maps * 8, feature_maps * 4)  # 4 -> 8
        self.up2 = block(feature_maps * 4, feature_maps * 2)  # 8 -> 16
        self.up3 = block(feature_maps * 2, feature_maps)      # 16 -> 32
        self.up4 = block(feature_maps, feature_maps)          # 32 -> 64
        self.to_rgb = nn.Conv2d(feature_maps, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, z: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        z = z.view(z.size(0), -1)
        x = self.fc(z)
        x = x.view(z.size(0), self.feature_maps * 8, 4, 4)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.to_rgb(x)
        return self.tanh(x)


class MinibatchStdDev(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        batch, _, h, w = x.shape
        if batch == 1:
            return x
        std = torch.sqrt(x.var(dim=0, unbiased=False) + 1e-8)
        mean_std = std.mean().view(1, 1, 1, 1).expand(batch, 1, h, w)
        return torch.cat([x, mean_std], dim=1)


class Discriminator(nn.Module):
    def __init__(self, in_channels: int = 3, feature_maps: int = 64, spectral_norm: bool = True) -> None:
        super().__init__()
        def maybe_sn(layer: nn.Module) -> nn.Module:
            return nn.utils.spectral_norm(layer) if spectral_norm else layer

        self.features = nn.Sequential(
            maybe_sn(nn.Conv2d(in_channels, feature_maps, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            maybe_sn(nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            maybe_sn(nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            maybe_sn(nn.Conv2d(feature_maps * 4, feature_maps * 8, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.minibatch = MinibatchStdDev()
        self.final = maybe_sn(nn.Conv2d(feature_maps * 8 + 1, 1, 4, 1, 0, bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.features(x)
        x = self.minibatch(x)
        x = self.final(x)
        return x


def dcgan_weights_init(m: nn.Module) -> None:
    """DCGAN ağırlık başlatma (mean=0, std=0.02)."""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# -----------------------------
# Eğitim
# -----------------------------


@dataclass
class TrainConfig:
    image_dir: str
    out_dir: str = "outputs"
    image_size: int = 64
    batch_size: int = 64
    latent_dim: int = 128
    epochs: int = 100
    lr_g: float = 0.0002
    lr_d: float = 0.0002
    beta1: float = 0.5
    beta2: float = 0.999
    num_workers: int = 0  # Windows için güvenli varsayılan
    seed: int = 42
    sample_every: int = 1  # epoch bazlı
    use_hinge: bool = True
    d_spectral_norm: bool = True
    # En iyi modeli kaydetmek için izleme ayarları
    monitor: str = "lossG"          # "lossG" veya "lossD"
    monitor_mode: str = "min"       # "min" veya "max"


def ensure_dirs(*dirs: Iterable[Path]) -> None:
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def save_samples(generator: Generator, fixed_noise: torch.Tensor, epoch: int, samples_dir: Path) -> None:
    generator.eval()
    with torch.no_grad():
        fake = generator(fixed_noise).cpu()
    grid_path = samples_dir / f"epoch_{epoch:03d}.png"
    vutils.save_image(fake, grid_path.as_posix(), nrow=int(math.sqrt(fixed_noise.size(0))), normalize=True, value_range=(-1, 1))
    generator.train()


def save_dataset_preview(dataset: Dataset, out_path: Path, num: int = 64) -> None:
    dl = DataLoader(dataset, batch_size=num, shuffle=True)
    x = next(iter(dl))[:num]
    vutils.save_image(x, out_path.as_posix(), nrow=int(math.sqrt(num)), normalize=True, value_range=(-1, 1))


def train(config: TrainConfig) -> None:
    torch.manual_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Eğitim cihazı: {device}")
    if device.type == "cuda":
        try:
            print(f"CUDA etkin. GPU: {torch.cuda.get_device_name(0)}")
        except Exception:
            pass

    dataset = SimpleImageDataset(Path(config.image_dir), config.image_size)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    netG = Generator(latent_dim=config.latent_dim).to(device)
    netD = Discriminator(spectral_norm=config.d_spectral_norm).to(device)
    netG.apply(dcgan_weights_init)
    netD.apply(dcgan_weights_init)

    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=config.lr_d, betas=(config.beta1, config.beta2))
    optimizerG = optim.Adam(netG.parameters(), lr=config.lr_g, betas=(config.beta1, config.beta2))

    fixed_noise = torch.randn(16, config.latent_dim, 1, 1, device=device)

    out_dir = Path(config.out_dir)
    samples_dir = out_dir / "samples"
    ckpt_dir = out_dir / "checkpoints"
    ensure_dirs(out_dir, samples_dir, ckpt_dir)

    # Konfig kaydet
    (out_dir / "config.json").write_text(json.dumps(asdict(config), indent=2, ensure_ascii=False), encoding="utf-8")
    # Dataset preview kaydet
    try:
        save_dataset_preview(dataset, out_dir / "dataset_preview.png", num=min(64, config.batch_size * 2))
    except Exception:
        pass

    global_step = 0
    best_metric: float | None = None
    best_epoch: int | None = None
    for epoch in range(1, config.epochs + 1):
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{config.epochs}", leave=False)
        epoch_lossD_sum = 0.0
        epoch_lossG_sum = 0.0
        epoch_steps = 0
        for real in pbar:
            real = real.to(device)
            b_size = real.size(0)

            # -----------------
            # Discriminator (hinge destekli)
            # -----------------
            netD.zero_grad(set_to_none=True)
            output_real = netD(real).reshape(-1)
            noise = torch.randn(b_size, config.latent_dim, 1, 1, device=device)
            fake = netG(noise)
            output_fake = netD(fake.detach()).reshape(-1)

            if config.use_hinge:
                lossD_real = F.relu(1.0 - output_real).mean()
                lossD_fake = F.relu(1.0 + output_fake).mean()
                lossD = lossD_real + lossD_fake
            else:
                labels_real = torch.ones_like(output_real, device=device)
                labels_fake = torch.zeros_like(output_fake, device=device)
                lossD_real = criterion(output_real, labels_real)
                lossD_fake = criterion(output_fake, labels_fake)
                lossD = lossD_real + lossD_fake

            lossD.backward()
            optimizerD.step()

            # -----------------
            # Generator (hinge destekli)
            # -----------------
            netG.zero_grad(set_to_none=True)
            output_for_G = netD(fake).reshape(-1)
            if config.use_hinge:
                lossG = (-output_for_G).mean()
            else:
                labels_for_G = torch.ones_like(output_for_G, device=device)
                lossG = criterion(output_for_G, labels_for_G)
            lossG.backward()
            optimizerG.step()

            global_step += 1
            pbar.set_postfix({"lossD": f"{lossD.item():.3f}", "lossG": f"{lossG.item():.3f}"})
            epoch_lossD_sum += float(lossD.item())
            epoch_lossG_sum += float(lossG.item())
            epoch_steps += 1

        # Örnek kaydet
        if (epoch % config.sample_every) == 0:
            save_samples(netG, fixed_noise, epoch, samples_dir)

        # Epoch bitti: ortalama metrikleri hesapla
        if epoch_steps > 0:
            avg_lossD = epoch_lossD_sum / epoch_steps
            avg_lossG = epoch_lossG_sum / epoch_steps
        else:
            avg_lossD = float("inf")
            avg_lossG = float("inf")

        # En iyi modeli güncelle
        monitor_key = config.monitor.lower()
        current_metric = avg_lossG if monitor_key == "lossg" else avg_lossD
        is_better = (
            best_metric is None
            or (config.monitor_mode == "min" and current_metric < best_metric)
            or (config.monitor_mode == "max" and current_metric > best_metric)
        )
        if is_better:
            best_metric = current_metric
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "netG": netG.state_dict(),
                    "netD": netD.state_dict(),
                    "optG": optimizerG.state_dict(),
                    "optD": optimizerD.state_dict(),
                    "config": asdict(config),
                    "best_metric": best_metric,
                    "best_epoch": best_epoch,
                    "best_monitor": config.monitor,
                    "best_mode": config.monitor_mode,
                    "avg_lossD": avg_lossD,
                    "avg_lossG": avg_lossG,
                },
                (ckpt_dir / "bestmodel.pt").as_posix(),
            )

        # Checkpoint kaydet
        torch.save(
            {
                "epoch": epoch,
                "netG": netG.state_dict(),
                "netD": netD.state_dict(),
                "optG": optimizerG.state_dict(),
                "optD": optimizerD.state_dict(),
                "config": asdict(config),
                "avg_lossD": avg_lossD,
                "avg_lossG": avg_lossG,
            },
            (ckpt_dir / f"dcgan_epoch_{epoch:03d}.pt").as_posix(),
        )


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Boat fotoğrafları ile DCGAN eğitimi")
    parser.add_argument("--image_dir", type=str, default=".", help="Görsellerin bulunduğu klasör (varsayılan: .)")
    parser.add_argument("--out_dir", type=str, default="outputs", help="Çıktı klasörü (samples & checkpoints)")
    parser.add_argument("--image_size", type=int, default=64, help="Girdi boyutu (64 tavsiye edilir)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr_g", type=float, default=2e-4)
    parser.add_argument("--lr_d", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample_every", type=int, default=1)
    parser.add_argument("--no_hinge", action="store_true", help="Hinge loss kapat (BCE'ye geç)")
    parser.add_argument("--no_spectral", action="store_true", help="Discriminator'da SpectralNorm kapat")
    parser.add_argument("--monitor", type=str, default="lossG", choices=["lossG", "lossD"], help="En iyi model için izlenecek metrik")
    parser.add_argument("--monitor_mode", type=str, default="min", choices=["min", "max"], help="Metrik iyileşme yönü: min veya max")
    args = parser.parse_args()

    return TrainConfig(
        image_dir=args.image_dir,
        out_dir=args.out_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        lr_g=args.lr_g,
        lr_d=args.lr_d,
        beta1=args.beta1,
        beta2=args.beta2,
        num_workers=args.num_workers,
        seed=args.seed,
        sample_every=args.sample_every,
        use_hinge=(not args.no_hinge),
        d_spectral_norm=(not args.no_spectral),
        monitor=args.monitor,
        monitor_mode=args.monitor_mode,
    )


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)