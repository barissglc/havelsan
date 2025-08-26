# DCGAN ile Boat Görsel Üretimi

Bu proje, Deep Convolutional Generative Adversarial Network (DCGAN) kullanarak gemi/tekne görselleri üretmeyi amaçlar. PyTorch framework'ü kullanılarak geliştirilmiştir.

## Proje Yapısı

```
GAN/
├── egitim.py          # DCGAN eğitim kodu
├── generate.py        # Eğitilmiş modelden görsel üretimi
└── README.md          # Bu dosya
```

##  Özellikler

- **Modern DCGAN Mimarisi**: Generator ve Discriminator ağları
- **Spectral Normalization**: Discriminator'da kararlılık için
- **Hinge Loss Desteği**: Daha kararlı eğitim için alternatif loss fonksiyonu
- **EMA (Exponential Moving Average)**: Daha kaliteli sonuçlar için
- **Minibatch Standard Deviation**: Çeşitliliği artırmak için
- **Otomatik Checkpoint Yönetimi**: En iyi modelin otomatik kaydedilmesi
- **Dataset Preview**: Eğitim öncesi veri görselleştirmesi
- **Batch Generation**: Toplu görsel üretimi

## 🔧 Kurulum

Gerekli kütüphaneler:

```bash
pip install torch torchvision pillow tqdm
```

##  Kullanım

### Eğitim

```bash
python egitim.py --image_dir /path/to/boat/images --epochs 100 --batch_size 64
```

#### Eğitim Parametreleri

- `--image_dir`: Eğitim görsellerinin bulunduğu klasör
- `--out_dir`: Çıktıların kaydedileceği klasör (varsayılan: outputs)
- `--image_size`: Görsel boyutu (varsayılan: 64)
- `--batch_size`: Batch boyutu (varsayılan: 64)
- `--latent_dim`: Latent vektör boyutu (varsayılan: 128)
- `--epochs`: Epoch sayısı (varsayılan: 100)
- `--lr_g/--lr_d`: Generator/Discriminator öğrenme oranı (varsayılan: 0.0002)
- `--use_hinge`: Hinge loss kullanımı (varsayılan: True)
- `--d_spectral_norm`: Spectral normalization (varsayılan: True)

### Görsel Üretimi

```bash
python generate.py --ckpt outputs/checkpoints/bestmodel.pt --out_dir generated --n 64 --grid
```

#### Üretim Parametreleri

- `--ckpt`: Eğitilmiş model checkpoint'i
- `--out_dir`: Üretilen görsellerin kaydedileceği klasör
- `--n`: Üretilecek görsel sayısı
- `--grid`: Görselleri tek bir grid olarak kaydet

##  Model Mimarisi

### Generator
- Latent boyut: 128
- Upsample + Conv2D + BatchNorm + ReLU blokları
- Son katman: Tanh aktivasyonu
- Çıktı boyutu: 64x64x3

### Discriminator
- Spectral Normalization destekli Conv2D katmanları
- Minibatch Standard Deviation katmanı
- LeakyReLU aktivasyonu
- Hinge Loss veya BCE Loss desteği

##  Eğitim Süreci

1. **Veri Hazırlığı**: Görseller kare forma getirilir, 64x64 boyutuna resize edilir
2. **Augmentation**: Random horizontal flip (%20 olasılık)
3. **Normalization**: [-1, 1] aralığına normalize edilir
4. **Loss Hesabı**: Hinge loss veya Binary Cross Entropy
5. **Checkpoint**: Her epoch'ta model kaydedilir, en iyi model ayrıca saklanır

##  Çıktı Yapısı

```
outputs/
├── config.json              # Eğitim konfigürasyonu
├── dataset_preview.png      # Veri kümesi önizlemesi
├── samples/                 # Epoch bazlı örnek üretimler
│   ├── epoch_001.png
│   ├── epoch_002.png
│   └── ...
└── checkpoints/             # Model checkpoint'leri
    ├── bestmodel.pt         # En iyi performanslı model
    ├── dcgan_epoch_001.pt
    ├── dcgan_epoch_002.pt
    └── ...
```

##  Dipnot

⚠️: Bu projenin eğitim verileri ve sonuçları boyut kısıtlamaları nedeniyle GitHub'a yüklenememiştir. Ancak model başarıyla eğitilmiş ve sonuçlar Kağan hocaya sunulmuştur. Hoca, üretilen gemi görsellerinin kalitesinden ve modelin performansından memnun kaldığını belirtmiştir.


##  Teknik Detaylar

- **Framework**: PyTorch
- **Model**: DCGAN (Deep Convolutional GAN)
- **Loss**: Hinge Loss / Binary Cross Entropy
- **Optimizer**: Adam
- **Normalization**: Spectral Normalization, Batch Normalization
- **Data Augmentation**: Random Horizontal Flip, Square Padding
