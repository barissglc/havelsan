# HAVELSAN STAJ

⚠️⚠️️️⚠️️️️️ **Önemli Not**: Bu repo gizlilik bilgisi içermemektedir. Burada paylaşılan kodlar ve yapay zeka modelleri, şirket içi yapılan diğer çalışmalardan tamamen ayrıdır. Staj süresince şirket için yaptığım diğer çalışmalar gizlidir ve asla paylaşılmamıştır. (Gizlilik içeren bilgiler, fotoğraflar, LLM Çalışmaları vb.) ⚠️️️⚠️️️⚠️️️

1.  **Sis Giderme (Fog Removal)**: Görüntülerdeki sis ve pusu ortadan kaldırmak için "Dark Channel Prior" algoritmasını kullanan bir Python betiği.
2.  **GAN ile Görüntü Üretimi**: PyTorch kullanılarak geliştirilmiş bir DCGAN (Deep Convolutional Generative Adversarial Network) ile tekne/gemi görselleri üreten bir proje.
3.  **Stable Diffusion ile Görüntüden Görüntüye Dönüşüm**: Stable Diffusion 3.5 ve IP-Adapter kullanarak mevcut görselleri yeniden yorumlayan ve iyileştiren Jupyter Notebook tabanlı bir proje.
4.  **RT-DETR Fine-Tuning & TensorRT**: Özel veri seti üzerinde RT-DETR modelinin fine-tune edilmesi ve Torch-TensorRT VE ONNX ile gerçek zamanlı hızlandırılmış çıkarım.

---

## 1. Sis Giderme (`fog-removal`)

Bu proje, sisli fotoğrafları netleştirmek için **Dark Channel Prior (DCP)** algoritmasını uygular. Algoritma, sisli olmayan görüntülerdeki "karanlık pikseller" gözlemine dayanarak atmosferik ışığı ve iletim haritasını tahmin eder ve bu bilgileri kullanarak orijinal, sissiz görüntüyü matematiksel olarak kurtarır.

- **Teknoloji**: Python, OpenCV, NumPy
- **Ana Dosya**: `fog_removal.py`

Daha fazla teknik detay ve algoritmanın adımları için [fog-removal/readme.md](fog-removal/readme.md) dosyasına göz atın.

### Kullanım

Giriş ve çıkış klasörlerini belirterek `fog_removal.py` betiğini çalıştırabilirsiniz.

```bash
python fog-removal/fog_removal.py
```

---

## 2. GAN ile Görüntü Üretimi (`GAN`)

Bu proje, **DCGAN** mimarisi kullanarak sentetik tekne/gemi görselleri üretir. Proje, bir `egitim.py` betiği ile modelin eğitilmesini ve bir `generate.py` betiği ile eğitilmiş modelden yeni görseller üretilmesini sağlar.

- **Teknoloji**: PyTorch, DCGAN
- **Ana Dosyalar**: `egitim.py`, `generate.py`

Model mimarisi, eğitim süreci ve kullanım detayları için [GAN/README.md](GAN/README.md) dosyasını inceleyebilirsiniz.

### Kullanım

**Modeli Eğitme:**

```bash
python GAN/egitim.py --image_dir <veri_seti_klasoru> --epochs 100
```

**Görsel Üretme:**

```bash
python GAN/generate.py --ckpt <kayitli_model_yolu> --n 64
```

---

## 3. Stable Diffusion ile Görüntüden Görüntüye Dönüşüm (`SD-GENERATED`)

Bu bölümde, **Stable Diffusion 3.5 Large** modeli ve **IP-Adapter** kullanılarak görüntüden görüntüye dönüşüm (img2img) işlemleri gerçekleştirilir. İki farklı yaklaşım için iki ayrı Jupyter Notebook bulunmaktadır:

1.  **StableDiffusionv1.ipynb**: Metinsel komutlar (prompt) ile birlikte yeniden boyutlandırma yaparak küçük veya düşük kaliteli görselleri iyileştirmeyi ve stilize etmeyi hedefler.
2.  **StableDiffusionv2.ipynb**: Metinsel komut kullanmadan, orijinal boyutu koruyarak (sadece 16'nın katına tamamlama) giriş görüntüsünün kompozisyonuna maksimum sadakatle temizlik ve iyileştirme yapar.

- **Teknoloji**: Diffusers, PyTorch, Stable Diffusion 3.5, IP-Adapter
- **Notebooks**: `StableDiffusionv1.ipynb`, `StableDiffusionv2.ipynb`
- **Üretilen Görseller**: `SD-GENERATED/FULL_IMAGE/` ve `SD-GENERATED/ONLY_BOAT/` klasörlerinde bulunur.

Parametreler, kullanım senaryoları ve iki versiyon arasındaki farklar hakkında detaylı bilgi için [SD-GENERATED/readme.md](SD-GENERATED/readme.md) dosyasına bakın.

---

## 4. RT-DETR Nesne Tespiti Fine-Tuning & TensorRT Hızlandırma (`rt-detr-finetuning-tensorrt`)

Bu bölümde Roboflow üzerinden indirilen özel veri seti üzerinde **RT-DETR (PekingU)** modeli (başlangıç checkpoint: `PekingU/rtdetr_r18vd`) HuggingFace ekosistemi kullanılarak fine-tune edilmiştir. Eğitim sonrasında model, gerçek zamanlı çıkarım için **Torch-TensorRT** ile derlenmiş (FP16 denemesi, başarısızsa FP32 fallback) ve video üzerinde FPS ölçümleri yapılmıştır. Tüm süreç `rt_detr_finetuning_LAST.ipynb` defterinde yer alır.

Öne Çıkan Adımlar (Özet):
- Kurulum: `transformers`, `accelerate`, `supervision`, `roboflow`, `albumentations`, `torchmetrics`, opsiyonel `torch-tensorrt`.
- Veri: Roboflow API ile YOLOv5 formatında train / valid indirilip özel `SimpleDataset` ile COCO benzeri yapıya dönüştürülür.
- Eğitim: Birden fazla `TrainingArguments` denemesi; `EarlyStopping (patience=4)` + `load_best_model_at_end=True`.
- Değerlendirme: mAP@50 ve mAP@50-95 `supervision.metrics.MeanAveragePrecision` ile; ek olarak IoU matrisi ve örnek görseller.
- Kaydetme: En iyi checkpoint + `processor` arşivlenir (zip).
- TensorRT: `torch_tensorrt.compile` ile engine oluşturulur; FP16 başarısız olursa FP32 fallback.
- Video Inference: Warmup, zamanlama, ortalama/anlık FPS, kutu çizimi ve çıktı video kaydı.

Hızlı Başlangıç (PowerShell):
```powershell
# API anahtarlarını ortam değişkeni olarak ayarlayın
$Env:ROBOFLOW_API_KEY = "YOUR_KEY"
$Env:WANDB_API_KEY    = "YOUR_KEY"   # (opsiyonel izlemede)

# Gerekli paketler
pip install -U transformers accelerate supervision roboflow albumentations torchmetrics torch-tensorrt
```

Notlar:
- Farklı giriş çözünürlüğü kullanmak isterseniz TensorRT engine'i yeniden derleyin.
- VRAM yetersizliğinde `per_device_train_batch_size` azaltıp `gradient_accumulation_steps` artırın.
- FP16 derlemesi hata verirse otomatik FP32 ile devam edebilirsiniz.

Detaylı anlatım, grafikler ve sonuçlar için: [`rt-detr-finetuning-tensorrt/readme.md`](rt-detr-finetuning-tensorrt/readme.md) ve defter: `rt-detr-finetuning-tensorrt/rt_detr_finetuning_LAST.ipynb`.

