# SD3.5 + IP‑Adapter (Img2Img) — v1 & v2

> Stable Diffusion 3.5 **Large** + **IP‑Adapter (SigLIP)** ile görüntüden‑görüntüye üretim. İki varyant:
>
> * **v1**: Prompt’lu, **resize** var → küçük kaynaklarda kontrollü iyileştirme ve stil dokunuşu.
> * **v2**: **Promptsuz**, **orijinal boyut** → giriş kompozisyonuna maksimum sadakat (yalnızca 16’ya pad).

---

## İçindekiler

* [Genel Bakış](#genel-bakış)
* [v1 vs v2 (Özet Tablo)](#v1-vs-v2-özet-tablo)
* [Kurulum](#kurulum)
* [Model Erişimi ve HF Girişi](#model-erişimi-ve-hf-girişi)
* [Ortak Bileşenler ve Akış](#ortak-bileşenler-ve-akış)
* [Parametreler (Anlamı ve Önerilen Aralıklar)](#parametreler-anlamı-ve-önerilen-aralıklar)
* [Klasörler, IO ve Çıktı İsimlendirme](#klasörler-io-ve-çıktı-isimlendirme)
* [v1 Kullanımı (Prompt’lu + Resize)](#v1-kullanımı-promptlu--resize)
* [v2 Kullanımı (Promptsuz + Orijinal Boyut)](#v2-kullanımı-promptsuz--orijinal-boyut)
* [Hazır Tarifler (Reçeteler)](#hazır-tarifler-reçeteler)
* [Performans & VRAM İpuçları](#performans--vram-ipuçları)
* [Sürüm Notları](#sürüm-notları)

---

## Genel Bakış

Bu proje, **SD3.5-Large** modelini **IP‑Adapter** ile koşullayıp, giriş görüntüsünün kompozisyonunu (kadraj, şekil, düzen) koruyarak yeni görüntüler üretir.

* **v1**: Küçük ve düşük kaliteli kaynaklarda, uzun kenara göre **yeniden boyutlandırma** ve **prompt/negatif prompt** ile daha kontrollü stil veya detay iyileştirmesi sağlar.
* **v2**: **Promptu boş** bırakarak modelin metinsel yönlendirmesini devre dışı bırakır; görüntüyü **ölçeklemez**, yalnızca 16’nın katına **pad** eder; böylece giriş yapısına en yüksek sadakat korunur. Daha çok büyük görüntüler için uygundur. Küçük görüntüler için versiyon 1 tercih edilmelidir.

FULL_IMAGE ve ONLY_BOAT klasörleri Stable Diffusion kullanılarak üretilen görselleri içerir.  
FULL_IMAGE klasörü tüm fotoğrafları (v2), ONLY_BOAT ise yalnızca tekne fotoğraflarını (v1) barındırır.

## v1 vs v2 (Özet Tablo)

| Özellik               | v1 (Prompt’lu + Resize)                       | v2 (Promptsuz + Orijinal Boyut)                 |
| --------------------- | --------------------------------------------- | ----------------------------------------------- |
| Prompt etkisi         | **Var** (`BASE_PROMPT`, `NEG_PROMPT`)         | **Yok** (`""`)                                  |
| Boyut politikası      | **Uzun kenar** → `LONG_EDGE` (16’ya yuvarlar) | **No resize**, sadece **pad to 16** (opsiyonel) |
| Kompozisyona sadakat  | Orta‑Yüksek (parametrelere bağlı)             | Çok yüksek                                      |
| Aşırı küçük görseller | `pre_upscale_min` ile önce büyütür            | Olduğu gibi (yalnız pad)                        |
| Kullanım amacı        | Stil/iyileştirme + dataset augment            | Yapıyı koruyarak temizlik                       |
| Varsayılan CFG        | 4.3                                           | 1.0 (fiilen kapalı gibi)                        |

---

## Kurulum

Colab veya yerelde aynı paketler:

```bash
pip install -U "diffusers>=0.33.0" "transformers>=4.43.0" accelerate safetensors pillow opencv-python bitsandbytes
```

* Python **3.10–3.11** önerilir.
* CUDA destekli PyTorch tavsiye edilir (CPU çalışır ama yavaştır).

---

## Model Erişimi ve HF Girişi

Model depolarına erişmek için Hugging Face girişine ihtiyaç duyabilirsiniz:

```python
from huggingface_hub import login
login()  # HF token girmeniz gerekebilir
```

> SD3.5 ve IP‑Adapter için lisans/koşulları kabul ettiğinizden emin olun.

---

## Ortak Bileşenler ve Akış

* **Model:** `stabilityai/stable-diffusion-3.5-large`
* **IP‑Adapter:** `InstantX/SD3.5-Large-IP-Adapter` (ağırlık dosyası: `ip-adapter.bin`, rev: `f1f54ca369ae759f9278ae9c87d46def9f133c78`)
* **Görüntü encoder:** `google/siglip-so400m-patch14-384`
* **Pipeline:** `StableDiffusion3Img2ImgPipeline` (Diffusers)
* **Optimize:** `enable_sdpa()` (Torch 2.x) veya `enable_attention_slicing()`, VAE slicing/tiling, **CPU offload**
* **Text encoder 3 (T5) düşürme:** `DROP_T5=True` ile VRAM kazanımı
* **Safety checker:** kapatılabilir (üretim sorumluluğu kullanıcıda)

**Akış:** Görselleri topla → (v1: min‑upscale + resize / v2: pad‑to‑16) → SigLIP ile IP‑Adapter embed veya görsel → `img2img` çağrısı → çıktı kaydet → klasörü zip’le.

Not: img2img çağrısında height/width verilmezse Diffusers çıktıyı modelin varsayılan çözünürlüğü olan 1024×1024’e yuvarlar. İstediğin boyutu korumak için w,h = init_image.size alıp çağrıya (width=w, height=h,) eklenmeli hocam.

---

## Parametreler (Anlamı ve Önerilen Aralıklar)

* `BASE_PROMPT` / `NEG_PROMPT`

  * (v1’de etkili) Metinsel rehberlik; negatifte `lowres, blurry, text, watermark, logo, duplicate` vb.

* `GUIDANCE` (CFG scale)

  * Prompt’a uyum derecesi.
  * **v1:** 3.5–5.5 iyi aralık (varsayılan 4.3).
  * **v2:** 1.0 (fiilen kapalı → giriş sadakati maksimize).

* `STEPS` (adım sayısı)

  * 20–28 tipik; daha fazlası maliyetli, kazancı sınırlı.

* `STRENGTH` (img2img gücü)

  * **Düşük (0.18–0.26):** Kompozisyonu iyi korur.
  * **Yüksek (0.30+):** Prompt/stil baskınlaşır, girişten sapma artar.
  * Varsayılanlar: v1=0.28, v2=0.24.

* `LONG_EDGE` (yalnız **v1**)

  * Uzun kenara göre yeniden boyutlandırma (16’nın katına yuvarlar).
  * **512–768** aralığı pratik; küçük kaynakta **768** iyi.

* `IP_SCALE`

  * IP‑Adapter’ın **görsel kompozisyon kilidi** gücü.
  * **0.5–0.7:** Prompt/stil daha baskın.
  * **0.8–1.0:** Girdi kompozisyonu daha baskın.

* `USE_EMBEDS`

  * `True`: `prepare_ip_adapter_image_embeds` ile tek seferlik embed → hız/istikrar.
  * `False`: doğrudan `ip_adapter_image` geçir.

* `VARIANTS_PER_IMAGE`

  * Aynı girişten kaç varyant; dataset augment için 2–3 faydalı.

* `BASE_SEED`

  * Determinizm: aynı giriş + seed → aynı çıktı.

* `PAD_TO_MULTIPLE_OF_16` (yalnız **v2**)

  * Boyutları 16’nın katına pad eder.

> **Öneri (kişisel):** Hocam bence küçük ve zor görsellerde `IP_SCALE`’i **0.7–0.9**, `STRENGTH`’i **0.20–0.26** aralığında tutmak, yapıyı korurken temiz sonuç verir.

---

## Klasörler, IO ve Çıktı İsimlendirme

* **Girdi klasörleri** (örnek):

  * v1: `CANDIDATE_DIRS = ["/content/drive/MyDrive/boat_dataset/roboflow_onlyboat"]`
  * v2: `CANDIDATE_DIRS = ["/content/drive/MyDrive/boat_dataset/fullphoto"]`

* **Çıktı klasörleri**:

  * v1: `/content/boat_out_sd35_i2i_ip`
  * v2: `/content/boat_out_sd35_i2i_ip4`

* **İsimlendirme (v1 örnek)**:
  `name_sd35IP_L{LONG_EDGE}_st{STRENGTH}_ip{IP_SCALE}_gs{GUIDANCE}_s{STEPS}[_vK].png`

* **Arşiv oluşturma**:

  * v1: `!zip -r /content/BOAT.zip /content/boat_out_sd35_i2i_ip`
  * v2: `!zip -r /content/file6.zip /content/boat_out_sd35_i2i_ip4`

---

## v1 Kullanımı (Prompt’lu + Resize)

### Başlıca Farklar

* Çok küçük görseller için `pre_upscale_min` ile önce min eşik üstüne çıkarır.
* Ardından `LONG_EDGE` ile uzun kenara göre **resize** (LANCZOS / çok küçükte OpenCV Bicubic).
* Prompt/negatif prompt **etkili**; stil ve detay kontrolü daha yüksek.

### Örnek Parametre Bloğu

```python
# --- Prompts ---
BASE_PROMPT = (
    "photorealistic small boat on the sea, natural lighting, sharp hull edges,"
    "realistic water ripples and reflections"
)
NEG_PROMPT = "lowres, blurry, cartoon, duplicate boats, text, watermark, logo, deformed hull"

# --- Parametreler ---
GUIDANCE = 4.3
STEPS    = 24
STRENGTH = 0.28
LONG_EDGE= 768
IP_SCALE = 0.70
USE_EMBEDS = True
VARIANTS_PER_IMAGE = 1
BASE_SEED = 2025
```

### Hızlı Başlangıç (Benim Tercihim)

```python
LONG_EDGE=768; STRENGTH=0.26; GUIDANCE=4.5; IP_SCALE=0.75; STEPS=24
```

> **İpucu:** Promptu biraz öne çıkarmak için `STRENGTH`’i 0.30 civarına, `GUIDANCE`’ı 4.5–5.0’a, `IP_SCALE`’i 0.6–0.7’ye çekebilirsin.

---

## v2 Kullanımı (Promptsuz + Orijinal Boyut)

### Başlıca Farklar

* **Resize yok**; yalnızca (opsiyonel) **16’ya pad**.
* `BASE_PROMPT = NEG_PROMPT = ""` → metin etkisi **yok**.
* Girdi kompozisyonu **maksimum** korunur (özellikle detection‑friendly çıktı isterken ideal).

### Örnek Parametre Bloğu

```python
BASE_PROMPT = ""
NEG_PROMPT  = ""

GUIDANCE           = 1.0
STEPS              = 24
STRENGTH           = 0.24
IP_SCALE           = 0.90
USE_EMBEDS         = True
VARIANTS_PER_IMAGE = 1
BASE_SEED          = 2025

PAD_TO_MULTIPLE_OF_16 = True
```

### Hızlı Başlangıç (Benim Tercihim)

```python
GUIDANCE=1.0; STRENGTH=0.22; IP_SCALE=0.92; STEPS=24; PAD_TO_MULTIPLE_OF_16=True
```

> **İpucu:** Temizlik/iyileştirme gücünü artırmak için `STRENGTH`’i 0.24–0.26’ya, çok **aynı** kalması istenirse 0.20–0.22’ye çek.

---

## Hazır Tarifler (Reçeteler)

**A) Kompozisyona çok sadık (neredeyse aynı kadraj)**

* v2: `GUIDANCE=1.0`, `STRENGTH=0.20–0.24`, `IP_SCALE=0.9–1.0`.

**B) Promptu biraz öne çıkar (stil/renk dokunuşu)**

* v1: `STRENGTH≈0.28–0.32`, `IP_SCALE≈0.6–0.75`, `GUIDANCE≈4.5–5.0`.

**C) Aşırı küçük kaynak (≤32px uzun kenar)**

* v1: `LONG_EDGE=512–768`, `STRENGTH≈0.20–0.24`, `IP_SCALE≈0.8–0.9`, `STEPS≈26`.

**D) Çok gürültülü/kirli giriş (temizlik)**

* v2: `STRENGTH≈0.24–0.26`, `IP_SCALE≈0.9` (gerekirse v1’e geçip sade negatiflerle destekle).

---

## Performans & VRAM İpuçları

* **VRAM düşür:** `pipe.enable_model_cpu_offload()`, VAE slicing/tiling **açık** kalsın.
* **Hızlandır:** CUDA + `enable_sdpa()` (Torch 2.x).
* **OOM durumunda:** `LONG_EDGE` (v1) düşür, `STEPS` azalt, `VARIANTS_PER_IMAGE=1`, `DROP_T5=True` kalsın.
* **Toplu üretim:** `USE_EMBEDS=True` ile her resim için embed’i bir kez hesaplamak hız kazandırır.


## Sürüm Notları

* **v1:** Prompt etkisi açık, min‑upscale + `LONG_EDGE` ile resize, küçük objelerde kontrollü stil/iyileştirme.
* **v2:** Prompt kapalı, **no‑resize**, pad‑to‑16; detection/pipeline sadakati yüksek çıktılar.

