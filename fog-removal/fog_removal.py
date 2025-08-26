
import cv2
import numpy as np
import os
from pathlib import Path
import time

# Yolllarr
SOURCE_DIR  = "sisli"
OUTPUT_DIR  = "sisli_dehazed"

# Parametreler (hocam bunlarla oynayabilirsiniz :D)
OMEGA   = 0.95
RADIUS  = 40
T0      = 0.10
GAMMA   = None    # hocam none olunca daha hızlı calisiyor biz fotoğraf farkını anlayamadık ama bence sizde bir bakın.
#multi processing vardı kapattik onla baya hızliydi ama zaten gpuda calisacak diye karistirmayalim dedik :D
start_time = time.time()


def dark_channel(img, k=15):
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    return cv2.erode(np.min(img, axis=2), kernel)

def atmospheric_light(img, dark, top=0.001):
    # Atmosferik isigi hesapla
    h, w = dark.shape
    num_pixels = max(int(h * w * top), 1)
    
    flat_dark = dark.ravel()
    indices = np.argpartition(flat_dark, -num_pixels)[-num_pixels:]
    return img.reshape(-1, 3)[indices].mean(0)

def transmission(img, A, w=OMEGA, k=15):
    # Transmisyon haritasini hesapla
    return 1 - w * dark_channel(img / A, k)

def guided_filter_optimized(I, p, r=RADIUS, eps=1e-3):
    # Daha az islemle optimize edilmis guided filter
    ksize = (r, r)
    
    
    mean_I = cv2.boxFilter(I, cv2.CV_32F, ksize)
    mean_p = cv2.boxFilter(p, cv2.CV_32F, ksize)
    
    
    corr_Ip = cv2.boxFilter(I * p, cv2.CV_32F, ksize)
    corr_II = cv2.boxFilter(I * I, cv2.CV_32F, ksize)
    
    cov_Ip = corr_Ip - mean_I * mean_p
    var_I = corr_II - mean_I * mean_I
    
    
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    
    
    return cv2.boxFilter(a, cv2.CV_32F, ksize) * I + cv2.boxFilter(b, cv2.CV_32F, ksize)

def recover(img, A, t, t0=T0, gamma=GAMMA):
    # Orijinal goruntuyu kurtarma islemi
    t = np.clip(t, t0, 1)[..., None]
    J = (img - A) / t + A
    J = np.clip(J, 0, 1)
    if gamma is not None:
        J = np.power(J, gamma)
    return J

def dehaze_bgr_optimized(bgr):
    # RGB goruntuusunu alir ve optimize edilmis sis giderme islemi uygular
    # Gereksiz renk donusumlerinden kacin
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    
    # Koyu kanal hesapla
    dark = dark_channel(rgb)
    
    # Atmosferik isigi hesapla
    A = atmospheric_light(rgb, dark)
    
    # Transmisyonu tahmin et
    t_est = transmission(rgb, A)
    
    # Guided filter icin gri tonlamaya cevir (daha verimli)
    gray = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    
    # Optimize edilmis guided filter uygula
    t_ref = guided_filter_optimized(gray, t_est)
    
    # Kurtarma islemi ve geri donusum
    result = recover(rgb, A, t_ref)
    return cv2.cvtColor((result * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


Path(OUTPUT_DIR).mkdir(exist_ok=True)
valid_ext = {".jpg", ".jpeg", ".png"}

# Tum resim dosyalarini topla
image_files = []
for root, _, files in os.walk(SOURCE_DIR):
    for fname in files:
        if Path(fname).suffix.lower() in valid_ext:
            image_files.append(Path(root) / fname)

print(f"{len(image_files)} adet resim bulundu")

# Resimleri tek tek isle
successful = 0
for fpath in image_files:
    try:
        print(f"{fpath.name} isleniyor")
        img_bgr = cv2.imread(str(fpath))
        if img_bgr is None:
            print(f"    {fpath.name} atlatildi (okunamadi)")
            continue
        
        dehazed = dehaze_bgr_optimized(img_bgr)
        out_path = Path(OUTPUT_DIR) / (fpath.stem + "_dehazed" + fpath.suffix)
        cv2.imwrite(str(out_path), dehazed)
        print(f"    {out_path.name} kaydedildi")
        successful += 1
        
    except Exception as e:
        print(f"    {fpath.name} islenirken hata: {e}")

print(f"\n{successful}/{len(image_files)} resim basariyla islendi")

print("\nTamamlandi! Tum resimler islendi.")
end_time = time.time()
duration = end_time - start_time
print(f"Toplam islem suresi: {duration:.2f} saniye ({duration/60:.1f} dakika)")
if len(image_files) > 0:
    print(f"Resim basina ortalama sure: {duration/len(image_files):.2f} saniye")