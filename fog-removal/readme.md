# Algoritma Açıklaması: Dark Channel Prior

Bu betik, sisli görüntüleri temizlemek için Dark Channel Prior (DCP) adı verilen bir algoritma kullanır. Algoritmanın temel mantığı, sisli olmayan dış mekan görüntülerinin çoğunda, bazı piksellerin en az bir renk kanalında (RGB) çok düşük yoğunluk değerlerine sahip olduğu gözlemine dayanır. Bu "karanlık pikseller" topluluğuna "dark channel" denir.

## Algoritma şu adımları izler:

### 1. Dark Channel'ı Hesapla (dark_channel)
Her piksel için üç renk kanalının (R, G, B) en düşük değerini bularak bir "dark channel" haritası oluşturur.
Sis, bu karanlık piksellerin parlaklığını artırır. Dolayısıyla, dark channel'daki yoğunluk, sisin kalınlığı hakkında bir ipucu verir.

### 2. Atmosfer Işığını Tahmin Et (atmospheric_light)
Görüntüdeki en sisli bölgeyi tespit eder. Bu genellikle dark channel'daki en parlak %0.1'lik piksellere karşılık gelir.
Bu en parlak piksellerin orijinal görüntüdeki ortalama renk değeri, atmosferik ışık (sisi aydınlatan ortam ışığı, A olarak adlandırılır) olarak kabul edilir.

### 3. İletim Haritasını (Transmission Map) Hesapla (transmission)
Bu harita, her piksel için ne kadar ışığın kameraya ulaştığını (sis tarafından engellenmediğini) gösterir. Değer 0 (tamamen sisli) ile 1 (sis yok) arasında değişir.
Formül: 1 - w * dark_channel(I/A)
- I: Sisli görüntü
- A: Atmosfer ışığı
- w (omega): Görüntüde bir miktar sis bırakarak daha doğal bir görünüm elde etmeyi sağlayan bir sabittir (genellikle 0.95).

### 4. İletim Haritasını İyileştir (guided)
Ham iletim haritası genellikle bloklu ve pürüzlüdür. Guided Filter kullanılarak bu harita yumuşatılır ve nesne kenarları daha belirgin hale getirilir. Bu, sonuçtaki görüntüde hale (halo) etkilerinin oluşmasını engeller.

### 5. Görüntüyü Kurtar (recover)
Sisli bir görüntünün oluşumunu modelleyen fiziksel formül tersine çevrilerek orijinal, sissiz görüntü (J) kurtarılır.
- Sis Modeli: I = J*t + A*(1-t)
- Kurtarma Formülü: J = (I - A) / t + A
- t: İyileştirilmiş iletim haritası.

## 6. Algoritmanın Genel Mantığı
- **Gözlem**: Sisli olmayan alanlarda her zaman çok koyu pikseller bulunur.
- **Sis Varlığında**: Bu koyu pikseller daha açık görünür.
- **Dark Channel**: Koyu pikselleri tespit eder.
- **Atmospheric Light**: Sisin rengini ve yoğunluğunu belirler.
- **Transmission**: Her pikseldeki sis yoğunluğunu hesaplar.
- **Recovery**: Matematiksel formüllerle orijinal, sissiz görüntüyü geri kazandırır.

Bu adımlar sonucunda, orijinal sahnenin renkleri ve detayları geri kazanılarak net bir görüntü elde edilir.