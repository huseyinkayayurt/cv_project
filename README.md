# CV Project - Araç Sayma (Şerit Bazlı)

Bu proje, bir yol videosu üzerinden arka plan çıkarımı ve basit izleme (centroid tracker)
kullanarak şerit bazında araç sayımı yapar. Sayım çizgisi video yüksekliğinin belirli
bir oranında konumlandırılır; şerit sınırları bu çizgi üzerinde otomatik olarak
tahmin edilir.

## Özellikler
- Şerit bazlı araç sayımı (1-6 arası şeritler)
- Otomatik şerit sınırı tahmini (sayım çizgisi üzerinde)
- Basit arka plan çıkarımı (MOG2) ve centroid tabanlı takip
- Görsel çıktı ve opsiyonel video kaydı
- Çeşitli parametrelerle kolay ayar

## Gereksinimler
- Python 3.9+
- OpenCV (`opencv-python`)
- NumPy

Kurulum:
```
pip install opencv-python numpy
```

## Kullanım
```
python main.py --video /path/to/input.mp4
```

Örnek:
```
python main.py --video data/traffic.mp4 --scale 0.75 --show 1 --save outputs/out.mp4
```

## Önemli Argümanlar
- `--video` (zorunlu): Girdi `.mp4` dosyası yolu
- `--scale`: Video yeniden boyutlandırma oranı (varsayılan `1.0`)
- `--show`: Pencere gösterimi (1/0)
- `--debug`: Debug pencereleri (1/0)
- `--save`: Çıkış video yolu (opsiyonel)
- `--time_limit_sec`: İşlem süresi sınırı (sn)

### Şerit / Sayım Ayarları
- `--sample_secs`: Şerit tahmini için örnekleme süresi
- `--max_samples`: Örnekleme için maksimum kare sayısı
- `--line_y_ratio`: Sayım çizgisinin y oranı
- `--lane_band_h`: Sayım çizgisi etrafındaki bant yarı-yüksekliği
- `--lane_peaks_min_prom`, `--lane_peaks_min_dist`: Şerit sınırı tepe noktası ayarları

### Takip / Sayım Mantığı
- `--min_area`, `--max_area`: Araç aday kontur alanı sınırları
- `--max_missed`: Kayıp izlerin tutulma süresi (kare)
- `--match_dist`: Eşleştirme için maksimum merkez mesafesi
- `--side_margin_px`: Sayım bandı yarı-yüksekliği
- `--warmup_sec`: İlk N saniye sayımı kapat
- `--post_count_px_above`, `--post_min_age`, `--post_min_up_px`, `--lane_cooldown_frames`:
  Erken sayım ve parçalanma önleme ayarları

## Çıktı
- Ekranda her şerit için anlık sayım
- İşlem sonunda terminalde toplam sayım
- `--save` verilirse işlenmiş video kaydı

## Notlar
- Farklı video açıları için `--line_y_ratio`, `--lane_band_h`, `--lane_peaks_min_prom` gibi
  parametreleri ayarlamanız gerekebilir.
- Hızlı testlerde `--scale` ile çözünürlüğü düşürmek performansı artırır.