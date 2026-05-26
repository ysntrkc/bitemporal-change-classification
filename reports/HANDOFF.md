# reports/ handoff

Bu klasör BLM5135 dönem projesinin IEEE konferans formatındaki Türkçe
raporunun tüm kaynaklarını içerir. CoWork üzerinde düzenlemeye buradan
devam edebilirsin.

## Klasör haritası

```
reports/
├── main.tex                 Ana belge (XeLaTeX gerektirir)
├── refs.bib                 11 referans (ConvNeXt-V2, ASL, BIT, Q2L, UWL, ...)
├── IEEEtran.cls             IEEE conference class
├── HANDOFF.md               bu dosya
├── tikz/
│   └── architecture.tex     Fig. 2 -- TikZ mimari diyagramı (Aşama 1 | Aşama 2)
├── tables/
│   ├── main_results.tex            Tablo I -- Aşama 1 vs Aşama 2 ana sonuçlar (macro-F1)
│   ├── main_method_extended.tex    Tablo I-genişletilmiş -- yalnız Aşama 2 BIT-only:
│   │                               macro/micro F1 + macro/micro P + macro/micro R + mAP
│   └── ablation.tex                Tablo II -- 9 satırlık ablation tablosu
├── figs/
│   ├── teaser.png                       Fig. 1 -- success + failure örneği
│   ├── perclass_phase2_object.png       Fig. 3 -- canonical seçim
│   ├── perclass_phase2_event.png        (alternatif/ek)
│   ├── perclass_phase2_attribute.png    (alternatif/ek)
│   ├── perclass_phase1_*.png            (Aşama 1 muadilleri)
│   ├── curves_phase2.png                Fig. 4 -- canonical Aşama 2 eğitim eğrileri
│   ├── curves_phase1_*.png              (Aşama 1 per-family eğrileri)
│   └── qualitative.png                  Fig. 5 -- 2 başarı + 2 hata örneği
├── eda/
│   ├── {object,event,attribute}_hist.png   per-family sınıf frekansları
│   ├── split_dist.png                      train/val/test dağılımı + no-change oranı
│   └── positives_per_sample.png            ortalama pozitif sayısı
└── data/                    Raw markdown + JSON referansları (rapora gömme)
    ├── ablation_table.md                9-row tablo, ablation.tex kaynak verisi
    ├── phase1_table.md                  Aşama 1 mean ± std (macro/micro F1, P/R macro, mAP)
    ├── phase2_table.md                  Aşama 2 mean ± std (aynı 5-sütun, P1 ile delta)
    ├── main_method_extended_table.md    Yalnız Aşama 2 BIT-only:
    │                                    + precision_micro + recall_micro (rapor politikası)
    ├── per_class_metrics.{md,json}      sınıf bazlı F1/P/R/AP/support
    ├── changed_subset_table.md          is_change=1 alt-küme makro-F1
    └── eda_summary.json                 EDA sayıları (28% no-change, 270:1 vb.)
```

## Compile

XeLaTeX ya da LuaLaTeX gerekli (UTF-8 native + fontspec).

**Lokal (Linux + tectonic):**

```
cd reports
tectonic -X compile main.tex
```

DejaVu Serif/Sans/Mono fontları sistem font yollarından çekilir
(Ubuntu/Debian'da hazır gelir).

**Overleaf:** Project Settings → Compiler → XeLaTeX olarak ayarla.
`main.tex` içindeki font satırını TeX Gyre Termes ile değiştir:

```latex
\setmainfont{TeX Gyre Termes}
\setsansfont{TeX Gyre Heros}
\setmonofont{TeX Gyre Cursor}
```

Overleaf'te TeX Gyre fontları bundle ile gelir, DejaVu gelmez.

## main.tex bölüm durumu

| bölüm | durum | yapılacak |
|---|---|---|
| Abstract | dolu | sayıları gözden geçir (0.277, 0.282) |
| I. Giriş | %50 dolu (motivasyon paragrafı + katkı listesi var, sonuç paragrafı kısa) | bir paragraf daha: bitemporal change classification niye bu projenin alanı, neden bu 3 etiket ailesi |
| II. İlgili Çalışmalar | dolu, tek paragraf | yeterli |
| III. Yöntem | dolu (Ortak backbone + Aşama 1 + Aşama 2 + Eğitim reçetesi) | sayıları/parametreleri gözden geçir, opsiyonel "dataset" alt-bölümü Method veya Experiments'a |
| IV. Deneyler | dolu (kısa, 0.4 sayfa) | yeterli; istersen veri kümesi cümleleri buradan Yönteme alınabilir |
| V. Sonuçlar | iskelet + tablo include'lar | **en kritik bölüm**: alt-bölümlerin yorum paragraflarını yaz (BIT ne kattı, gate niye işe yaramadı, DBLoss kalibrasyon hikayesi, full-stack neden underperform) |
| VI. Sonuç | dolu (1 paragraf) | yeterli, gerekirse kısalt |
| References | otomatik, refs.bib'ten | yeterli |

`main.tex` içindeki `% TODO` ve `% Walk through...` yorumları nereye ne
yazılacağını gösterir.

## Doldurulması gereken alanlar

| satır | yer | iş |
|---|---|---|
| 47-50 | `\author{}` | "TODO\_SOYISIM" yerine soyadın + öğrenci no |
| 73-77 | Giriş ikinci paragraf | bitemporal change classification motivasyonu (1 paragraf) |
| ~125-140 | V.A "Ana sonuçlar" | "BIT küçük ortalama kazanım + büyük varyans düşüşü" yorum paragrafı (1-2 paragraf) |
| ~144-155 | V.B "Ablation analizi" | 6 satırlık ablation hikayesi (zaten yorum olarak yazılı; düz metne çevir) |
| şekil/tablo refs | gerektiği yerde `Tablo~\ref{tab:main}`, `Şekil~\ref{fig:arch}` gibi | atıfları metne göm |

## Anahtar sayılar (rapora gömmek için kopyala-yapıştır seti)

| metrik | değer | nereden |
|---|---|---|
| Toplam etiket ailesi | 3 (nesne, olay, öznitelik) | spec |
| Toplam sigmoid başlık | 48 (12 + 12 + 24) | spec |
| Nesne sınıf imbalance | 270:1 | EDA |
| Öznitelik sınıf imbalance | 67:1 | EDA |
| Olay sınıf imbalance | 28:1 | EDA |
| No-change oranı (her split) | ~28% | data/eda_summary.json |
| Train/val/test pair sayısı | bkz. data/eda_summary.json | -- |
| Backbone | ConvNeXt-V2 Tiny, FCMAE pre-trained | |
| Seed seti | 42, 1337, 2024 | |
| Aşama 1 macro-F1 ortalama | 0.277 ± 0.023 | data/ablation_table.md |
| Aşama 2 macro-F1 ortalama | 0.282 ± 0.029 | data/ablation_table.md |
| Aşama 2 std (no-BIT) | 0.013 (canonical 0.029'un yarısı) | -- |
| ResNet-50 vs ConvNeXt-V2 fark | +0.030 macro-F1 (A1 0.247 → P1 0.277) | -- |
| TTA katkısı | +0.006 (no-TTA 0.271 → TTA 0.277) | -- |
| no-change gate etkisi | -0.002 (null result) | -- |
| DBLoss kalibrasyonsuz | 0.266 (object) | -- |
| DBLoss tuned thr | 0.285 (object, ASL'nin 0.303'ünü geçmiyor) | -- |
| Full stack (Q2L+UWL) | 0.181 ± 0.042 (canonical'in altında) | -- |

## Şekil seçimi (5 limit)

| # | LaTeX label | dosya | yorum |
|---|---|---|---|
| Fig 1 | `fig:teaser` | figs/teaser.png | 1 success + 1 failure |
| Fig 2 | `fig:arch` | tikz/architecture.tex | TikZ -- 2-sütun figure* |
| Fig 3 | `fig:perclass` | figs/perclass_phase2_object.png | kanonik seçim (nesne ailesi, uzun-kuyruk göstergesi) |
| Fig 4 | `fig:curves` | figs/curves_phase2.png | Aşama 2 canonical eğrileri |
| Fig 5 | `fig:qual` | figs/qualitative.png | 4-satır success/failure detay |

Alternatif: Fig 3'te Aşama 1 muadili (figs/perclass_phase1_*.png) ya da
Fig 4'te Aşama 1 eğrileri (figs/curves_phase1_*.png) tercih edilebilir.

## Raporlama politikası: metrik kapsamı

- **Ana metrik: macro-F1** (uzun-kuyruklu multi-label için doğru ana
  sayı). Tablo~I (`main_results.tex`) ve Tablo~II (`ablation.tex`)
  yalnız macro-F1 raporlar.
- **Ek metrikler (sadece ana yöntem için)**: Aşama~2 BIT-only canonical
  konfigürasyonu için `main_method_extended.tex` ek tablosu macro-F1,
  micro-F1, P (macro), P (micro), R (macro), R (micro), mAP'i yan yana
  verir. Bu uzun set ablation satırlarında raporlanmaz; tablo kalabalığı
  yaratmamak ve macro-F1'in tek ana sayı olarak kalmasını korumak için.
- `phase1_table.md` ve `phase2_table.md` aile ve ortalama bazında 5
  sütun içerir (macro-F1, micro-F1, P macro, R macro, mAP) -- bunlar
  rapor tabloları için kaynak verisidir; rapora doğrudan girmez.

## Önemli noktalar / dikkat

- **Sigmoid yalnız** (softmax YOK -- hocanın hard rule'u). Tüm 48 başlık
  bağımsız sigmoid.
- **Train/val/test split asla değiştirilmedi**.
- **3 seed mean ± std** her ana satırda raporlanmış; tek-seed varyantlar
  kaldırıldı.
- "q2l_uwl" satırı epoch 50 cap'inde durmuş (hâlâ yükseliyordu).
  Bunu Sonuçlar bölümünde "early-stop cap notu" olarak işleyebilirsin.
- Sayısal aralığın küçüklüğü (0.18-0.31 macro-F1) doğru -- bu veri
  kümesinde state-of-the-art bu civarda; ölçek sınırı net.

## Terim politikası

Türkçe gövde, İngilizce teknik terimler. Kullanılan ayrım:

- **Türkçe**: bölüm başlıkları, şekil/tablo başlıkları, "Aşama 1/2",
  etiket ailesi, doğrulama, ortalama, standart sapma, açıklayıcı gövde
- **İngilizce**: backbone, Siamese encoder, four-way fusion, BIT, Q2L,
  UWL, ASL, DBLoss, FixedWeightLoss, no-change gate, TTA, seed,
  threshold tuning, mean, learning rate, weight decay, mixed precision,
  augmentation, CutMix, EMA, gradient clipping, ablation, drop-in,
  full-stack, macro/micro-F1, mAP

Yeni teknik terimler eklersen aynı çizgide tut.

## ZIP teslimi

Spec'e göre teslim dizini adı: `OgrenciNo_Yasin_Soyisim/`. Bu klasör
zip'lenir ve Google Classroom'a yüklenir. ZIP içinde olması gerekenler:

```
<OgrNo>_Yasin_<Soyisim>/
├── train_phase1.py, train_phase2.py, eval_phase1.py, eval_phase2.py
├── src/                              (proje kodu)
├── configs/                          (10 YAML + label_vocab.json)
├── ReadMe.txt                        (Türkçe, üst dizinde değil burada)
├── requirements.txt
├── <OgrNo>_Yasin_<Soyisim>.pdf       (= reports/main.pdf renamed)
└── results/
    └── phase{1,2}_*/metrics_test_*.json   (yalnız metric JSON'lar)
```

**ZIP'e koyma**: dataset/, checkpoints (*.pth), tensorboard, _archive/,
docs/, runs/.
