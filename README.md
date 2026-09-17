# Car Damage Detector

Mashinalardagi tashqi shikastlarni (dent — botiq, scratch — tirnalgan, damage — umumiy shikast) avtomatik aniqlaydigan obyekt aniqlash (object detection) modeli.

**Muhim izoh:** Ushbu loyihadagi YOLO-uslubidagi arxitektura tayyor kutubxonalardan (masalan, Ultralytics YOLOv5/v8) foydalanmasdan, PyTorch asosida noldan (from scratch) mustaqil ishlab chiqilgan. Backbone, anchor mexanizmi, detection head va loss funksiyasining barchasi custom kod sifatida yozilgan.

## Loyihaning maqsadi

Mashina korpusidagi tashqi shikastlarni rasm orqali avtomatik aniqlash va ularning joylashuvini (bounding box) belgilash. Loyiha sug'urta kompaniyalari, avtosalonlar yoki transport vositalarini baholash xizmatlari uchun amaliy ahamiyatga ega bo'lgan computer vision masalasini yechishga qaratilgan.

## Arxitektura

Model YOLO g'oyasidan ilhomlangan, ammo barcha komponentlari mustaqil ishlab chiqilgan.

### 1. Backbone

Rasmdan xususiyatlarni (feature) ajratib oluvchi qism quyidagi bloklardan tashkil topgan:

- `ConvBNAct` — Conv2d, BatchNorm va SiLU aktivatsiya funksiyasidan iborat asosiy qurilish bloki
- `DepthwiseSep` — hisoblash yukini kamaytirish maqsadida qo'llanilgan depthwise separable convolution qatlamlari

Bu bloklar ketma-ket joylashtirilib, rasmdan past darajadagi (chiziq, burchak) va yuqori darajadagi (shakl, obyekt) xususiyatlarni bosqichma-bosqich ajratib oladi.

### 2. Ko'p o'lchamli aniqlash (Multi-scale detection)

Model uchta turli grid darajasi orqali bashorat qiladi, bu turli o'lchamdagi shikastlarni bir xilda samarali aniqlash imkonini beradi:

| Daraja | Grid o'lchami | Stride | Vazifasi |
|--------|---------------|--------|----------|
| P3 | 80×80 | 8 | Kichik shikastlar |
| P4 | 40×40 | 16 | O'rta kattalikdagi shikastlar |
| P5 | 20×20 | 32 | Katta shikastlar |

### 3. Anchor mexanizmi

Har bir grid darajasi uchun oldindan hisoblangan anchor box o'lchamlari belgilangan (masalan, P3 darajasi uchun: (10,13), (16,30), (33,23)). Model ushbu anchorlarga nisbatan obyektning joylashuvi va o'lchamini bashorat qiladi.

### 4. IoU (Intersection over Union) hisoblash

Bashorat qilingan box bilan haqiqiy (ground truth) box orasidagi mosligini o'lchash uchun IoU funksiyasi mustaqil yozilgan. Bu funksiya training jarayonida qaysi anchor qaysi obyektga mos kelishini aniqlashda ishlatiladi.

### 5. Non-Maximum Suppression (NMS)

Bashorat paytida bitta obyekt uchun bir nechta overlapping box hosil bo'lishining oldini olish maqsadida NMS algoritmi qo'llanilgan (CONF_THRESH = 0.3, NMS_THRESH = 0.45).

## Aniqlanadigan klasslar

| ID | Klass | Tavsif |
|----|-------|--------|
| 0 | dent | Botiq, urilgan joy |
| 1 | scratch | Tirnalgan, chiziqli shikast |
| 2 | damage | Umumiy shikast |

## Dataset formati

Dataset YOLO formatida tayyorlangan. Label fayllari `class cx cy w h` formatida, normalizatsiya qilingan koordinatalar bilan ifodalangan:

```
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

## O'qitish jarayoni

1. **Data augmentation** — Albumentations kutubxonasi orqali rasmlarga turli o'zgartirishlar (aylantirish, kesish, rang o'zgarishi va h.k.) qo'llaniladi, bu modelning umumlashtirish qobiliyatini oshiradi
2. **Optimizer** — AdamW, o'rganish tezligi (learning rate) 1e-3
3. **Scheduler** — Cosine Annealing LR, epochlar davomida o'rganish tezligini silliq kamaytiradi
4. **Checkpoint tizimi** — eng past val loss ko'rsatkichiga ega model `best_model.pt` sifatida saqlanadi; zarur bo'lganda ushbu checkpoint'dan o'qitishni davom ettirish mumkin
5. Model 30 epoch davomida o'qitilgan, checkpoint asosida qo'shimcha 30 epoch bilan davom ettirilgan

## Natijalarni baholash

O'qitish yakunida train va validation loss grafigi chiziladi hamda `training_history.png` sifatida saqlanadi. Bu modelning o'qitish jarayonidagi dinamikasini vizual tahlil qilish imkonini beradi.

## Inference

Modelning `predict()` funksiyasi yangi rasmni qabul qiladi va undagi shikastlarni bounding box hamda klass nomi bilan belgilab chiqaradi.

## Talab qilinadigan kutubxonalar

```bash
pip install torch torchvision albumentations opencv-python-headless matplotlib tqdm
```

## Loyihani ishga tushirish

1. Notebook'ni Google Colab muhitida oching
2. Runtime → Change runtime type → GPU ni yoqing
3. Katakchalarni yuqoridan pastga ketma-ket ishga tushiring
4. So'ralganda `final_yolo_dataset.zip` faylini yuklang
5. Modelni o'qiting, checkpoint'larni saqlang va natijalarni tahlil qiling

## Texnologiyalar

- PyTorch — model arxitekturasi va training uchun asosiy freymvork
- Albumentations — data augmentation
- OpenCV — rasmlar bilan ishlash
- Matplotlib — natijalarni vizualizatsiya qilish
- tqdm — progress monitoring

## Muallif haqida

Ushbu loyihadagi model arxitekturasi — backbone, anchor tizimi, detection head va loss funksiyasi — tayyor YOLO kutubxonalaridan foydalanmasdan, PyTorch asosida mustaqil ishlab chiqilgan. Loyiha deep learning va computer vision sohasidagi obyekt aniqlash algoritmlarini chuqur o'zlashtirish maqsadida amalga oshirilgan.
