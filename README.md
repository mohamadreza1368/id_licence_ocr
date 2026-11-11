# سامانه فوق‌سبک پردازش و OCR کارت ملی ایرانی (Mobile & CPU Ready)

یک پایپ‌لاین کامل، سبک و صنعتی جهت استخراج خودکار تمام اطلاعات کارت ملی از روی **عکس‌های خام دوربین موبایل** (حتی با زاویه، تغییر فاصله و پس‌زمینه دست/میز)، بدون نیاز به مدل‌های سنگین VLM و بهینه‌سازی‌شده برای اجرا روی **موبایل (Android/iOS) و پردازنده‌های معمولی (CPU)**.

---

## 🏗 معماری دو مرحله‌ای (Two-Stage Pipeline)

حجم کل مدل‌ها در مجموع تنها **حدود ۷ مگابایت** بوده و زمان پردازش **زیر ۵۰ میلی‌ثانیه** است:

```
[ عکس خام دوربین موبایل (با پس‌زمینه میز، دست یا هر زاویه‌ای) ]
                                 │
                                 ▼
   ┌────────────────────────────────────────────────────────┐
   │  فاز اول: دیتکتور سبک فیلدها (YOLO-Nano ~3MB)          │
   │  - اسکن کل تصویر و مکان‌یابی کادر هر فیلد              │
   │    (کد ملی، تاریخ تولد، نام، نام خانوادگی، انقضا و...) │
   └────────────────────────────────────────────────────────┘
                                 │
                  برش خودکار هر کادر شناسایی‌شده
                                 │
                                 ▼
   ┌────────────────────────────────────────────────────────┐
   │  فاز دوم: تشخیص متن (MobileOCRNet ~4MB)                │
   │  - خواندن حروف و ارقام فارسی/انگلیسی هر کادر برش‌خورده │
   │  - مبتنی بر MobileNet Convolutions + BiLSTM + CTC Loss │
   └────────────────────────────────────────────────────────┘
                                 │
                                 ▼
   ┌────────────────────────────────────────────────────────┐
   │  فاز سوم: پس‌پردازش و اعتبارسنجی الگوریتم ۱۰ رقمی      │
   └────────────────────────────────────────────────────────┘
                                 │
                                 ▼
        خروجی ساخت‌یافته:
        {
          "NationalID": "0012345678",
          "BirthDate": "1369/07/03",
          "FirstName": "علی",
          "LastName": "محمدی"
        }
```

---

## 📁 فهرست فایل‌های پروژه

| فایل | وظیفه |
| :--- | :--- |
| [pipeline.py](file:///d:/projects/ocr_id/pipeline.py) | **خط‌لوله نهایی End-to-End:** دریافت عکس خام موبایل، دتکشن فیلدها، کراپ، OCR و اعتبارسنجی کد ملی |
| [prepare_detector_data.py](file:///d:/projects/ocr_id/prepare_detector_data.py) | تبدیل دیتای JSON و تصاویر به ساختار دتکشن YOLO (`detector_data/dataset.yaml`) |
| [train_detector.py](file:///d:/projects/ocr_id/train_detector.py) | آموزش دیتکتور سبک YOLO-Nano و ذخیره `checkpoints/best_field_detector.pt` و خروجی ONNX |
| [prepare_dataset.py](file:///d:/projects/ocr_id/prepare_dataset.py) | برش فیلدهای متنی و آماده‌سازی فایل‌های آموزش OCR (`data/train.txt` و `data/val.txt`) |
| [model.py](file:///d:/projects/ocr_id/model.py) | معماری سبک شبکه `MobileOCRNet` (~۴ مگابایت) |
| [train.py](file:///d:/projects/ocr_id/train.py) | حلقه آموزش OCR با CTCLoss، نوار پیشرفت زنده (`tqdm`) و اتصال به سرور MLflow |
| [tokenizer.py](file:///d:/projects/ocr_id/tokenizer.py) | انکودر/دیکودر الفبا و ارقام فارسی به روش حریصانه CTC |
| [dataset.py](file:///d:/projects/ocr_id/dataset.py) | دیتالودر بهینه با پدینگ پویا و حفظ نسبت تصویر |
| [export_model.py](file:///d:/projects/ocr_id/export_model.py) | اکسپورت مدل آموزش‌دیده OCR به TorchScript موبایل (`model_mobile.pt`) و ONNX |
| [infer.py](file:///d:/projects/ocr_id/infer.py) | استنتاج مستقل OCR و اعتبارسنجی کد ملی |
| [config.py](file:///d:/projects/ocr_id/config.py) | تنظیمات الفبا، ابعاد تصاویر، هایپرپارامترها و سرور MLflow |
| [run_nohup.sh](file:///d:/projects/ocr_id/run_nohup.sh) / [run_background.ps1](file:///d:/projects/ocr_id/run_background.ps1) | اسکریپت‌های اجرای پس‌زمینه در لینوکس و ویندوز |

---

## 🚀 مراحل آموزش و استفاده گام‌به‌گام

### فاز ۱: آموزش مدل YOLO (دتکتور فیلدها - کادرها کجا هستند؟)

در این فاز مدل یاد می‌گیرد روی هر عکس کامل کارت ملی، محل تک‌تک فیلدها را مکان‌یابی کند.

#### ۱. آماده‌سازی دیتای YOLO از روی لیبل‌های JSON:
```bash
python prepare_detector_data.py --data_dir "مسیر_پوشه_دیتای_شما"
```
* این اسکریپت برچسب‌های `updated_bbox` تمام فیلدها را خوانده، به مختصات نرمال‌سازی‌شده YOLO تبدیل می‌کند و پوشه `detector_data/` را می‌سازد.

#### ۲. آموزش دتکتور فوق‌سبک YOLO-Nano:
```bash
python train_detector.py --epochs 30 --batch_size 16
```
* خروجی مدل در مسیر `checkpoints/best_field_detector.pt` ذخیره شده و نسخه بهینه ONNX برای موبایل نیز ساخته می‌شود.

---

### فاز ۲: آموزش مدل OCR (تشخیص متن - داخل هر کادر چه نوشته شده؟)

در این فاز مدل یاد می‌گیرد ارقام و حروف فارسی/انگلیسی موجود در فیلدهای برش‌خورده را بخواند.

#### ۱. برش فیلدهای متنی برای آموزش OCR:
```bash
python prepare_dataset.py --data_dir "مسیر_پوشه_دیتای_شما"
```
* این اسکریپت تمام فیلدها را برش داده، داخل `data/crops/` ذخیره می‌کند و فایل‌های `data/train.txt` و `data/val.txt` را با لیبل متنی (`value`) ایجاد می‌نماید.

#### ۲. اجرای آموزش مدل OCR با لاگ در MLflow:
```bash
python train.py --epochs 30 --batch_size 32
```
* به صورت پیش‌فرض تمام لاگ‌ها، پارامترها و متریک‌ها به سرور MLflow ارسال می‌شود:
  * **آدرس سرور MLflow:** `http://172.16.101.42:8082`
  * **نام اکسپریمنت:** `iranian_id_card_ocr`
* **اجرای آفلاین بدون MLflow (اختیاری):**
  ```bash
  python train.py --no_mlflow
  ```
* خروجی بهترین وزن‌های OCR در `checkpoints/best_mobile_ocr.pth` ذخیره می‌شود.

#### ۳. خروجی مخصوص موبایل (Export):
```bash
python export_model.py
```
* خروجی `checkpoints/model_mobile.pt` (به حجم ۴ مگابایت) برای استفاده مستقیم در Android و iOS آماده می‌شود.

---

### فاز ۳: استخراج نهایی از عکس خام دوربین موبایل (Pipeline)

پس از آموزش دو مدل بالا، می‌توانید هر عکسی از دوربین موبایل را به پایپ‌لاین بدهید:

```bash
python pipeline.py --image "mobile_photo.jpg" --output "annotated_card.jpg"
```

**نمونه خروجی ترمینال:**
```text
[*] Processing mobile card photo: mobile_photo.jpg

=======================================================
  Extracted National ID Card Information
=======================================================
  NationalID      : 0012345678 [VALID CHECKSUM] (conf: 0.94)
  BirthDate       : 1369/07/03 (conf: 0.91)
  FirstName       : علی (conf: 0.88)
  LastName        : محمدی (conf: 0.89)
=======================================================
[+] Annotated visual image saved to: annotated_card.jpg
```

---

## ⚙ اجرای پس‌زمینه و بلندمدت (Background Execution)

#### در سرور لینوکس با `nohup`:
```bash
# اجرای دتکتور در پس‌زمینه:
nohup python -u train_detector.py --epochs 30 --batch_size 16 > train_detector.log 2>&1 &

# اجرای OCR در پس‌زمینه:
nohup python -u train.py --epochs 30 --batch_size 32 > train.log 2>&1 &

# مشاهده لحظه‌ای لاگ‌ها:
tail -f train.log
```

#### در محیط ویندوز با PowerShell:
```powershell
.\run_background.ps1

# مشاهده لاگ در ویندوز:
Get-Content train.log -Wait
```
