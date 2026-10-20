# Iranian National ID Card Lightweight OCR (Mobile & CPU Ready)

یک پایپ‌لاین سبک، کامل و صنعتی برای استخراج اطلاعات کارت ملی ایرانی با مدل فوق‌سبک (~۴ مگابایت) بدون نیاز به VLM و قابل اجرا روی موبایل (Android/iOS) و پردازنده‌های معمولی (CPU).

---

## 📁 ساختار فایل‌های پروژه

- [config.py](file:///d:/projects/ocr_id/config.py): تنظیمات ابعاد تصویر، نرخ یادگیری، دسته‌بندی کاراکترها و الفبای فارسی/اعداد
- [prepare_dataset.py](file:///d:/projects/ocr_id/prepare_dataset.py): استخراج و کراپ خودکار فیلدهای متنی کارت ملی از فایل‌های JSON و تصاویر و ساخت فایل‌های `train.txt` و `val.txt`
- [model.py](file:///d:/projects/ocr_id/model.py): معماری سبک `MobileOCRNet` مبتنی بر Depthwise-Separable Convolutions و BiLSTM + CTC
- [tokenizer.py](file:///d:/projects/ocr_id/tokenizer.py): تبدیل کاراکترها به ایندکس و دیکدینگ حریصانه (Greedy CTC Decoding)
- [dataset.py](file:///d:/projects/ocr_id/dataset.py): لودر دیتا، تغییر اندازه پویا با حفظ نسبت تصویر (Aspect Ratio) و پدینگ متغیر
- [train.py](file:///d:/projects/ocr_id/train.py): چرخه آموزش مدل با CTCLoss، ارزیابی دقت کاراکتر و کلمه و ذخیره بهترین مدل
- [export_model.py](file:///d:/projects/ocr_id/export_model.py): خروجی گرفتن برای موبایل در قالب TorchScript (`model_mobile.pt`) و ONNX (`model.onnx`)
- [infer.py](file:///d:/projects/ocr_id/infer.py): استنتاج و خواندن اطلاعات کارت ملی همراه با اعتبارسنجی الگوریتم ۱۰ رقمی کد ملی

---

## 🚀 مراحل آموزش و استفاده گام‌به‌گام

### گام اول: آماده‌سازی دیتا و برش فیلدها
دیتای شما که دارای دو پوشه `labels` و `images` است را به راحتی با یکی از دستورات زیر اجرا کنید:

**حالت ۱ (ساده‌ترین - دادن پوشه اصلی که شامل دو فولدر labels و images است):**
```bash
python prepare_dataset.py --data_dir "D:/path/to/dataset"
```

**حالت ۲ (مشخص کردن مستقیم هر دو پوشه):**
```bash
python prepare_dataset.py --labels_dir "D:/path/to/labels" --images_dir "D:/path/to/images"
```
این اسکریپت تمام عکس‌ها را ایندکس کرده، فایل‌های JSON مربوطه را تطبیق می‌دهد و فیلدها (`BirthDate`، کد ملی، نام و...) را بر اساس مختصات `updated_bbox` برش زده و در `data/crops/` ذخیره می‌کند.

### گام دوم: اجرای آموزش (Training) به همراه لاگ در MLflow
برای شروع آموزش مدل و ثبت تمام متریک‌ها و پارامترها در سرور MLflow:
```bash
python train.py
```
به صورت پیش‌فرض، آدرس سرور `http://172.16.101.42:8082` و نام اکسپریمنت `iranian_id_card_ocr` است.

**تنظیمات پیشرفته اختیاری:**
```bash
# تغییر آدرس سرور MLflow، تعداد Epochs یا نام Run
python train.py --mlflow_uri "http://172.16.101.42:8082" --experiment_name "my_experiment" --run_name "run_v1" --epochs 40 --batch_size 32

# غیرفعال کردن موقت لاگ در MLflow (اجرای آفلاین)
python train.py --no_mlflow
```

**اجرا در پس‌زمینه با nohup (مخصوص سرور لینوکس یا اجرای بلندمدت):**
برای اینکه با بستن ترمینال یا قطع اتصال SSH فرایند آموزش متوقف نشود:
```bash
# روش مستقیم با دستور nohup (حالت unbuffered برای نوشتن فوری لاگ):
nohup python -u train.py > train.log 2>&1 &

# مشاهده لحظه‌ای خروجی و لاگ‌ها:
tail -f train.log

# یا اجرای اسکریپت آماده:
bash run_nohup.sh
```

**اجرا در پس‌زمینه در ویندوز (PowerShell):**
```powershell
.\run_background.ps1

# مشاهده لاگ در ویندوز:
Get-Content train.log -Wait
```

**موارد ثبت شده در MLflow:**
* **پارامترها (Parameters):** ابعاد تصویر، نرخ یادگیری، اندازه بچ، تعداد پارامترهای مدل، حجم مدل، تنظیمات CTC
* **متریک‌ها در هر Epoch (Metrics):** مقدار `train_loss`، `val_loss`، دقت ترتیبی کلمه `val_seq_acc`، دقت کاراکتر `val_char_acc`، زمان هر Epoch و مقدار Learning Rate
* **آرتیفکت‌ها و خروجی‌ها (Artifacts):** وزن‌های بهترین مدل (`best_mobile_ocr.pth`)، مدل خروجی موبایل (`model_mobile.pt`)، فایل دیکشنری حروف (`vocab.txt`) و گزارش نمونه خروجی‌های مدل (`sample_predictions.txt`)



### گام سوم: خروجی برای موبایل و دستگاه‌های سبک
برای تبدیل مدل آموزش‌دیده به فرمت بهینه موبایل:
```bash
python export_model.py
```
خروجی `model_mobile.pt` (حجم ~ ۴ مگابایت) برای استفاده در اپلیکیشن‌های اندروید و iOS با سرعت آنی آماده است.

### گام چهارم: استخراج خودکار از کل عکس دوربین موبایل (پایپ‌لاین کامل End-to-End)
برای اینکه کاربر بتواند یک عکس کامل از دوربین موبایل بدهد (بدون نیاز به دانستن محل کادرها یا کراپ قبلی):

1. **تبدیل دیتای فعلی شما به فرمت دیتکتور:**
   ```bash
   python prepare_detector_data.py --data_dir "مسیر_دایرکتوری_دیتای_شما"
   ```
   این اسکریپت برچسب‌های `updated_bbox` تمام فیلدها را به فرمت استاندارد دیتکشن تبدیل کرده و فایل `detector_data/dataset.yaml` را می‌سازد.

2. **آموزش دیتکتور فوق‌سبک فیلدها (YOLO-Nano ~3MB):**
   ```bash
   python train_detector.py --epochs 30 --batch_size 16
   ```

3. **اجرای استخراج آنی روی هر عکس دوربین موبایل:**
   ```bash
   python pipeline.py --image "mobile_photo.jpg" --output "annotated_card.jpg"
   ```
   این دستور به طور خودکار فیلدهای کارت (`BirthDate`, `NationalID`, `FirstName` و...) را از روی کل عکس پیدا کرده، برش می‌زند، متن آن‌ها را می‌خواند و ساختار JSON نهایی را خروجی می‌دهد.

