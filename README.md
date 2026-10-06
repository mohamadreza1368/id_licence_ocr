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
پوشه حاوی تصاویر و فایل‌های JSON دیتای خود را مشخص کنید و دستور زیر را اجرا کنید:
```bash
python prepare_dataset.py --data_dir "D:/path/to/your/dataset" --val_split 0.15
```
این اسکریپت تمام فیلدها (`BirthDate`، کد ملی، نام و...) را بر اساس مختصات `updated_bbox` برش زده و در `data/crops/` ذخیره می‌کند.

### گام دوم: اجرای آموزش (Training)
برای شروع آموزش مدل:
```bash
python train.py
```
مدل روی CPU یا GPU شروع به یادگیری فونت‌ها و ارقام کارت ملی می‌کند و بهترین وزن‌ها در `checkpoints/best_mobile_ocr.pth` ذخیره می‌شود.

### گام سوم: خروجی برای موبایل و دستگاه‌های سبک
برای تبدیل مدل آموزش‌دیده به فرمت بهینه موبایل:
```bash
python export_model.py
```
خروجی `model_mobile.pt` (حجم ~ ۴ مگابایت) برای استفاده در اپلیکیشن‌های اندروید و iOS با سرعت آنی آماده است.

### گام چهارم: تست و استخراج اطلاعات
برای تست خواندن اطلاعات یک کارت جدید:
```bash
python infer.py --image "path/to/card.png" --json "path/to/card.json"
```
یا برای تست روی تک‌تصویر کراپ‌شده یک فیلد:
```bash
python infer.py --image "data/crops/sample.png"
```
