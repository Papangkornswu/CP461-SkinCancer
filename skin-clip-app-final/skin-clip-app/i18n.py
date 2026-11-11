"""UI strings (English / Thai). Stored data always uses language-independent codes (nv, mel, ...)."""

S = {
 "hero_sub": ("Fine-tuned CLIP for skin-lesion triage, with physician approval that feeds continuous model improvement.",
              "CLIP ที่ fine-tune แล้วสำหรับคัดกรองรอยโรคผิวหนัง พร้อมระบบให้แพทย์ตรวจสอบเพื่อนำไปพัฒนาโมเดลอย่างต่อเนื่อง"),
 "disclaimer": ("Decision-support prototype for qualified clinicians. It is <b>not</b> a medical device and must not be "
                "the sole basis for diagnosis or treatment.",
                "ต้นแบบเครื่องมือช่วยตัดสินใจสำหรับแพทย์ผู้เชี่ยวชาญ <b>ไม่ใช่</b>เครื่องมือแพทย์ "
                "และห้ามใช้เป็นเกณฑ์เดียวในการวินิจฉัยหรือรักษา"),
 "tab_analyze": ("Analyze & review", "วิเคราะห์และตรวจสอบ"),
 "tab_feedback": ("Feedback analytics", "สถิติ Feedback"),
 "tab_card": ("Model card", "ข้อมูลโมเดล"),
 "lang_label": ("Language", "ภาษา"),
 "theme_label": ("Theme", "ธีม"),
 "theme_light": ("☀️ Light", "☀️ สว่าง"),
 "theme_dark": ("🌙 Dark", "🌙 มืด"),
 # login
 "login_only": ("Authorised clinicians only.", "สำหรับแพทย์ที่ได้รับอนุญาตเท่านั้น"),
 "passcode": ("Access passcode", "รหัสผ่านเข้าใช้งาน"),
 "sign_in": ("Sign in", "เข้าสู่ระบบ"),
 "wrong_pass": ("Incorrect passcode.", "รหัสผ่านไม่ถูกต้อง"),
 "open_warn": ("REVIEWER_PASSCODE is not set: the app is open to anyone who has the URL. Set it before deploying.",
               "ยังไม่ได้ตั้งค่า REVIEWER_PASSCODE: ใครมี URL ก็เข้าใช้แอปได้ โปรดตั้งค่าก่อนนำไปใช้งานจริง"),
 # step 1
 "step1": ("1 · Upload lesion image", "1 · อัปโหลดภาพรอยโรค"),
 "uploader": ("Dermoscopic image (JPG/PNG)", "ภาพ dermoscopy (JPG/PNG)"),
 "deid_caption": ("Use de-identified images only: no names, IDs or visible patient information.",
                  "ใช้เฉพาะภาพที่ลบข้อมูลระบุตัวตนแล้ว: ไม่มีชื่อ รหัส หรือข้อมูลผู้ป่วยที่มองเห็นได้"),
 "deid_check": ("I confirm this image is de-identified and I am authorised to use it.",
                "ฉันยืนยันว่าภาพนี้ไม่มีข้อมูลระบุตัวตน และฉันได้รับอนุญาตให้ใช้ภาพนี้"),
 # step 2
 "step2": ("2 · Model prediction", "2 · ผลทำนายของโมเดล"),
 "upload_prompt": ("Upload an image to see the model's assessment.", "อัปโหลดภาพเพื่อดูผลประเมินของโมเดล"),
 "confirm_prompt": ("Please confirm the de-identification statement to run the model.",
                    "โปรดยืนยันว่าภาพนี้ไม่มีข้อมูลระบุตัวตนก่อนรันโมเดล"),
 "checking": ("Checking image...", "กำลังตรวจสอบภาพ..."),
 "analysing": ("Analysing...", "กำลังวิเคราะห์..."),
 "gate_label": ("Image check", "ตรวจสอบภาพ"),
 "gate_title": ("This does not look like a skin-lesion image", "ภาพนี้ดูไม่เหมือนภาพรอยโรคผิวหนัง"),
 "gate_body": ("Skin-image score {s}% (minimum {m}%). No diagnosis is shown, because the classifier can only choose "
               "among 7 skin diseases and would still return one of them for any picture. Upload a dermoscopic or "
               "close-up lesion photo.",
               "คะแนนความเป็นภาพผิวหนัง {s}% (ขั้นต่ำ {m}%) จึงไม่แสดงผลวินิจฉัย เพราะโมเดลเลือกได้เฉพาะจาก 7 โรคผิวหนัง "
               "และจะตอบเป็นหนึ่งในนั้นเสมอไม่ว่าภาพจะเป็นอะไร โปรดอัปโหลดภาพ dermoscopy หรือภาพถ่ายใกล้ของรอยโรค"),
 "gate_btn": ("Analyze anyway (I am sure this is a skin-lesion image)",
              "วิเคราะห์ต่อ (ฉันมั่นใจว่าเป็นภาพรอยโรคผิวหนัง)"),
 "gate_override": ("Shown because you overrode the image check: treat this output with extra caution.",
                   "แสดงผลเพราะคุณข้ามการตรวจสอบภาพ โปรดใช้ผลนี้ด้วยความระมัดระวังเป็นพิเศษ"),
 "card_top": ("Most likely diagnosis", "การวินิจฉัยที่เป็นไปได้มากที่สุด"),
 "card_adj": ("Physician-adjusted diagnosis", "ผลวินิจฉัยที่แพทย์ปรับแก้"),
 "card_meta": ("confidence <b>{c}%</b> · margin to 2nd {m} pts · model <code>{v}</code>",
               "ความมั่นใจ <b>{c}%</b> · ห่างจากอันดับ 2 อยู่ {m} จุด · โมเดล <code>{v}</code>"),
 "card_meta_adj": ("Model predicted <b>{pred}</b> ({c}%) and gave this diagnosis <b>{p}%</b> · model <code>{v}</code>",
                   "โมเดลทำนายว่า <b>{pred}</b> ({c}%) และให้โรคนี้ <b>{p}%</b> · โมเดล <code>{v}</code>"),
 "demo": ("Demo mode: no fine-tuned checkpoint found in model/, showing the zero-shot baseline.",
          "โหมดสาธิต: ไม่พบโมเดลที่ fine-tune ใน model/ จึงแสดงผลจาก zero-shot baseline"),
 "low_conf": ("Low model confidence: please review this case with particular care.",
              "โมเดลมั่นใจต่ำ: โปรดตรวจเคสนี้อย่างละเอียดเป็นพิเศษ"),
 "prob_axis": ("Probability (%)", "ความน่าจะเป็น (%)"),
 "chart_legend": ("Teal = model's top prediction · Amber = physician's label",
                  "สีเขียวอมฟ้า = ผลทำนายสูงสุดของโมเดล · สีเหลืองอำพัน = ป้ายของแพทย์"),
 # step 3
 "step3": ("3 · Physician review", "3 · แพทย์ตรวจสอบ"),
 "reviewer": ("Reviewer ID / initials", "รหัส / อักษรย่อผู้ตรวจสอบ"),
 "reviewer_ph": ("e.g. Dr. A.B.", "เช่น นพ. ก.ข."),
 "q_correct": ("Is the model's prediction correct?", "ผลทำนายของโมเดลถูกต้องหรือไม่?"),
 "opt_approve": ("Approve prediction", "อนุมัติผลทำนาย"),
 "opt_correct": ("Correct diagnosis", "แก้ไขผลวินิจฉัย"),
 "opt_unusable": ("Image unusable", "ภาพใช้ไม่ได้"),
 "pick_label": ("Select the correct diagnosis", "เลือกผลวินิจฉัยที่ถูกต้อง"),
 "pick_ph": ("Choose a diagnosis", "เลือกผลวินิจฉัย"),
 "err_label": ("Please select the correct diagnosis.", "โปรดเลือกผลวินิจฉัยที่ถูกต้อง"),
 "live_hint": ("The result above updates immediately when you choose a diagnosis; it is saved when you press Submit.",
               "ผลด้านบนจะอัปเดตทันทีเมื่อคุณเลือกโรค และจะถูกบันทึกเมื่อกดปุ่มบันทึก"),
 "basis": ("Basis of your judgement", "เกณฑ์ที่ใช้ตัดสิน"),
 "certainty": ("Your certainty", "ความมั่นใจของคุณ"),
 "site": ("Lesion site (optional)", "ตำแหน่งรอยโรค (ไม่บังคับ)"),
 "notes": ("Notes (optional, no patient identifiers)", "หมายเหตุ (ไม่บังคับ ห้ามใส่ข้อมูลระบุตัวตนผู้ป่วย)"),
 "submit": ("Submit review", "บันทึกการตรวจสอบ"),
 "err_reviewer": ("Please enter your reviewer ID.", "โปรดกรอกรหัส/อักษรย่อผู้ตรวจสอบ"),
 "saved_ok": ("Review saved. Thank you: it will be used in the next model update.",
              "บันทึกการตรวจสอบแล้ว ขอบคุณ ข้อมูลนี้จะถูกนำไปใช้ในการอัปเดตโมเดลรอบถัดไป"),
 "saved_unusable": ("Image marked as unusable (excluded from training).",
                    "ทำเครื่องหมายว่าภาพใช้ไม่ได้แล้ว (จะไม่ถูกนำไปเทรน)"),
 "change_label": ("Change my label", "แก้ไขป้ายของฉัน"),
 "another": ("Review another image", "ตรวจภาพถัดไป"),
 # basis / certainty / sites (display only)
 "Clinical impression": ("Clinical impression", "การวินิจฉัยทางคลินิก"),
 "Dermoscopic review": ("Dermoscopic review", "การตรวจด้วย dermoscopy"),
 "Histopathology-confirmed": ("Histopathology-confirmed", "ยืนยันด้วยผลชิ้นเนื้อ"),
 "Low": ("Low", "ต่ำ"), "Medium": ("Medium", "ปานกลาง"), "High": ("High", "สูง"),
 "Head/neck": ("Head/neck", "ศีรษะ/คอ"), "Trunk": ("Trunk", "ลำตัว"), "Upper limb": ("Upper limb", "แขน"),
 "Lower limb": ("Lower limb", "ขา"), "Palms/soles": ("Palms/soles", "ฝ่ามือ/ฝ่าเท้า"),
 "Genital": ("Genital", "อวัยวะเพศ"), "Other/unknown": ("Other/unknown", "อื่น ๆ/ไม่ทราบ"),
 "Malignant": ("Malignant", "มะเร็ง"), "Pre-malignant": ("Pre-malignant", "ก่อนเป็นมะเร็ง"),
 "Benign": ("Benign", "ไม่ใช่มะเร็ง"),
 "choose": ("Choose an option", "เลือกตัวเลือก"),
 # analytics
 "an_title": ("Feedback analytics", "สถิติ Feedback จากแพทย์"),
 "an_none": ("No physician reviews yet. Reviews submitted on the first tab appear here.",
             "ยังไม่มีรีวิวจากแพทย์ รีวิวที่บันทึกในแท็บแรกจะแสดงที่นี่"),
 "m_reviews": ("Reviews", "จำนวนรีวิว"),
 "m_agree": ("Physician agreement", "แพทย์เห็นด้วยกับโมเดล"),
 "m_corr": ("Corrections", "จำนวนที่แก้ไข"),
 "m_ready": ("Ready for retraining", "พร้อมใช้เทรนซ้ำ"),
 "progress": ("Progress toward the next retraining batch", "ความคืบหน้าสู่รอบเทรนซ้ำถัดไป"),
 "ch_agree": ("Agreement by model prediction", "อัตราเห็นด้วย แยกตามผลทำนายของโมเดล"),
 "ch_agree_y": ("% approved", "% ที่อนุมัติ"),
 "ch_conf": ("Model vs. physician (final label)", "โมเดล เทียบกับ แพทย์ (ป้ายสุดท้าย)"),
 "ax_phys": ("Physician", "แพทย์"), "ax_model": ("Model", "โมเดล"),
 "recent": ("Recent reviews", "รีวิวล่าสุด"),
 "dl_csv": ("Download reviews (CSV)", "ดาวน์โหลดรีวิว (CSV)"),
 "dl_zip": ("Download training bundle (ZIP: CSV + images)", "ดาวน์โหลดชุดข้อมูลเทรน (ZIP: CSV + ภาพ)"),
 "retrain_cap": ("Retrain with: `python retrain.py --data-dir data --ckpt model/clip_skin_ft.pt`. The candidate model "
                 "is only promoted if it does not regress (see README).",
                 "เทรนซ้ำด้วย: `python retrain.py --data-dir data --ckpt model/clip_skin_ft.pt` "
                 "โมเดลใหม่จะถูกนำมาใช้ก็ต่อเมื่อไม่แย่ลง (ดู README)"),
 "col_time": ("time", "เวลา"), "col_reviewer": ("reviewer", "ผู้ตรวจ"), "col_model": ("model", "โมเดลทำนาย"),
 "col_conf": ("conf", "มั่นใจ"), "col_decision": ("decision", "การตัดสินใจ"), "col_final": ("final", "ป้ายสุดท้าย"),
 "col_basis": ("basis", "เกณฑ์"), "col_cert": ("certainty", "ความมั่นใจแพทย์"), "col_ver": ("version", "เวอร์ชัน"),
 "approved": ("approved", "อนุมัติ"), "corrected": ("corrected", "แก้ไข"), "unusable": ("unusable", "ใช้ไม่ได้"),
 # model card
 "mc_title": ("Model card", "ข้อมูลโมเดล"),
 "mc_head": ("**Version:** `{v}` · **Backbone:** `{b}` · **Classes:** {c}",
             "**เวอร์ชัน:** `{v}` · **Backbone:** `{b}` · **คลาส:** {c}"),
 "mc_metrics": ("**Held-out test performance** (lesion-level group split, HAM10000)",
                "**ประสิทธิภาพบนชุดทดสอบ** (แบ่งตาม lesion, HAM10000)"),
 "mc_beats": ("Fine-tuned model beats the baseline CLIP on test accuracy: **{x}**",
              "โมเดลที่ fine-tune ชนะ CLIP baseline บน test accuracy: **{x}**"),
 "mc_nometrics": ("Metrics appear here after you copy `metadata.json` from the training notebook into `model/`.",
                  "ตัวเลขประสิทธิภาพจะแสดงที่นี่หลังคัดลอก `metadata.json` จากโน้ตบุ๊กมาไว้ใน `model/`"),
 "mc_body": ("""
**Intended use.** Decision support for qualified physicians reviewing dermoscopic images. Not a diagnostic device.

**Known limitations.**
- Trained on HAM10000 (dermoscopy, mostly lighter skin tones, two clinics). Expect lower accuracy on other devices, skin tones and clinical (non-dermoscopic) photos.
- The classifier only knows 7 diseases; other conditions will still be assigned to one of them.
- Heavily class-imbalanced source data; rare classes (dermatofibroma, vascular) have few examples.
- Labels from physician feedback inherit reviewer bias; histopathology-confirmed labels are weighted highest.
- Confidence is calibrated on a validation split and can drift on new data: monitor the analytics tab.
""", """
**วัตถุประสงค์การใช้งาน** ช่วยสนับสนุนการตัดสินใจของแพทย์ผู้ตรวจภาพ dermoscopy ไม่ใช่เครื่องมือวินิจฉัย

**ข้อจำกัดที่ทราบ**
- เทรนจาก HAM10000 (ภาพ dermoscopy ส่วนใหญ่เป็นผิวสีอ่อน จาก 2 สถาบัน) ความแม่นยำอาจลดลงกับอุปกรณ์ สีผิว และภาพถ่ายทั่วไป (ไม่ใช่ dermoscopy) ที่ต่างออกไป
- โมเดลรู้จักเพียง 7 โรค โรคอื่นจะถูกจัดเข้าหนึ่งในเจ็ดโรคนี้เสมอ
- ข้อมูลต้นทางไม่สมดุลมาก คลาสที่พบน้อย (dermatofibroma, vascular) มีตัวอย่างน้อย
- ป้ายจาก feedback ของแพทย์ย่อมมีอคติของผู้ตรวจ ป้ายที่ยืนยันด้วยชิ้นเนื้อจึงถูกให้น้ำหนักสูงสุด
- ความมั่นใจถูกปรับด้วย validation set และอาจคลาดเคลื่อนกับข้อมูลใหม่ โปรดติดตามในแท็บสถิติ
"""),
}

CLASS_TH = {
    "nv": "ไฝธรรมดา", "mel": "เมลาโนมา", "bkl": "เนื้องอกคล้ายเคราโทซิสชนิดไม่ร้ายแรง",
    "bcc": "มะเร็งผิวหนังเบซัลเซลล์", "akiec": "รอยโรคก่อนมะเร็ง (แอคทินิกเคราโทซิส)",
    "vasc": "รอยโรคหลอดเลือด", "df": "เดอร์มาโทไฟโบรมา (ตุ่มพังผืดใต้ผิวหนัง)",
}


def tr(lang: str, key: str, **kw) -> str:
    en, th = S[key]
    s = th if lang == "th" else en
    return s.format(**kw) if kw else s
