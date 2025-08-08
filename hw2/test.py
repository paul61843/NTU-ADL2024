# -*- coding: utf-8 -*-

from transformers import MT5Tokenizer, MT5ForConditionalGeneration
import torch

# ---------- 設定模型路徑（訓練儲存過的） ----------
model_path = "./output_model"  # ✅ 跟你儲存的資料夾一致

# ---------- 載入模型與 tokenizer ----------
tokenizer = MT5Tokenizer.from_pretrained(model_path)
model = MT5ForConditionalGeneration.from_pretrained(model_path)

# ---------- 移動到 GPU / MPS / CPU ----------
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# ---------- 測試輸入 ----------
# 根據你的任務可能是翻譯、摘要、分類等
# 通常格式： "summarize: ..." / "translate English to French: ..." / "question: ... context: ..."
input_text = 'summarize: 「2020新社花海活動」活動由行政院農業委員會種苗改良繁殖場與台中市政府共同主辦，訂於11月14日至12月6日(共計23日) ，每日8時至17時，在新社區協興街30號，即行政院農業委員會種苗改良繁殖場第二農場旁田間，結合「2020台中國際花毯節」、「農特產品展售」等活動盛大辦理（ https://flowersea.tw/ )，為維護會場周邊道路交通秩序，中市警局配合主辦單位及交通局規劃實施交通疏導管制措施。\n中市警局交通管制規劃：行人徒步區（禁止車輛通行）。協興街路段（由苗圃橋至花海接駁車站車輛出入口全段範圍），管制時段為活動期間平假日8時至18時；興義街路段（花海區域內），管制時段為活動期間平假日8時至18時。\n行車管制、車種管制：單行道管制：興中街（北往南，東山街往協中街方向）單行道管制，管制時段為活動期間假日8時至20時；協興街（西北往東南，興中街往苗圃橋方向）單行道管制並禁止大型車輛通行，管制時段為活動期間平假日8時至20時；興義街（北往南，花海停車場經三嵙口至協中街路段）單行道管制，管制時段為活動期間平假日8時至20時；滯洪池旁道路（東南往西北，由苗圃橋至興中街路段）單行道管制，管制時段為活動期間假日8時至20時。\n接駁車專用道單行管制：興中街（北往南，東山街往華豐街方向），路段中央標線以交通錐區隔，分為私人車輛與接駁專用之車道，管制時段為活動期間假日8時至20時。\n華豐街（興中街至協興街），專供花海接駁車輛行駛，不開放予一般小型車通行，並由東勢分局視實際交通情形權宜管制措施，管制時段為活動期間平假日8時至20時。\n車種限制：東山街（東山路往興中街）由東勢分局視實際交通情形機動調整僅開放接駁車、大客車通行，管制時段為活動期間假日8時至20時。\n接駁車規劃：假日接駁車路線4條：太原線：太原停車場→花海會場。豐原線：豐原火車站→花海會場。東勢線：東勢河濱公園停車場→花海會場。\n櫻木花道線：花海會場→復盛公園→新社區公所→新社高中→花海會場。\n平日接駁車路線2條，分別為太原線、豐原線。歡迎民眾多加利用接駁車，臺中市區民眾可將車輛停於太原停車場，再搭乘太原線接駁車至會場，避免自行開車或騎車前往活動地點，造成周邊交通壅塞。\n停車場規劃：展區主要停車場：P1臨時停車場：供小型汽車、機車停放。\nP2臨時停車場：供小型汽車停放。P3臨時停車場：供小型汽車停放。九渠溝景觀滯洪池停車場：供身障或有親子停車證車輛停放。華豐街路側：供遊覽車停放。\n周圍臨時停車場：新社高中、新社區公所、復盛公園、中興嶺營區。區外停車場：北屯區太原停車場、東勢區東勢河濱公園。\n台中市警局局長楊源明提醒，為維護活動期間周邊交通秩序及順暢，警察局將派員於各管制路段起迄點進行管制，並加強周邊巡邏，針對違規停車等妨害交通秩序之行為予以執法。'

# ---------- 編碼輸入 ----------
inputs = tokenizer(
    input_text,
    return_tensors="pt",
    truncation=True,
    max_length=512  # T5 的最大長度
).to(device)

# ---------- 產生輸出 ----------
with torch.no_grad():
    # outputs = model.generate(
    #     inputs["input_ids"],
    #     attention_mask=inputs["attention_mask"],
    #     max_length=64
    # )

    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=64,
        do_sample=True,
        top_p=0.92, 
        top_k=0,
        temperature=1,  # 控制生成的隨機性
    )

# ---------- 解碼輸出 ----------
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("生成結果：", result)
print("Raw output tokens:", outputs[0])
print("Decoded (no skip):", tokenizer.decode(outputs[0]))
print("Decoded (skip special):", tokenizer.decode(outputs[0], skip_special_tokens=True))