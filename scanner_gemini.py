"""
============================================================
CRYPTO AI SCANNER - POWERED BY GOOGLE GEMINI
============================================================
✅ Data dari Kraken public API (tidak diblokir Railway)
✅ 15 pairs top crypto
✅ Analisa FULL oleh Gemini AI (indikator, arah, confidence)
✅ Leverage & modal ditentukan AI berdasarkan confidence
✅ Simulasi modal Rp20.000.000
✅ Monitor TP/SL real-time
✅ Sinyal + hasil WIN/LOSS ke Telegram
✅ Rekap harian jam 21.00 WIB
✅ Command /rekap /status /help
============================================================
"""

import os
import time
import json
import requests
import threading
import numpy as np
from datetime import datetime, timezone, timedelta
from flask import Flask, request

# ─────────────────────────────────────────────
# ⚙️ KONFIGURASI
# ─────────────────────────────────────────────
TG_TOKEN      = os.environ.get('TG_TOKEN',      'ISI_TOKEN_TELEGRAM')
TG_CHAT_ID    = os.environ.get('TG_CHAT_ID',    'ISI_CHAT_ID_TELEGRAM')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', 'ISI_GEMINI_API_KEY')

MODAL_IDR     = 20_000_000   # Rp20 juta
USD_TO_IDR    = 16_300
MODAL_USD     = MODAL_IDR / USD_TO_IDR   # ~$1227
SCAN_INTERVAL = 60           # detik antar scan (lebih lama karena pakai AI)
WIB           = timezone(timedelta(hours=7))

# 15 Pairs terbaik (tidak diblokir Kraken)
PAIRS = [
    'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT',
    'ADAUSDT', 'AVAXUSDT', 'LINKUSDT', 'DOTUSDT', 'LTCUSDT',
    'UNIUSDT', 'AAVEUSDT', 'ATOMUSDT', 'NEARUSDT', 'MATICUSDT'
]

# ─────────────────────────────────────────────
# 💾 STORAGE IN-MEMORY
# ─────────────────────────────────────────────
signals      = {}   # posisi aktif  {symbol: {...}}
daily_stats  = {}   # statistik harian
scan_log     = []   # log scan terakhir (maks 50)

app = Flask(__name__)

# ─────────────────────────────────────────────
# 📡 FETCH DATA KRAKEN
# ─────────────────────────────────────────────
KRAKEN_MAP = {
    'BTC': 'XBT', 'DOGE': 'XDG', 'MATIC': 'POL',
    'ETH': 'ETH', 'SOL': 'SOL', 'XRP': 'XRP',
    'ADA': 'ADA', 'AVAX': 'AVAX', 'LINK': 'LINK',
    'DOT': 'DOT', 'LTC': 'LTC', 'UNI': 'UNI',
    'AAVE': 'AAVE', 'ATOM': 'ATOM', 'NEAR': 'NEAR'
}

def fetch_klines(symbol, limit=100):
    coin = symbol.replace('USDT', '')
    coin = KRAKEN_MAP.get(coin, coin)
    url  = 'https://api.kraken.com/0/public/OHLC'
    r    = requests.get(url, params={'pair': f'{coin}USD', 'interval': 15}, timeout=10)
    r.raise_for_status()
    data = r.json()
    if data.get('error') and data['error']:
        raise Exception(f"Kraken: {data['error']}")
    keys = [k for k in data['result'] if k != 'last']
    if not keys:
        raise Exception('No Kraken data')
    raw = data['result'][keys[0]][-limit:]
    return [{
        'time': k[0],
        'o': float(k[1]),
        'h': float(k[2]),
        'l': float(k[3]),
        'c': float(k[4]),
        'v': float(k[6])
    } for k in raw]

# ─────────────────────────────────────────────
# 📊 HITUNG INDIKATOR DASAR (untuk data ke Gemini)
# ─────────────────────────────────────────────
def calc_indicators(klines):
    closes  = np.array([k['c'] for k in klines])
    highs   = np.array([k['h'] for k in klines])
    lows    = np.array([k['l'] for k in klines])
    volumes = np.array([k['v'] for k in klines])

    # EMA
    def ema(data, n):
        k, e = 2/(n+1), np.mean(data[:n])
        for p in data[n:]: e = p*k + e*(1-k)
        return e

    # RSI
    def rsi(data, n=14):
        d = np.diff(data[-(n+2):])
        g = np.where(d>0,d,0); l = np.where(d<0,-d,0)
        ag = np.mean(g[:n]); al = np.mean(l[:n])
        return 100.0 if al==0 else 100-(100/(1+ag/al))

    # MACD
    e12 = ema(closes, 12); e26 = ema(closes, 26)
    macd = e12 - e26

    # Bollinger Bands
    bb_mid = np.mean(closes[-20:])
    bb_std = np.std(closes[-20:])
    bb_upper = bb_mid + 2*bb_std
    bb_lower = bb_mid - 2*bb_std

    # ATR
    trs = [max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1]))
           for i in range(-14, 0)]
    atr = np.mean(trs)

    # Volume
    vol_avg = np.mean(volumes[-20:-1])
    vol_ratio = volumes[-1]/vol_avg if vol_avg > 0 else 1.0

    # Stochastic RSI
    rsi_vals = [rsi(closes[:i+1]) for i in range(len(closes)-15, len(closes))]
    rsi_min, rsi_max = min(rsi_vals), max(rsi_vals)
    stoch_rsi = (rsi_vals[-1]-rsi_min)/(rsi_max-rsi_min)*100 if rsi_max!=rsi_min else 50

    # Support/Resistance (simple)
    recent_high = float(np.max(highs[-20:]))
    recent_low  = float(np.min(lows[-20:]))

    # Candle pattern last 3
    last3 = []
    for k in klines[-3:]:
        body = k['c'] - k['o']
        last3.append('bullish' if body > 0 else 'bearish' if body < 0 else 'doji')

    return {
        'price':       round(closes[-1], 6),
        'ema9':        round(ema(closes, 9), 6),
        'ema21':       round(ema(closes, 21), 6),
        'ema50':       round(ema(closes, 50), 6),
        'ema200':      round(ema(closes, 100), 6),  # pakai 100 karena limit data
        'rsi':         round(rsi(closes), 2),
        'stoch_rsi':   round(stoch_rsi, 2),
        'macd':        round(macd, 6),
        'bb_upper':    round(bb_upper, 6),
        'bb_mid':      round(bb_mid, 6),
        'bb_lower':    round(bb_lower, 6),
        'atr':         round(atr, 6),
        'vol_ratio':   round(vol_ratio, 2),
        'recent_high': round(recent_high, 6),
        'recent_low':  round(recent_low, 6),
        'candle_last3': last3,
        'price_vs_bb':  round((closes[-1] - bb_lower) / (bb_upper - bb_lower) * 100, 1) if bb_upper != bb_lower else 50
    }

# ─────────────────────────────────────────────
# 🤖 ANALISA GEMINI AI
# ─────────────────────────────────────────────
def ask_gemini(symbol, indicators, modal_usd):
    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash-latest:generateContent?key={GEMINI_API_KEY}"

    prompt = f"""Kamu adalah AI trading analyst profesional untuk crypto futures.
Analisa data berikut untuk {symbol} timeframe 15 menit dan berikan keputusan trading.

DATA TEKNIKAL:
- Harga saat ini: ${indicators['price']}
- EMA9: {indicators['ema9']} | EMA21: {indicators['ema21']} | EMA50: {indicators['ema50']}
- RSI(14): {indicators['rsi']} | Stoch RSI: {indicators['stoch_rsi']}
- MACD: {indicators['macd']}
- Bollinger: Upper={indicators['bb_upper']} Mid={indicators['bb_mid']} Lower={indicators['bb_lower']}
- Posisi harga di BB: {indicators['price_vs_bb']}% (0=lower band, 100=upper band)
- ATR(14): {indicators['atr']}
- Volume ratio vs avg20: {indicators['vol_ratio']}x
- Resistance: {indicators['recent_high']} | Support: {indicators['recent_low']}
- 3 candle terakhir: {indicators['candle_last3']}
- Modal tersedia: ${modal_usd:.2f}

TUGAS KAMU:
Berikan analisa mendalam dan keputusan trading. Kamu bebas memilih indikator mana yang paling relevan.

Jawab HANYA dalam format JSON ini (tanpa markdown, tanpa backtick):
{{
  "direction": "LONG" atau "SHORT" atau "SKIP",
  "confidence": angka 1-100,
  "leverage": angka 2-20,
  "modal_pct": angka 5-30 (persen modal yang dipakai),
  "tp_pct": angka persentase TP dari entry (misal 1.5),
  "sl_pct": angka persentase SL dari entry (misal 0.8),
  "reason": "penjelasan singkat max 100 karakter",
  "key_indicators": ["indikator1", "indikator2", "indikator3"],
  "market_condition": "trending" atau "ranging" atau "volatile",
  "risk_level": "low" atau "medium" atau "high"
}}

ATURAN:
- confidence >= 75: boleh LONG/SHORT
- confidence < 75: SKIP
- leverage: sesuaikan dengan confidence (confidence 75=lev 3, 85=lev 5, 90=lev 8, 95+=lev 10-15)
- modal_pct: sesuaikan risk (low=20-30%, medium=10-20%, high=5-10%)
- tp_pct dan sl_pct harus realistis berdasarkan ATR dan volatilitas
- Jika kondisi tidak jelas/sideways, pilih SKIP
"""

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 500
        }
    }

    try:
        r = requests.post(url, json=payload, timeout=20)
        r.raise_for_status()
        resp = r.json()
        text = resp['candidates'][0]['content']['parts'][0]['text'].strip()
        # Bersihkan jika ada markdown
        text = text.replace('```json', '').replace('```', '').strip()
        result = json.loads(text)
        return result
    except Exception as e:
        print(f"[GEMINI] Error untuk {symbol}: {e}")
        return None

# ─────────────────────────────────────────────
# 📤 TELEGRAM
# ─────────────────────────────────────────────
def send_tg(msg):
    try:
        url = f'https://api.telegram.org/bot{TG_TOKEN}/sendMessage'
        requests.post(url, json={
            'chat_id': TG_CHAT_ID,
            'text': msg,
            'parse_mode': 'HTML'
        }, timeout=10)
    except Exception as e:
        print(f"TG error: {e}")

# ─────────────────────────────────────────────
# 📈 KIRIM SINYAL KE TELEGRAM
# ─────────────────────────────────────────────
def send_signal(symbol, direction, price, ai_result, modal_used_usd):
    entry = price
    tp_pct = ai_result['tp_pct'] / 100
    sl_pct = ai_result['sl_pct'] / 100
    lev    = ai_result['leverage']
    conf   = ai_result['confidence']
    reason = ai_result['reason']
    keys   = ', '.join(ai_result.get('key_indicators', []))
    cond   = ai_result.get('market_condition', '-')
    risk   = ai_result.get('risk_level', '-')

    if direction == 'LONG':
        tp = entry * (1 + tp_pct)
        sl = entry * (1 - sl_pct)
        emoji = '🟢'
    else:
        tp = entry * (1 - tp_pct)
        sl = entry * (1 + sl_pct)
        emoji = '🔴'

    pot_profit = modal_used_usd * lev * tp_pct
    pot_loss   = modal_used_usd * lev * sl_pct
    rr         = round(tp_pct / sl_pct, 1)

    modal_used_idr = int(modal_used_usd * USD_TO_IDR)
    pot_profit_idr = int(pot_profit * USD_TO_IDR)
    pot_loss_idr   = int(pot_loss * USD_TO_IDR)

    # Confidence bar
    conf_bar = '█' * (conf // 10) + '░' * (10 - conf // 10)

    msg = (
        f"{emoji} <b>{direction} SIGNAL — {symbol}</b>\n"
        f"{'─'*30}\n"
        f"🤖 <b>AI Confidence: {conf}%</b>\n"
        f"<code>[{conf_bar}]</code>\n"
        f"📊 Market: {cond.upper()} | Risk: {risk.upper()}\n"
        f"{'─'*30}\n"
        f"💲 Entry: <b>${entry:.4f}</b>\n"
        f"🎯 TP: ${tp:.4f} (+{ai_result['tp_pct']}%)\n"
        f"🛑 SL: ${sl:.4f} (-{ai_result['sl_pct']}%)\n"
        f"⚖️ R/R: 1:{rr}\n"
        f"{'─'*30}\n"
        f"💼 Leverage: <b>{lev}x</b>\n"
        f"💰 Modal: Rp{modal_used_idr:,} (${modal_used_usd:.1f})\n"
        f"✅ Pot.Profit: +Rp{pot_profit_idr:,}\n"
        f"❌ Pot.Loss: -Rp{pot_loss_idr:,}\n"
        f"{'─'*30}\n"
        f"🧠 AI Keys: {keys}\n"
        f"📝 {reason}\n"
        f"🕐 {datetime.now(WIB).strftime('%d/%m/%Y %H:%M:%S')}"
    )
    send_tg(msg)

    # Simpan posisi aktif
    signals[symbol] = {
        'symbol':     symbol,
        'direction':  direction,
        'entry':      entry,
        'tp':         tp,
        'sl':         sl,
        'leverage':   lev,
        'modal_usd':  modal_used_usd,
        'confidence': conf,
        'tp_pct':     ai_result['tp_pct'],
        'sl_pct':     ai_result['sl_pct'],
        'reason':     reason,
        'time':       datetime.now(WIB).strftime('%H:%M'),
        'date':       datetime.now(WIB).strftime('%Y-%m-%d')
    }

# ─────────────────────────────────────────────
# 📉 MONITOR TP/SL
# ─────────────────────────────────────────────
def check_tpsl(symbol, current_price):
    if symbol not in signals:
        return
    sig = signals[symbol]
    direction = sig['direction']
    hit = None

    if direction == 'LONG':
        if current_price >= sig['tp']:
            hit = 'WIN'
        elif current_price <= sig['sl']:
            hit = 'LOSS'
    else:
        if current_price <= sig['tp']:
            hit = 'WIN'
        elif current_price >= sig['sl']:
            hit = 'LOSS'

    if hit:
        lev   = sig['leverage']
        modal = sig['modal_usd']
        tp_p  = sig['tp_pct'] / 100
        sl_p  = sig['sl_pct'] / 100

        if hit == 'WIN':
            pnl_usd = modal * lev * tp_p
            emoji   = '✅'
        else:
            pnl_usd = -(modal * lev * sl_p)
            emoji   = '❌'

        pnl_idr = int(pnl_usd * USD_TO_IDR)
        date    = sig['date']

        # Update statistik harian
        if date not in daily_stats:
            daily_stats[date] = {'win': 0, 'loss': 0, 'pnl': 0.0, 'trades': 0}
        daily_stats[date]['trades'] += 1
        daily_stats[date]['pnl']    += pnl_usd
        if hit == 'WIN':
            daily_stats[date]['win'] += 1
        else:
            daily_stats[date]['loss'] += 1

        msg = (
            f"{emoji} <b>{hit} — {symbol}</b>\n"
            f"{'─'*25}\n"
            f"📍 {'LONG' if direction=='LONG' else 'SHORT'} @ ${sig['entry']:.4f}\n"
            f"📌 Close @ ${current_price:.4f}\n"
            f"💼 {lev}x | Conf: {sig['confidence']}%\n"
            f"{'─'*25}\n"
            f"{'💰' if hit=='WIN' else '💸'} PnL: {'+'if pnl_usd>0 else ''}{pnl_usd:.2f}$ "
            f"({'+'if pnl_idr>0 else ''}Rp{pnl_idr:,})\n"
            f"🕐 {datetime.now(WIB).strftime('%H:%M:%S')}"
        )
        send_tg(msg)
        del signals[symbol]

# ─────────────────────────────────────────────
# 🔄 SCANNER UTAMA
# ─────────────────────────────────────────────
def scanner_loop():
    send_tg(
        "🤖 <b>AI Scanner AKTIF</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🧠 Engine: Google Gemini AI\n"
        f"📊 Pairs: {len(PAIRS)}\n"
        f"💰 Modal: Rp{MODAL_IDR:,}\n"
        f"⏱ Interval: {SCAN_INTERVAL}s\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🕐 {datetime.now(WIB).strftime('%d/%m/%Y %H:%M:%S')}"
    )

    while True:
        now = datetime.now(WIB)

        # Rekap otomatis jam 21:00
        if now.hour == 21 and now.minute == 0:
            rekap_manual()
            time.sleep(61)
            continue

        # Hitung sisa modal
        modal_terpakai = sum(s['modal_usd'] for s in signals.values())
        modal_sisa_usd = MODAL_USD - modal_terpakai

        for symbol in PAIRS:
            try:
                klines = fetch_klines(symbol)
                inds   = calc_indicators(klines)
                price  = inds['price']

                # Monitor TP/SL dulu
                check_tpsl(symbol, price)

                # Skip jika sudah ada posisi di pair ini
                if symbol in signals:
                    continue

                # Skip jika modal sisa < $20
                if modal_sisa_usd < 20:
                    print(f"[SKIP] Modal sisa terlalu kecil: ${modal_sisa_usd:.1f}")
                    continue

                # Tanya Gemini
                ai = ask_gemini(symbol, inds, modal_sisa_usd)
                if not ai:
                    continue

                direction = ai.get('direction', 'SKIP')
                conf      = ai.get('confidence', 0)

                # Log scan
                scan_log.append({
                    'time': now.strftime('%H:%M'),
                    'symbol': symbol,
                    'direction': direction,
                    'confidence': conf,
                    'price': price
                })
                if len(scan_log) > 100:
                    scan_log.pop(0)

                print(f"[{now.strftime('%H:%M:%S')}] {symbol}: {direction} conf={conf}% price={price}")

                if direction in ('LONG', 'SHORT') and conf >= 75:
                    modal_pct    = ai.get('modal_pct', 10) / 100
                    modal_used   = round(modal_sisa_usd * modal_pct, 2)
                    modal_used   = max(modal_used, 10.0)  # min $10
                    modal_sisa_usd -= modal_used

                    send_signal(symbol, direction, price, ai, modal_used)
                    time.sleep(2)  # jeda antar sinyal

            except Exception as e:
                print(f"[ERR] {symbol}: {e}")

            time.sleep(3)  # jeda antar pair (hindari rate limit Kraken)

        time.sleep(SCAN_INTERVAL)

# ─────────────────────────────────────────────
# 📊 REKAP
# ─────────────────────────────────────────────
def rekap_manual():
    today = datetime.now(WIB).strftime('%Y-%m-%d')
    stats = daily_stats.get(today, {'win': 0, 'loss': 0, 'pnl': 0.0, 'trades': 0})

    win    = stats['win']
    loss   = stats['loss']
    total  = stats['trades']
    pnl    = stats['pnl']
    wr     = round(win/total*100, 1) if total > 0 else 0
    pnl_idr = int(pnl * USD_TO_IDR)

    # Posisi masih terbuka
    open_pos = len(signals)

    emoji_pnl = '📈' if pnl >= 0 else '📉'

    msg = (
        f"📊 <b>REKAP HARIAN — {today}</b>\n"
        f"{'━'*25}\n"
        f"🤖 Engine: Gemini AI\n"
        f"{'━'*25}\n"
        f"✅ WIN  : {win}\n"
        f"❌ LOSS : {loss}\n"
        f"📋 Total: {total} trades\n"
        f"🎯 Win Rate: {wr}%\n"
        f"{'━'*25}\n"
        f"{emoji_pnl} PnL: {'+'if pnl>=0 else ''}{pnl:.2f}$ "
        f"({'+'if pnl_idr>=0 else ''}Rp{pnl_idr:,})\n"
        f"🔓 Open: {open_pos} posisi\n"
        f"{'━'*25}\n"
        f"💰 Modal Awal: Rp{MODAL_IDR:,}\n"
        f"💵 Modal ~: Rp{int((MODAL_USD + pnl) * USD_TO_IDR):,}\n"
        f"🕐 {datetime.now(WIB).strftime('%H:%M:%S')}"
    )
    send_tg(msg)

# ─────────────────────────────────────────────
# 🌐 FLASK ROUTES
# ─────────────────────────────────────────────
@app.route('/')
def home():
    return {
        'status': 'running',
        'engine': 'Gemini AI',
        'pairs': len(PAIRS),
        'open_positions': len(signals),
        'scan_log_count': len(scan_log)
    }

@app.route('/rekap')
def rekap_endpoint():
    rekap_manual()
    return {'status': 'sent'}

@app.route('/status')
def status_endpoint():
    open_list = []
    for s in signals.values():
        open_list.append({
            'symbol':    s['symbol'],
            'direction': s['direction'],
            'entry':     s['entry'],
            'confidence': s['confidence'],
            'leverage':  s['leverage'],
            'time':      s['time']
        })
    return {'open_positions': open_list, 'total': len(open_list)}

@app.route('/log')
def log_endpoint():
    return {'recent_scans': scan_log[-30:]}

@app.route(f'/webhook/{TG_TOKEN}', methods=['POST'])
def webhook():
    data = request.get_json(silent=True)
    if not data:
        return 'ok'
    try:
        msg  = data.get('message', {})
        text = msg.get('text', '').strip().lower()
        cid  = str(msg.get('chat', {}).get('id', ''))

        if cid != str(TG_CHAT_ID):
            return 'ok'

        if text == '/rekap':
            rekap_manual()

        elif text == '/status':
            if not signals:
                send_tg("📭 Tidak ada posisi terbuka saat ini.")
            else:
                lines = [f"🔓 <b>Posisi Terbuka ({len(signals)})</b>\n{'─'*25}"]
                for s in signals.values():
                    d = '🟢' if s['direction']=='LONG' else '🔴'
                    lines.append(
                        f"{d} {s['symbol']} {s['direction']}\n"
                        f"   Entry: ${s['entry']:.4f} | {s['leverage']}x\n"
                        f"   Conf: {s['confidence']}% | TP:{s['tp_pct']}% SL:{s['sl_pct']}%"
                    )
                send_tg('\n'.join(lines))

        elif text == '/help':
            send_tg(
                "🤖 <b>AI Scanner Commands</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                "/status — posisi terbuka\n"
                "/rekap  — rekap hari ini\n"
                "/help   — bantuan\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                f"🧠 Engine: Gemini AI\n"
                f"📊 Scanning {len(PAIRS)} pairs\n"
                f"⚡ Min confidence: 75%"
            )

        elif text == '/modal':
            terpakai = sum(s['modal_usd'] for s in signals.values())
            sisa     = MODAL_USD - terpakai
            send_tg(
                f"💰 <b>Info Modal</b>\n"
                f"Total: ${MODAL_USD:.0f} (Rp{MODAL_IDR:,})\n"
                f"Terpakai: ${terpakai:.1f}\n"
                f"Sisa: ${sisa:.1f}\n"
                f"Posisi aktif: {len(signals)}"
            )

    except Exception as e:
        print(f"Webhook error: {e}")

    return 'ok'

# ─────────────────────────────────────────────
# 🚀 MAIN
# ─────────────────────────────────────────────
if __name__ == '__main__':
    t = threading.Thread(target=scanner_loop, daemon=True)
    t.start()
    print("[SCANNER] Thread started!")
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
else:
    # Untuk gunicorn
    t = threading.Thread(target=scanner_loop, daemon=True)
    t.start()
    print("[SCANNER] Thread started via gunicorn!")
