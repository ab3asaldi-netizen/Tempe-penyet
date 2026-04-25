"""
============================================================
AI CRYPTO SCANNER - GEMINI POWERED
============================================================
✅ Analisa FULL oleh Gemini AI - tanpa indikator hardcoded
✅ AI tentukan sendiri: LONG/SHORT/HOLD
✅ AI tentukan leverage & size berdasarkan confidence
✅ Monitor TP/SL real-time via price cache
✅ Sinyal + reasoning lengkap ke Telegram
✅ Rekap harian jam 21.00 WIB
✅ Command /rekap /status /help via Telegram
✅ Modal Rp23jt, risk 2% per trade
============================================================
"""

import os
import re
import time
import json
import threading
import requests
import numpy as np
from datetime import datetime, timezone, timedelta
from flask import Flask, request

# ─────────────────────────────────────────────
# ⚙️ KONFIGURASI
# ─────────────────────────────────────────────
TG_TOKEN       = os.environ.get('TG_TOKEN',       'ISI_TOKEN_TELEGRAM')
TG_CHAT_ID     = os.environ.get('TG_CHAT_ID',     'ISI_CHAT_ID_TELEGRAM')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', 'ISI_GEMINI_API_KEY')

MODAL_USD  = 1380.0
RISK_PCT   = 0.02
TRADE_SIZE = MODAL_USD * RISK_PCT  # $27.6
USD_TO_IDR = 16300
MODAL_IDR  = 23_000_000
SCAN_INTERVAL = 60
WIB = timezone(timedelta(hours=7))

PAIRS = [
    'BTCUSDT','ETHUSDT','BNBUSDT','SOLUSDT','XRPUSDT',
    'DOGEUSDT','ADAUSDT','AVAXUSDT','LINKUSDT','DOTUSDT',
    
]

LEVERAGE_MAP = {
    'VERY_HIGH': 10,
    'HIGH':       7,
    'MEDIUM':     5,
    'LOW':        3,
}

# ─────────────────────────────────────────────
# 💾 STORAGE IN-MEMORY
# ─────────────────────────────────────────────
signals      = {}
daily_stats  = {}
snapshot_21  = {}
price_cache  = {}

# ─────────────────────────────────────────────
# 📡 FETCH DATA KRAKEN
# ─────────────────────────────────────────────
def fetch_klines(symbol, limit=100):
    coin = symbol.replace('USDT', '')
    kraken_map = {
        'BTC':'XBT','DOGE':'XDG','MATIC':'POL',
       'FET':'FET','LDO':'LDO','SEI':'SEI',
        'SUI':'SUI','APT':'APT','ARB':'ARB',
        'OP':'OP','MKR':'MKR','UNI':'UNI',
        'AAVE':'AAVE','INJ':'INJ','NEAR':'NEAR'
    }
    coin = kraken_map.get(coin, coin)
    r = requests.get(
        'https://api.kraken.com/0/public/OHLC',
        params={'pair': f'{coin}USD', 'interval': 15},
        timeout=10
    )
    r.raise_for_status()
    data = r.json()
    if data.get('error') and len(data['error']) > 0:
        raise Exception(f"Kraken: {data['error']}")
    keys = [k for k in data['result'] if k != 'last']
    if not keys:
        raise Exception('No data from Kraken')
    raw = data['result'][keys[0]][-limit:]
    return [{'time':int(k[0]),'o':float(k[1]),'h':float(k[2]),
              'l':float(k[3]),'c':float(k[4]),'v':float(k[6])} for k in raw]

# ─────────────────────────────────────────────
# 📊 SIAPKAN DATA UNTUK AI
# ─────────────────────────────────────────────
def prepare_data(klines, symbol):
    closes  = [k['c'] for k in klines]
    highs   = [k['h'] for k in klines]
    lows    = [k['l'] for k in klines]
    volumes = [k['v'] for k in klines]

    last     = closes[-1]
    change   = (last - closes[-2]) / closes[-2] * 100
    h20      = max(highs[-20:])
    l20      = min(lows[-20:])
    vol_avg  = np.mean(volumes[-20:])
    vol_ratio = volumes[-1] / vol_avg if vol_avg > 0 else 1

    candles = []
    for k in klines[-30:]:
        t = datetime.fromtimestamp(k['time'], tz=WIB).strftime('%H:%M')
        candles.append(f"{t} O:{k['o']:.4f} H:{k['h']:.4f} L:{k['l']:.4f} C:{k['c']:.4f} V:{k['v']:.2f}")

    return {
        'symbol':    symbol,
        'last':      last,
        'change':    change,
        'h20':       h20,
        'l20':       l20,
        'vol_ratio': vol_ratio,
        'candles':   '\n'.join(candles)
    }

# ─────────────────────────────────────────────
# 🤖 ANALISA DENGAN GEMINI AI
# ─────────────────────────────────────────────
def analyze_with_gemini(data):
    prompt = f"""Kamu adalah trader crypto profesional berpengalaman 10+ tahun.
Analisa data candle berikut dan berikan keputusan trading FUTURES.

PAIR: {data['symbol']}
TIMEFRAME: 15 menit
HARGA SEKARANG: ${data['last']:.4f}
PERUBAHAN: {data['change']:+.2f}%
HIGH 20 CANDLE: ${data['h20']:.4f}
LOW 20 CANDLE: ${data['l20']:.4f}
VOLUME RATIO: {data['vol_ratio']:.2f}x dari rata-rata

DATA CANDLE 30 TERAKHIR (Waktu O H L C Volume):
{data['candles']}

Analisa secara mendalam pola harga, momentum, trend, support/resistance, volume.
Tentukan leverage berdasarkan confidence (makin tinggi confidence makin tinggi leverage).

Berikan response dalam format JSON PERSIS seperti ini (hanya JSON, tanpa teks lain):
{{
  "direction": "LONG atau SHORT atau HOLD",
  "confidence": 75,
  "leverage": 5,
  "stop_loss": {data['last']:.4f},
  "take_profit": {data['last']:.4f},
  "reasoning": "Penjelasan analisa dalam bahasa Indonesia max 3 kalimat",
  "key_levels": "Level support/resistance penting",
  "risk_note": "Catatan risiko jika ada"
}}

Aturan penting:
- confidence: 0-100
- leverage: 3-10 sesuai confidence
- stop_loss harus LEBIH RENDAH dari harga jika LONG, LEBIH TINGGI jika SHORT
- take_profit harus LEBIH TINGGI dari harga jika LONG, LEBIH RENDAH jika SHORT
- Jika sinyal tidak jelas, gunakan HOLD
- HANYA berikan JSON valid, tidak ada teks tambahan"""

    url = (f'https://generativelanguage.googleapis.com/v1/'
           f'models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}')

for attempt in range(3):
    r = requests.post(url, json={
        'contents': [{'parts': [{'text': prompt}]}],
        'generationConfig': {'temperature': 0.3, 'maxOutputTokens': 400}
    }, timeout=30)
    if r.status_code == 429:
        wait = (attempt + 1) * 20  # 20, 40, 60 detik
        print(f'Rate limit, tunggu {wait}s...')
        time.sleep(wait)
        continue
    r.raise_for_status()
    break

    text = r.json()['candidates'][0]['content']['parts'][0]['text'].strip()
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    text = text.strip()

    parsed     = json.loads(text)
    direction  = str(parsed.get('direction', 'HOLD')).upper()
    confidence = int(parsed.get('confidence', 50))
    leverage   = int(parsed.get('leverage', 5))
    sl         = float(parsed.get('stop_loss', 0))
    tp         = float(parsed.get('take_profit', 0))
    reasoning  = str(parsed.get('reasoning', '-'))
    key_levels = str(parsed.get('key_levels', '-'))
    risk_note  = str(parsed.get('risk_note', '-'))
    last       = data['last']

    # Validasi SL/TP
    if direction == 'LONG':
        if sl >= last: sl = last * 0.985
        if tp <= last: tp = last * 1.025
    elif direction == 'SHORT':
        if sl <= last: sl = last * 1.015
        if tp >= last: tp = last * 0.975

    # Confidence level
    if confidence >= 85:   conf_level = 'VERY_HIGH'
    elif confidence >= 70: conf_level = 'HIGH'
    elif confidence >= 55: conf_level = 'MEDIUM'
    else:                  conf_level = 'LOW'

    leverage = min(leverage, LEVERAGE_MAP[conf_level])

    return {
        'direction':  direction,
        'confidence': confidence,
        'conf_level': conf_level,
        'leverage':   leverage,
        'entry':      last,
        'sl':         sl,
        'tp':         tp,
        'reasoning':  reasoning,
        'key_levels': key_levels,
        'risk_note':  risk_note
    }

# ─────────────────────────────────────────────
# 💾 STATISTIK
# ─────────────────────────────────────────────
def get_today():
    return datetime.now(WIB).strftime('%d/%m/%Y')

def get_stats():
    today = get_today()
    if today not in daily_stats:
        daily_stats[today] = {
            'win':0,'loss':0,'total_profit':0.0,
            'total_loss':0.0,'pnl':0.0,
            'total_signal':0,'pairs':{}
        }
    return daily_stats[today]

def update_stats(hit, pnl, symbol):
    s = get_stats()
    if hit == 'WIN': s['win'] += 1; s['total_profit'] += abs(pnl)
    else:            s['loss'] += 1; s['total_loss']   += abs(pnl)
    s['pnl'] += pnl
    if symbol not in s['pairs']:
        s['pairs'][symbol] = {'win':0,'loss':0,'pnl':0.0}
    if hit == 'WIN': s['pairs'][symbol]['win'] += 1
    else:            s['pairs'][symbol]['loss'] += 1
    s['pairs'][symbol]['pnl'] += pnl

# ─────────────────────────────────────────────
# 💬 TELEGRAM
# ─────────────────────────────────────────────
def send_tg(text, chat_id=None):
    if not chat_id: chat_id = TG_CHAT_ID
    try:
        requests.post(
            f'https://api.telegram.org/bot{TG_TOKEN}/sendMessage',
            json={'chat_id':chat_id,'text':text,'parse_mode':'Markdown'},
            timeout=10
        )
    except Exception as e:
        print(f'TG error: {e}')

# ─────────────────────────────────────────────
# 📨 KIRIM SINYAL
# ─────────────────────────────────────────────
def send_signal(symbol, ai):
    sig_id = f"{symbol}_{int(time.time())}"
    size   = TRADE_SIZE

    pot_profit = abs((ai['tp'] - ai['entry']) / ai['entry'] * size * ai['leverage'])
    pot_loss   = abs((ai['sl'] - ai['entry']) / ai['entry'] * size * ai['leverage'])
    profit_idr = int(pot_profit * USD_TO_IDR)
    loss_idr   = int(pot_loss   * USD_TO_IDR)
    rr         = pot_profit / pot_loss if pot_loss > 0 else 0

    conf_emoji = {'VERY_HIGH':'🔥🔥🔥','HIGH':'🔥🔥','MEDIUM':'🔥','LOW':'⚠️'}.get(ai['conf_level'],'🔥')

    signals[sig_id] = {
        'id':sig_id,'symbol':symbol,'dir':ai['direction'],
        'entry':ai['entry'],'sl':ai['sl'],'tp':ai['tp'],
        'leverage':ai['leverage'],'size':size,
        'status':'OPEN','time':datetime.now(WIB).strftime('%d/%m/%Y %H:%M:%S'),
        'open_ts':time.time(),'pnl':0.0
    }
    get_stats()['total_signal'] += 1

    e = '🟢' if ai['direction'] == 'LONG' else '🔴'
    send_tg(
        f"{e} *{ai['direction']} SIGNAL — {symbol}*\n"
        f"━━━━━━━━━━━━━━\n"
        f"💹 *Harga: ${ai['entry']:.4f}*\n"
        f"🛑 Stop Loss: ${ai['sl']:.4f}\n"
        f"🎯 Take Profit: ${ai['tp']:.4f}\n"
        f"━━━━━━━━━━━━━━\n"
        f"🤖 *Analisa AI:*\n"
        f"_{ai['reasoning']}_\n\n"
        f"📍 *Level Kunci:* {ai['key_levels']}\n"
        f"⚠️ *Risk:* {ai['risk_note']}\n"
        f"━━━━━━━━━━━━━━\n"
        f"{conf_emoji} *Confidence: {ai['confidence']}%* ({ai['conf_level']})\n"
        f"⚡ *Leverage: {ai['leverage']}x* (AI determined)\n"
        f"⚖️ *R/R: 1:{rr:.1f}*\n"
        f"━━━━━━━━━━━━━━\n"
        f"💵 Modal: ${size:.1f} x{ai['leverage']}x\n"
        f"✅ Pot.Profit: +${pot_profit:.2f} (+Rp{profit_idr:,})\n"
        f"❌ Pot.Loss:   -${pot_loss:.2f} (-Rp{loss_idr:,})\n"
        f"⏰ {datetime.now(WIB).strftime('%d/%m/%Y %H:%M:%S')}\n"
        f"⏳ _Memantau TP/SL real-time..._"
    )
    print(f"[SIGNAL] {symbol} {ai['direction']} conf={ai['confidence']}% lev={ai['leverage']}x")

# ─────────────────────────────────────────────
# 🔍 MONITOR POSISI
# ─────────────────────────────────────────────
def monitor_positions():
    now = datetime.now(WIB)
    for sig_id, d in list(signals.items()):
        if d['status'] != 'OPEN': continue
        if d['symbol'] not in price_cache: continue

        pc  = price_cache[d['symbol']]
        hit = None

        if d['dir'] == 'LONG':
            if pc['h'] >= d['tp']: hit = 'WIN'
            if pc['l'] <= d['sl']: hit = 'LOSS'
        else:
            if pc['l'] <= d['tp']: hit = 'WIN'
            if pc['h'] >= d['sl']: hit = 'LOSS'

        if not hit: continue

        exit_p  = d['tp'] if hit == 'WIN' else d['sl']
        diff    = (exit_p - d['entry']) if d['dir'] == 'LONG' else (d['entry'] - exit_p)
        pnl     = (diff / d['entry']) * d['size'] * d['leverage']
        pnl_abs = abs(pnl)
        pnl_idr = int(pnl_abs * USD_TO_IDR)

        dur_sec = int(time.time() - d['open_ts'])
        dur_str = (f"{dur_sec//3600}j {(dur_sec%3600)//60}m"
                   if dur_sec >= 3600 else f"{dur_sec//60} menit")

        signals[sig_id].update({
            'status': hit, 'exit': exit_p,
            'pnl': pnl,
            'exit_time': now.strftime('%d/%m/%Y %H:%M:%S')
        })
        update_stats(hit, pnl, d['symbol'])

        emoji   = '✅' if hit == 'WIN' else '❌'
        pnl_str = f"+${pnl_abs:.2f}" if pnl >= 0 else f"-${pnl_abs:.2f}"
        idr_str = f"+Rp{pnl_idr:,}" if pnl >= 0 else f"-Rp{pnl_idr:,}"

        send_tg(
            f"{emoji} *HASIL: {hit}*\n"
            f"━━━━━━━━━━━━━━\n"
            f"🔹 *{d['symbol']}* — {d['dir']} x{d['leverage']}\n"
            f"💹 Harga: ${pc['c']:.4f}\n"
            f"📍 Entry: ${d['entry']:.4f}\n"
            f"🚪 Exit: ${exit_p:.4f}\n"
            f"━━━━━━━━━━━━━━\n"
            f"💵 PnL: *{pnl_str}* ({idr_str})\n"
            f"⏱️ Durasi: {dur_str}\n"
            f"⏰ {signals[sig_id]['exit_time']}\n"
            f"━━━━━━━━━━━━━━\n"
            + ("🎉 _Take profit tercapai!_" if hit == 'WIN' else "⚠️ _Stop loss terkena. Next!_")
        )

# ─────────────────────────────────────────────
# 📊 REKAP
# ─────────────────────────────────────────────
def build_rekap(stats, title, since=''):
    total  = stats['win'] + stats['loss']
    wr     = f"{stats['win']/total*100:.1f}" if total > 0 else '0'
    profit = stats.get('total_profit', 0)
    loss   = stats.get('total_loss', 0)
    pnl    = stats.get('pnl', 0)
    open_c = sum(1 for s in signals.values() if s['status'] == 'OPEN')

    profit_idr = int(profit * USD_TO_IDR)
    loss_idr   = int(loss   * USD_TO_IDR)
    pnl_idr    = int(abs(pnl) * USD_TO_IDR)
    pnl_str    = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
    idr_str    = f"+Rp{pnl_idr:,}" if pnl >= 0 else f"-Rp{pnl_idr:,}"

    best = worst = '-'
    bwr = -1; wwr = 101
    for sym, p in stats.get('pairs', {}).items():
        pt = p['win'] + p['loss']
        if pt == 0: continue
        pwr = p['win'] / pt * 100
        if pwr > bwr: bwr = pwr; best  = f"{sym} ({p['win']}W/{p['loss']}L)"
        if pwr < wwr: wwr = pwr; worst = f"{sym} ({p['win']}W/{p['loss']}L)"

    return (
        f"📊 *{title}*\n"
        + (f"_{since}_\n" if since else '')
        + f"━━━━━━━━━━━━━━\n"
        f"🤖 Powered by Gemini AI\n"
        f"📅 {get_today()}\n"
        f"━━━━━━━━━━━━━━\n"
        f"📡 Total Sinyal: *{stats.get('total_signal',0)}*\n"
        f"✅ Win: *{stats['win']}*\n"
        f"❌ Loss: *{stats['loss']}*\n"
        f"⏳ Open: *{open_c}*\n"
        f"📈 Closed: *{total}*\n"
        f"🏆 Win Rate: *{wr}%*\n"
        f"━━━━━━━━━━━━━━\n"
        f"💰 *DETAIL PnL:*\n"
        f"💵 Modal: Rp{MODAL_IDR:,}\n"
        f"📈 Profit: *+${profit:.2f}* (+Rp{profit_idr:,})\n"
        f"📉 Loss:   *-${loss:.2f}* (-Rp{loss_idr:,})\n"
        f"💹 Net: *{pnl_str}* ({idr_str})\n"
        f"━━━━━━━━━━━━━━\n"
        f"🔥 Best: {best}\n"
        f"💀 Worst: {worst}\n"
        f"━━━━━━━━━━━━━━\n"
        f"⚠️ _Simulasi modal Rp{MODAL_IDR:,}_"
    )

def rekap_harian():
    today = get_today()
    send_tg(build_rekap(get_stats(), 'REKAP BACKTEST HARIAN'))
    snapshot_21[today] = json.loads(json.dumps(get_stats()))

def rekap_manual(chat_id=None):
    today     = get_today()
    curr      = get_stats()
    yesterday = (datetime.now(WIB) - timedelta(days=1)).strftime('%d/%m/%Y')
    snap      = snapshot_21.get(yesterday, {
        'win':0,'loss':0,'total_profit':0,
        'total_loss':0,'pnl':0,'total_signal':0,'pairs':{}
    })
    delta = {
        'win':          curr['win']          - snap.get('win',0),
        'loss':         curr['loss']         - snap.get('loss',0),
        'total_profit': curr['total_profit'] - snap.get('total_profit',0),
        'total_loss':   curr['total_loss']   - snap.get('total_loss',0),
        'pnl':          curr['pnl']          - snap.get('pnl',0),
        'total_signal': curr['total_signal'] - snap.get('total_signal',0),
        'pairs':        curr['pairs']
    }
    send_tg(build_rekap(delta, 'REKAP SEMENTARA',
                        f"Sejak {yesterday} 21.00 → Sekarang"), chat_id)

# ─────────────────────────────────────────────
# 🔁 LOOP SCANNER
# ─────────────────────────────────────────────
def scanner_loop():
    pair_idx       = 0
    last_rekap_day = None
    batch_size     = 1

    print(f'[{datetime.now(WIB)}] AI Scanner dimulai — {len(PAIRS)} pairs')
    send_tg(
        f"🤖 *AI Scanner Aktif!*\n\n"
        f"⚙️ *Setting:*\n"
        f"• Engine: *Gemini AI*\n"
        f"• Modal: Rp{MODAL_IDR:,}\n"
        f"• Risk: {RISK_PCT*100:.0f}% per trade (${TRADE_SIZE:.1f})\n"
        f"• Leverage: AI determined (3-10x)\n"
        f"• Pairs: {len(PAIRS)}\n"
        f"• Rekap: Jam 21.00 WIB\n\n"
        f"📱 Commands: /rekap /status /help"
    )

    while True:
        try:
            now = datetime.now(WIB)

            if now.hour == 21 and now.minute == 0:
                today = now.strftime('%d/%m/%Y')
                if last_rekap_day != today:
                    rekap_harian()
                    last_rekap_day = today

            for i in range(batch_size):
                pidx   = (pair_idx + i) % len(PAIRS)
                symbol = PAIRS[pidx]
                try:
                    klines = fetch_klines(symbol)
                    last   = klines[-1]
                    price_cache[symbol] = {
                        'h':last['h'],'l':last['l'],
                        'c':last['c'],'t':time.time()
                    }
                    data = prepare_data(klines, symbol)
                    ai   = analyze_with_gemini(data)

                    print(f"[{now.strftime('%H:%M:%S')}] {symbol}: "
                          f"{ai['direction']} conf={ai['confidence']}% lev={ai['leverage']}x")

                    if ai['direction'] in ('LONG','SHORT'):
                        existing = any(
                            s['symbol'] == symbol and s['status'] == 'OPEN'
                            for s in signals.values()
                        )
                        if not existing:
                            send_signal(symbol, ai)

                except Exception as e:
                    print(f"[{now.strftime('%H:%M:%S')}] {symbol} error: {e}")

                time.sleep(5)

            pair_idx = (pair_idx + batch_size) % len(PAIRS)
            monitor_positions()

        except Exception as e:
            print(f'[{datetime.now(WIB)}] Loop error: {e}')

        time.sleep(10)

# ─────────────────────────────────────────────
# 🌐 FLASK + AUTO START
# ─────────────────────────────────────────────
app = Flask(__name__)


@app.route('/')
def home():
    s  = get_stats()
    oc = sum(1 for x in signals.values() if x['status'] == 'OPEN')
    t  = s['win'] + s['loss']
    return {
        'status':'running','engine':'Gemini AI',
        'pairs':len(PAIRS),
        'time':datetime.now(WIB).strftime('%d/%m/%Y %H:%M:%S WIB'),
        'today':{
            'signals':s['total_signal'],'win':s['win'],
            'loss':s['loss'],'open':oc,
            'win_rate':f"{s['win']/t*100:.1f}%" if t>0 else '0%',
            'pnl_usd':round(s['pnl'],2),
            'pnl_idr':int(s['pnl']*USD_TO_IDR)
        }
    }

@app.route('/rekap')
def rekap_ep():
    rekap_manual(); return {'status':'sent'}

@app.route('/status')
def status_ep():
    op = [{'symbol':s['symbol'],'dir':s['dir'],'entry':s['entry'],
            'leverage':s['leverage'],'sl':s['sl'],'tp':s['tp']}
          for s in signals.values() if s['status']=='OPEN']
    return {'open_positions':op,'count':len(op)}

@app.route(f'/webhook/{TG_TOKEN}', methods=['POST'])
def webhook():
    data = request.json
    if not data: return 'ok'
    msg  = data.get('message',{})
    if not msg: return 'ok'
    text    = msg.get('text','')
    chat_id = str(msg.get('chat',{}).get('id',''))

    if text == '/rekap':
        threading.Thread(target=rekap_manual, args=(chat_id,)).start()
    elif text == '/status':
        oc = sum(1 for s in signals.values() if s['status']=='OPEN')
        s  = get_stats()
        t  = s['win']+s['loss']
        send_tg(
            f"📈 *STATUS AI SCANNER*\n\n"
            f"🤖 Engine: Gemini AI\n"
            f"🟢 Scanner: Aktif\n"
            f"⏳ Open: {oc}\n"
            f"📡 Sinyal: {s['total_signal']}\n"
            f"✅ Win: {s['win']} | ❌ Loss: {s['loss']}\n"
            f"🏆 WR: {f'{s[chr(119)]*100/t:.1f}%' if t>0 else '0%'}\n"
            f"💹 PnL: ${s['pnl']:.2f}\n"
            f"⏰ {datetime.now(WIB).strftime('%d/%m/%Y %H:%M:%S')}",
            chat_id
        )
    elif text == '/help':
        send_tg(
            "📋 *COMMANDS:*\n\n"
            "/rekap — Rekap sejak 21.00\n"
            "/status — Status scanner\n"
            "/help — Bantuan\n\n"
            "🤖 _Powered by Gemini AI_",
            chat_id
        )
    return 'ok'
if __name__ == '__main__':
    print('[SCANNER] Starting...')
    t = threading.Thread(target=scanner_loop, daemon=True)
    t.start()
    print('[SCANNER] Thread started!')
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, threaded=True, use_reloader=False)
