"""
============================================================
AI CRYPTO SCANNER - OPENROUTER POWERED (FREE)
============================================================
✅ Data dari Kraken public API (tidak diblokir Railway)
✅ Analisa FULL oleh AI via OpenRouter (gratis, tanpa CC)
✅ AI tentukan sendiri: LONG/SHORT/HOLD
✅ AI tentukan leverage & size berdasarkan confidence
✅ Monitor TP/SL real-time via price cache
✅ Sinyal + reasoning lengkap ke Telegram
✅ Rekap harian jam 21.00 WIB
✅ Command /rekap /status /help via Telegram
✅ Modal Rp23jt, simulasi backtest
✅ Fallback otomatis jika model AI error
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
TG_TOKEN          = os.environ.get('TG_TOKEN',          'ISI_TOKEN_TELEGRAM')
TG_CHAT_ID        = os.environ.get('TG_CHAT_ID',        'ISI_CHAT_ID_TELEGRAM')
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY', 'ISI_OPENROUTER_KEY')

MODAL_USD     = 1380.0
RISK_PCT      = 0.02
TRADE_SIZE    = MODAL_USD * RISK_PCT   # $27.6
USD_TO_IDR    = 16_300
MODAL_IDR     = 23_000_000
SCAN_INTERVAL = 10   # detik antar pair
WIB           = timezone(timedelta(hours=7))

# Model OpenRouter gratis (fallback berurutan jika gagal)
AI_MODELS = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "mistralai/mistral-7b-instruct:free",
    "google/gemma-3-27b-it:free",
    "deepseek/deepseek-r1:free",
]

PAIRS = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
    'DOGEUSDT', 'ADAUSDT', 'AVAXUSDT', 'LINKUSDT', 'DOTUSDT',
    'LTCUSDT', 'UNIUSDT', 'NEARUSDT', 'ATOMUSDT', 'AAVEUSDT',
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
signals     = {}
daily_stats = {}
snapshot_21 = {}
price_cache = {}
app         = Flask(__name__)

# ─────────────────────────────────────────────
# 📡 FETCH DATA KRAKEN
# ─────────────────────────────────────────────
KRAKEN_MAP = {
    'BTC': 'XBT', 'DOGE': 'XDG', 'MATIC': 'POL',
    'FET': 'FET', 'LDO': 'LDO', 'SEI': 'SEI',
    'SUI': 'SUI', 'APT': 'APT', 'ARB': 'ARB',
    'OP': 'OP',   'MKR': 'MKR', 'UNI': 'UNI',
    'AAVE': 'AAVE', 'INJ': 'INJ', 'NEAR': 'NEAR',
    'BNB': 'BNB',
}

def fetch_klines(symbol, limit=100):
    coin = symbol.replace('USDT', '')
    coin = KRAKEN_MAP.get(coin, coin)
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
    return [{
        'time': int(k[0]),
        'o': float(k[1]),
        'h': float(k[2]),
        'l': float(k[3]),
        'c': float(k[4]),
        'v': float(k[6])
    } for k in raw]

# ─────────────────────────────────────────────
# 📊 SIAPKAN DATA UNTUK AI
# ─────────────────────────────────────────────
def prepare_data(klines, symbol):
    closes  = [k['c'] for k in klines]
    highs   = [k['h'] for k in klines]
    lows    = [k['l'] for k in klines]
    volumes = [k['v'] for k in klines]

    last      = closes[-1]
    change    = (last - closes[-2]) / closes[-2] * 100
    h20       = max(highs[-20:])
    l20       = min(lows[-20:])
    vol_avg   = np.mean(volumes[-20:])
    vol_ratio = volumes[-1] / vol_avg if vol_avg > 0 else 1.0

    # Hitung indikator dasar sebagai konteks tambahan untuk AI
    def ema(data, n):
        arr = np.array(data)
        k, e = 2/(n+1), np.mean(arr[:n])
        for p in arr[n:]: e = p*k + e*(1-k)
        return round(e, 6)

    def rsi(data, n=14):
        arr = np.array(data)
        d = np.diff(arr[-(n+2):])
        g = np.where(d>0,d,0); l = np.where(d<0,-d,0)
        ag = np.mean(g[:n]); al = np.mean(l[:n])
        return round(100.0 if al==0 else 100-(100/(1+ag/al)), 2)

    # Candle data ringkas untuk AI
    candles = []
    for k in klines[-20:]:
        t = datetime.fromtimestamp(k['time'], tz=WIB).strftime('%H:%M')
        body = 'Bull' if k['c'] >= k['o'] else 'Bear'
        candles.append(
            f"{t} {body} O:{k['o']:.4f} H:{k['h']:.4f} "
            f"L:{k['l']:.4f} C:{k['c']:.4f} V:{k['v']:.1f}"
        )

    return {
        'symbol':    symbol,
        'last':      last,
        'change':    round(change, 3),
        'h20':       h20,
        'l20':       l20,
        'vol_ratio': round(vol_ratio, 2),
        'ema9':      ema(closes, 9),
        'ema21':     ema(closes, 21),
        'ema50':     ema(closes, 50),
        'rsi':       rsi(closes),
        'candles':   '\n'.join(candles)
    }

# ─────────────────────────────────────────────
# 🤖 ANALISA DENGAN OPENROUTER AI
# ─────────────────────────────────────────────
def build_prompt(data):
    return f"""Kamu adalah trader crypto profesional berpengalaman 10+ tahun, spesialis futures trading.
Analisa data berikut dan berikan keputusan trading FUTURES dengan teliti.

PAIR: {data['symbol']} | TIMEFRAME: 15 menit
HARGA: ${data['last']:.4f} | PERUBAHAN: {data['change']:+.3f}%
HIGH-20: ${data['h20']:.4f} | LOW-20: ${data['l20']:.4f}
VOLUME RATIO: {data['vol_ratio']:.2f}x rata-rata
EMA9: {data['ema9']} | EMA21: {data['ema21']} | EMA50: {data['ema50']}
RSI(14): {data['rsi']}

DATA CANDLE 20 TERAKHIR (15 menit):
{data['candles']}

Analisa mendalam: pola harga, momentum, trend, support/resistance, volume, divergence.
Kamu bebas pakai indikator atau pola apapun yang kamu anggap relevan.

PENTING: Jawab HANYA dengan JSON valid ini, tanpa teks lain, tanpa markdown:
{{"direction":"LONG atau SHORT atau HOLD","confidence":75,"leverage":5,"stop_loss":{data['last']:.6f},"take_profit":{data['last']:.6f},"reasoning":"Analisa singkat max 2 kalimat","key_levels":"Support/resistance penting","risk_note":"Catatan risiko"}}

ATURAN WAJIB:
- direction: LONG, SHORT, atau HOLD
- confidence: 0-100 (berdasarkan seberapa yakin kamu)
- leverage: 3-10 (makin tinggi confidence, makin tinggi leverage)
- stop_loss LONG: HARUS lebih rendah dari {data['last']:.6f}
- stop_loss SHORT: HARUS lebih tinggi dari {data['last']:.6f}
- take_profit LONG: HARUS lebih tinggi dari {data['last']:.6f}
- take_profit SHORT: HARUS lebih rendah dari {data['last']:.6f}
- Jika sinyal tidak jelas: gunakan HOLD
- HANYA JSON, tidak ada teks tambahan apapun"""

def analyze_with_ai(data):
    prompt = build_prompt(data)
    headers = {
        'Authorization': f'Bearer {OPENROUTER_API_KEY}',
        'Content-Type': 'application/json',
        'HTTP-Referer': 'https://crypto-scanner.railway.app',
        'X-Title': 'Crypto AI Scanner'
    }

    # Coba tiap model secara berurutan
    for model in AI_MODELS:
        for attempt in range(2):
            try:
                r = requests.post(
                    'https://openrouter.ai/api/v1/chat/completions',
                    headers=headers,
                    json={
                        'model': model,
                        'messages': [{'role': 'user', 'content': prompt}],
                        'temperature': 0.2,
                        'max_tokens': 300,
                    },
                    timeout=25
                )

                if r.status_code == 429:
                    print(f'[AI] Rate limit model {model}, ganti model...')
                    break  # coba model berikutnya

                if r.status_code != 200:
                    print(f'[AI] Error {r.status_code} model {model}: {r.text[:100]}')
                    break

                resp = r.json()
                text = resp['choices'][0]['message']['content'].strip()

                # Bersihkan markdown jika ada
                text = re.sub(r'```json\s*', '', text)
                text = re.sub(r'```\s*', '', text)
                # Ambil hanya bagian JSON
                match = re.search(r'\{.*\}', text, re.DOTALL)
                if match:
                    text = match.group(0)

                parsed = json.loads(text)

                direction  = str(parsed.get('direction', 'HOLD')).upper()
                confidence = int(parsed.get('confidence', 50))
                leverage   = int(parsed.get('leverage', 5))
                sl         = float(parsed.get('stop_loss', 0))
                tp         = float(parsed.get('take_profit', 0))
                reasoning  = str(parsed.get('reasoning', '-'))
                key_levels = str(parsed.get('key_levels', '-'))
                risk_note  = str(parsed.get('risk_note', '-'))
                last       = data['last']

                # Validasi & koreksi SL/TP
                if direction == 'LONG':
                    if sl <= 0 or sl >= last: sl = last * 0.985
                    if tp <= 0 or tp <= last: tp = last * 1.025
                elif direction == 'SHORT':
                    if sl <= 0 or sl <= last: sl = last * 1.015
                    if tp <= 0 or tp >= last: tp = last * 0.975

                # Confidence level → leverage cap
                if confidence >= 85:   conf_level = 'VERY_HIGH'
                elif confidence >= 70: conf_level = 'HIGH'
                elif confidence >= 55: conf_level = 'MEDIUM'
                else:                  conf_level = 'LOW'

                leverage = min(max(leverage, 3), LEVERAGE_MAP[conf_level])

                print(f'[AI] {data["symbol"]} → {direction} conf={confidence}% lev={leverage}x [{model.split("/")[0]}]')

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
                    'risk_note':  risk_note,
                    'model':      model.split('/')[0]
                }

            except json.JSONDecodeError as e:
                print(f'[AI] JSON error model {model}: {e} | text: {text[:80]}')
                break
            except Exception as e:
                print(f'[AI] Exception model {model} attempt {attempt+1}: {e}')
                time.sleep(3)

    print(f'[AI] Semua model gagal untuk {data["symbol"]}')
    return None

# ─────────────────────────────────────────────
# 💾 STATISTIK
# ─────────────────────────────────────────────
def get_today():
    return datetime.now(WIB).strftime('%d/%m/%Y')

def get_stats():
    today = get_today()
    if today not in daily_stats:
        daily_stats[today] = {
            'win': 0, 'loss': 0,
            'total_profit': 0.0, 'total_loss': 0.0,
            'pnl': 0.0, 'total_signal': 0, 'pairs': {}
        }
    return daily_stats[today]

def update_stats(hit, pnl, symbol):
    s = get_stats()
    if hit == 'WIN':
        s['win'] += 1
        s['total_profit'] += abs(pnl)
    else:
        s['loss'] += 1
        s['total_loss'] += abs(pnl)
    s['pnl'] += pnl
    if symbol not in s['pairs']:
        s['pairs'][symbol] = {'win': 0, 'loss': 0, 'pnl': 0.0}
    if hit == 'WIN':
        s['pairs'][symbol]['win'] += 1
    else:
        s['pairs'][symbol]['loss'] += 1
    s['pairs'][symbol]['pnl'] += pnl

# ─────────────────────────────────────────────
# 💬 TELEGRAM
# ─────────────────────────────────────────────
def send_tg(text, chat_id=None):
    if not chat_id:
        chat_id = TG_CHAT_ID
    try:
        requests.post(
            f'https://api.telegram.org/bot{TG_TOKEN}/sendMessage',
            json={'chat_id': chat_id, 'text': text, 'parse_mode': 'Markdown'},
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

    conf_emoji = {
        'VERY_HIGH': '🔥🔥🔥',
        'HIGH':      '🔥🔥',
        'MEDIUM':    '🔥',
        'LOW':       '⚠️'
    }.get(ai['conf_level'], '🔥')

    signals[sig_id] = {
        'id': sig_id, 'symbol': symbol,
        'dir': ai['direction'],
        'entry': ai['entry'], 'sl': ai['sl'], 'tp': ai['tp'],
        'leverage': ai['leverage'], 'size': size,
        'status': 'OPEN',
        'time': datetime.now(WIB).strftime('%d/%m/%Y %H:%M:%S'),
        'open_ts': time.time(), 'pnl': 0.0
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
        f"🤖 *Analisa AI [{ai['model']}]:*\n"
        f"_{ai['reasoning']}_\n\n"
        f"📍 *Level Kunci:* {ai['key_levels']}\n"
        f"⚠️ *Risk:* {ai['risk_note']}\n"
        f"━━━━━━━━━━━━━━\n"
        f"{conf_emoji} *Confidence: {ai['confidence']}%* ({ai['conf_level']})\n"
        f"⚡ *Leverage: {ai['leverage']}x*\n"
        f"⚖️ *R/R: 1:{rr:.1f}*\n"
        f"━━━━━━━━━━━━━━\n"
        f"💵 Modal: ${size:.1f} x{ai['leverage']}x\n"
        f"✅ Pot.Profit: +${pot_profit:.2f} (+Rp{profit_idr:,})\n"
        f"❌ Pot.Loss:   -${pot_loss:.2f} (-Rp{loss_idr:,})\n"
        f"⏰ {datetime.now(WIB).strftime('%d/%m/%Y %H:%M:%S')}"
    )

# ─────────────────────────────────────────────
# 📉 MONITOR TP/SL
# ─────────────────────────────────────────────
def monitor_positions():
    closed = []
    for sig_id, sig in signals.items():
        if sig['status'] != 'OPEN':
            continue

        symbol = sig['symbol']
        cache  = price_cache.get(symbol)
        if not cache:
            continue

        hit = None
        if sig['dir'] == 'LONG':
            if cache['h'] >= sig['tp']:  hit = 'WIN'
            elif cache['l'] <= sig['sl']: hit = 'LOSS'
        else:
            if cache['l'] <= sig['tp']:  hit = 'WIN'
            elif cache['h'] >= sig['sl']: hit = 'LOSS'

        if not hit:
            continue

        size = sig['size']
        lev  = sig['leverage']
        if hit == 'WIN':
            pnl = abs((sig['tp'] - sig['entry']) / sig['entry'] * size * lev)
        else:
            pnl = -abs((sig['sl'] - sig['entry']) / sig['entry'] * size * lev)

        pnl_idr  = int(abs(pnl) * USD_TO_IDR)
        dur_s    = int(time.time() - sig['open_ts'])
        dur_str  = f"{dur_s//3600}j {(dur_s%3600)//60}m" if dur_s >= 3600 else f"{dur_s//60}m {dur_s%60}s"
        exit_t   = datetime.now(WIB).strftime('%d/%m/%Y %H:%M:%S')

        sig['status']    = hit
        sig['pnl']       = pnl
        sig['exit_time'] = exit_t
        update_stats(hit, pnl, symbol)
        closed.append(sig_id)

        e = '✅' if hit == 'WIN' else '❌'
        send_tg(
            f"{e} *{hit} — {symbol}*\n"
            f"━━━━━━━━━━━━━━\n"
            f"📍 {sig['dir']} @ ${sig['entry']:.4f}\n"
            f"📌 Exit @ ${sig['tp'] if hit=='WIN' else sig['sl']:.4f}\n"
            f"⚡ Leverage: {lev}x\n"
            f"━━━━━━━━━━━━━━\n"
            f"{'💰' if pnl>0 else '💸'} PnL: *{'+'if pnl>=0 else ''}"
            f"${pnl:.2f}* ({'+'if pnl>=0 else '-'}Rp{pnl_idr:,})\n"
            f"⏱️ Durasi: {dur_str}\n"
            f"⏰ {exit_t}\n"
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
    pnl_idr = int(abs(pnl) * USD_TO_IDR)
    pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
    idr_str = f"+Rp{pnl_idr:,}" if pnl >= 0 else f"-Rp{pnl_idr:,}"

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
        f"🤖 Powered by OpenRouter AI\n"
        f"📅 {get_today()}\n"
        f"━━━━━━━━━━━━━━\n"
        f"📡 Total Sinyal: *{stats.get('total_signal', 0)}*\n"
        f"✅ Win: *{stats['win']}*\n"
        f"❌ Loss: *{stats['loss']}*\n"
        f"⏳ Open: *{open_c}*\n"
        f"📈 Closed: *{total}*\n"
        f"🏆 Win Rate: *{wr}%*\n"
        f"━━━━━━━━━━━━━━\n"
        f"💰 *DETAIL PnL:*\n"
        f"💵 Modal: Rp{MODAL_IDR:,}\n"
        f"📈 Profit: *+${profit:.2f}* (+Rp{int(profit*USD_TO_IDR):,})\n"
        f"📉 Loss:   *-${loss:.2f}* (-Rp{int(loss*USD_TO_IDR):,})\n"
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
    curr      = get_stats()
    yesterday = (datetime.now(WIB) - timedelta(days=1)).strftime('%d/%m/%Y')
    snap      = snapshot_21.get(yesterday, {
        'win': 0, 'loss': 0, 'total_profit': 0,
        'total_loss': 0, 'pnl': 0, 'total_signal': 0, 'pairs': {}
    })
    delta = {
        'win':          curr['win']          - snap.get('win', 0),
        'loss':         curr['loss']         - snap.get('loss', 0),
        'total_profit': curr['total_profit'] - snap.get('total_profit', 0),
        'total_loss':   curr['total_loss']   - snap.get('total_loss', 0),
        'pnl':          curr['pnl']          - snap.get('pnl', 0),
        'total_signal': curr['total_signal'] - snap.get('total_signal', 0),
        'pairs':        curr.get('pairs', {})
    }
    send_tg(build_rekap(delta, 'REKAP SEMENTARA',
                        f"Sejak {yesterday} 21.00 → Sekarang"), chat_id)

# ─────────────────────────────────────────────
# 🔁 LOOP SCANNER
# ─────────────────────────────────────────────
def scanner_loop():
    pair_idx       = 0
    last_rekap_day = None

    print(f'[{datetime.now(WIB)}] AI Scanner dimulai — {len(PAIRS)} pairs')
    send_tg(
        f"🤖 *AI Scanner Aktif!*\n\n"
        f"⚙️ *Setting:*\n"
        f"• Engine: *OpenRouter AI (Free)*\n"
        f"• Models: Llama3 / Mistral / Gemma\n"
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

            # Rekap otomatis jam 21:00
            if now.hour == 21 and now.minute == 0:
                today = now.strftime('%d/%m/%Y')
                if last_rekap_day != today:
                    rekap_harian()
                    last_rekap_day = today

            symbol = PAIRS[pair_idx % len(PAIRS)]
            try:
                klines = fetch_klines(symbol)
                last   = klines[-1]

                # Update price cache untuk monitor TP/SL
                price_cache[symbol] = {
                    'h': last['h'], 'l': last['l'],
                    'c': last['c'], 't': time.time()
                }

                data = prepare_data(klines, symbol)
                ai   = analyze_with_ai(data)

                if ai is None:
                    print(f"[{now.strftime('%H:%M:%S')}] {symbol}: AI gagal, skip")
                else:
                    print(
                        f"[{now.strftime('%H:%M:%S')}] {symbol}: "
                        f"{ai['direction']} conf={ai['confidence']}% lev={ai['leverage']}x"
                    )

                    if ai['direction'] in ('LONG', 'SHORT'):
                        # Cek belum ada posisi terbuka di pair ini
                        existing = any(
                            s['symbol'] == symbol and s['status'] == 'OPEN'
                            for s in signals.values()
                        )
                        if not existing:
                            send_signal(symbol, ai)

            except Exception as e:
                print(f"[{now.strftime('%H:%M:%S')}] {symbol} error: {e}")

            pair_idx += 1
            monitor_positions()

        except Exception as e:
            print(f'[{datetime.now(WIB)}] Loop error: {e}')

        time.sleep(SCAN_INTERVAL)

# ─────────────────────────────────────────────
# 🌐 FLASK ROUTES
# ─────────────────────────────────────────────
@app.route('/')
def home():
    s  = get_stats()
    oc = sum(1 for x in signals.values() if x['status'] == 'OPEN')
    t  = s['win'] + s['loss']
    return {
        'status': 'running',
        'engine': 'OpenRouter AI (Free)',
        'pairs':  len(PAIRS),
        'time':   datetime.now(WIB).strftime('%d/%m/%Y %H:%M:%S WIB'),
        'today': {
            'signals':  s['total_signal'],
            'win':      s['win'],
            'loss':     s['loss'],
            'open':     oc,
            'win_rate': f"{s['win']/t*100:.1f}%" if t > 0 else '0%',
            'pnl_usd':  round(s['pnl'], 2),
            'pnl_idr':  int(s['pnl'] * USD_TO_IDR)
        }
    }

@app.route('/rekap')
def rekap_ep():
    rekap_manual()
    return {'status': 'sent'}

@app.route('/status')
def status_ep():
    op = [
        {
            'symbol': s['symbol'], 'dir': s['dir'],
            'entry': s['entry'], 'leverage': s['leverage'],
            'sl': s['sl'], 'tp': s['tp']
        }
        for s in signals.values() if s['status'] == 'OPEN'
    ]
    return {'open_positions': op, 'count': len(op)}

@app.route(f'/webhook/{TG_TOKEN}', methods=['POST'])
def webhook():
    data = request.get_json(silent=True)
    if not data:
        return 'ok'
    msg  = data.get('message', {})
    if not msg:
        return 'ok'
    text    = msg.get('text', '').strip()
    chat_id = str(msg.get('chat', {}).get('id', ''))

    if text == '/rekap':
        threading.Thread(target=rekap_manual, args=(chat_id,), daemon=True).start()

    elif text == '/status':
        oc = sum(1 for s in signals.values() if s['status'] == 'OPEN')
        s  = get_stats()
        t  = s['win'] + s['loss']
        send_tg(
            f"📈 *STATUS AI SCANNER*\n\n"
            f"🤖 Engine: OpenRouter AI\n"
            f"🟢 Scanner: Aktif\n"
            f"⏳ Open: {oc}\n"
            f"📡 Sinyal: {s['total_signal']}\n"
            f"✅ Win: {s['win']} | ❌ Loss: {s['loss']}\n"
            f"🏆 WR: {f'{s[chr(119)]*100/t:.1f}%' if t > 0 else '0%'}\n"
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
            "🤖 _Powered by OpenRouter AI (Free)_\n"
            f"📊 Models: Llama3 / Mistral / Gemma",
            chat_id
        )

    return 'ok'

# ─────────────────────────────────────────────
# 🚀 STARTUP
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print('[SCANNER] Starting...')
    t = threading.Thread(target=scanner_loop, daemon=True)
    t.start()
    print('[SCANNER] Thread started!')
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, threaded=True, use_reloader=False)
else:
    # Untuk gunicorn
    t = threading.Thread(target=scanner_loop, daemon=True)
    t.start()
    print('[SCANNER] Thread started via gunicorn!')
