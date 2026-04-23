"""
============================================================
CRYPTO BACKTEST SCANNER - PYTHON untuk RENDER
============================================================
✅ Data dari Binance public API (gratis, tanpa login)
✅ Scan 30 pairs tiap 30 detik
✅ Monitor TP/SL real-time
✅ Sinyal + hasil WIN/LOSS ke Telegram
✅ Rekap harian jam 21.00 WIB
✅ Rekap manual via Telegram command
✅ Modal Rp23jt, leverage 5x, risk 2%
============================================================
"""

import os
import time
import json
import hmac
import hashlib
import requests
import threading
import numpy as np
from datetime import datetime, timezone, timedelta
from flask import Flask, request

# ─────────────────────────────────────────────
# ⚙️ KONFIGURASI
# ─────────────────────────────────────────────
TG_TOKEN   = os.environ.get('TG_TOKEN',   'ISI_TOKEN_TELEGRAM')
TG_CHAT_ID = os.environ.get('TG_CHAT_ID', 'ISI_CHAT_ID_TELEGRAM')
TIMEFRAME  = '15m'
LEVERAGE   = 5
TRADE_SIZE = 27.6       # 2% dari modal $1380
MODAL_IDR  = 23_000_000
USD_TO_IDR = 16_300
SCAN_INTERVAL = 30      # detik antar scan
WIB = timezone(timedelta(hours=7))

PAIRS = [
    # Top Market Cap
    'BTCUSDT','ETHUSDT','BNBUSDT','SOLUSDT','XRPUSDT',
    # Mid Cap
    'DOGEUSDT','ADAUSDT','AVAXUSDT','LINKUSDT','DOTUSDT',
    # Layer 2
    'MATICUSDT','ARBUSDT','OPUSDT','APTUSDT','SUIUSDT',
    # DeFi & Others
    'LTCUSDT','NEARUSDT','ATOMUSDT','INJUSDT','SEIUSDT',
    # Trending
    'FETUSDT','WLDUSDT','JUPUSDT','DYMUSDT','TIAUSDT',
    # Tambahan
    'AAVEUSDT','UNIUSDT','MKRUSDT','LDOUSDT','COMPUSDT'
]

# ─────────────────────────────────────────────
# 💾 STORAGE (in-memory, reset saat restart)
# ─────────────────────────────────────────────
signals    = {}   # posisi aktif
daily_stats = {}  # statistik harian
snapshot_21 = {}  # snapshot jam 21.00

# ─────────────────────────────────────────────
# 📡 FETCH DATA BINANCE
# ─────────────────────────────────────────────
def fetch_klines(symbol, interval='15m', limit=100):
    coin = symbol.replace('USDT', '')
    kraken_map = {
        'BTC': 'XBT', 'DOGE': 'XDG', 'MATIC': 'POL',
        'WLD': 'WLD', 'JUP': 'JUP', 'DYM': 'DYM',
        'TIA': 'TIA', 'FET': 'FET', 'LDO': 'LDO',
        'SEI': 'SEI', 'SUI': 'SUI', 'APT': 'APT',
        'ARB': 'ARB', 'OP': 'OP', 'COMP': 'COMP',
        'MKR': 'MKR', 'UNI': 'UNI', 'AAVE': 'AAVE'
    }
    coin = kraken_map.get(coin, coin)
    url = f'https://api.kraken.com/0/public/OHLC'
    params = {'pair': f'{coin}USD', 'interval': 15}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    if data.get('error'):
        raise Exception(f"Kraken error: {data['error']}")
    keys = [k for k in data['result'].keys() if k != 'last']
    if not keys:
        raise Exception('No data from Kraken')
    raw = data['result'][keys[0]]
    return [{
        'o': float(k[1]),
        'h': float(k[2]),
        'l': float(k[3]),
        'c': float(k[4]),
        'v': float(k[6])
    } for k in raw]

# ─────────────────────────────────────────────
# 📊 INDIKATOR
# ─────────────────────────────────────────────
def calc_ema(closes, n):
    closes = np.array(closes)
    k = 2 / (n + 1)
    ema = np.mean(closes[:n])
    for price in closes[n:]:
        ema = price * k + ema * (1 - k)
    return ema

def calc_rsi(closes, n=14):
    closes = np.array(closes)
    deltas = np.diff(closes[-(n+2):])
    gains  = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    ag = np.mean(gains[:n])
    al = np.mean(losses[:n])
    if al == 0:
        return 100.0
    return 100 - (100 / (1 + ag / al))

def calc_bb(closes, n=20):
    sl  = np.array(closes[-n:])
    avg = np.mean(sl)
    std = np.std(sl)
    return avg + 2*std, avg - 2*std  # upper, lower

def calc_vol_spike(volumes):
    vols = np.array(volumes)
    avg  = np.mean(vols[-20:-1])
    cur  = vols[-1]
    return cur / avg if avg > 0 else 1.0, cur > avg * 1.5

def calc_atr(klines, n=14):
    trs = []
    for i in range(-n, 0):
        h    = klines[i]['h']
        l    = klines[i]['l']
        prev = klines[i-1]['c']
        trs.append(max(h-l, abs(h-prev), abs(l-prev)))
    return np.mean(trs)

def calc_macd(closes, fast=12, slow=26, signal=9):
    ema_fast = calc_ema(closes, fast)
    ema_slow = calc_ema(closes, slow)
    macd     = ema_fast - ema_slow
    return macd

# ─────────────────────────────────────────────
# 🧠 ANALISIS SINYAL
# ─────────────────────────────────────────────
def analyze(klines, symbol):
    closes  = [k['c'] for k in klines]
    volumes = [k['v'] for k in klines]
    last    = closes[-1]

    rsi          = calc_rsi(closes)
    upper, lower = calc_bb(closes)
    ema20        = calc_ema(closes, 20)
    ema50        = calc_ema(closes, 50)
    macd         = calc_macd(closes)
    vol_ratio, vol_spike = calc_vol_spike(volumes)
    atr          = calc_atr(klines)

    score = 0
    why   = []

    # RSI
    if rsi < 35:
        score += 1
        why.append(f'RSI oversold {rsi:.0f}')
    elif rsi > 65:
        score -= 1
        why.append(f'RSI overbought {rsi:.0f}')

    # Bollinger Bands
    if last <= lower:
        score += 1
        why.append('Lower BB')
    elif last >= upper:
        score -= 1
        why.append('Upper BB')

    # EMA Trend
    if ema20 > ema50:
        score += 1
        why.append('Uptrend EMA')
    else:
        score -= 1
        why.append('Downtrend EMA')

    # MACD
    if macd > 0:
        score += 1
        why.append('MACD bullish')
    else:
        score -= 1
        why.append('MACD bearish')

    # Volume Spike (bobot ganda)
    if vol_spike:
        if score > 0:
            score += 1
            why.append(f'Vol spike {vol_ratio:.1f}x')
        elif score < 0:
            score -= 1
            why.append(f'Vol spike {vol_ratio:.1f}x')

    direction = 'LONG' if score >= 3 else ('SHORT' if score <= -3 else None)
    sl = (last - atr * 1.5) if direction == 'LONG' else (last + atr * 1.5)
    tp = (last + atr * 2.5) if direction == 'LONG' else (last - atr * 2.5)

    return {
        'symbol':    symbol,
        'direction': direction,
        'score':     score,
        'price':     last,
        'rsi':       rsi,
        'ema20':     ema20,
        'ema50':     ema50,
        'macd':      macd,
        'vol_ratio': vol_ratio,
        'vol_spike': vol_spike,
        'sl':        sl,
        'tp':        tp,
        'atr':       atr,
        'why':       why,
        'time':      datetime.now(WIB).strftime('%d/%m/%Y %H:%M:%S')
    }

# ─────────────────────────────────────────────
# 💬 TELEGRAM
# ─────────────────────────────────────────────
def send_tg(text, chat_id=None):
    if not chat_id:
        chat_id = TG_CHAT_ID
    try:
        url = f'https://api.telegram.org/bot{TG_TOKEN}/sendMessage'
        requests.post(url, json={
            'chat_id':    chat_id,
            'text':       text,
            'parse_mode': 'Markdown'
        }, timeout=10)
    except Exception as e:
        print(f'TG error: {e}')

# ─────────────────────────────────────────────
# 💾 STATISTIK HARIAN
# ─────────────────────────────────────────────
def get_today():
    return datetime.now(WIB).strftime('%d/%m/%Y')

def get_stats():
    today = get_today()
    if today not in daily_stats:
        daily_stats[today] = {
            'win': 0, 'loss': 0,
            'total_profit': 0.0, 'total_loss': 0.0,
            'pnl': 0.0, 'total_signal': 0,
            'pairs': {}
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
# 📨 KIRIM SINYAL KE TELEGRAM
# ─────────────────────────────────────────────
def send_signal(sig):
    sig_id = f"{sig['symbol']}_{int(time.time())}"
    pot_profit = abs((sig['tp'] - sig['price']) / sig['price'] * TRADE_SIZE * LEVERAGE)
    pot_loss   = abs((sig['sl'] - sig['price']) / sig['price'] * TRADE_SIZE * LEVERAGE)
    profit_idr = int(pot_profit * USD_TO_IDR)
    loss_idr   = int(pot_loss   * USD_TO_IDR)
    rr         = pot_profit / pot_loss if pot_loss > 0 else 0

    # Simpan sinyal
    signals[sig_id] = {
        'id':      sig_id,
        'symbol':  sig['symbol'],
        'dir':     sig['direction'],
        'entry':   sig['price'],
        'sl':      sig['sl'],
        'tp':      sig['tp'],
        'status':  'OPEN',
        'time':    sig['time'],
        'open_ts': time.time(),
        'pnl':     0.0
    }

    # Update total signal
    get_stats()['total_signal'] += 1

    e = '🟢' if sig['direction'] == 'LONG' else '🔴'
    trend = 'Uptrend ▲' if sig['ema20'] > sig['ema50'] else 'Downtrend ▼'

    send_tg(
        f"{e} *{sig['direction']} SIGNAL*\n"
        f"━━━━━━━━━━━━━━\n"
        f"🔹 *{sig['symbol']}*\n"
        f"💹 *Harga: ${sig['price']:.4f}*\n"
        f"━━━━━━━━━━━━━━\n"
        f"💰 Entry: ${sig['price']:.4f}\n"
        f"🛑 SL: ${sig['sl']:.4f}\n"
        f"🎯 TP: ${sig['tp']:.4f}\n"
        f"━━━━━━━━━━━━━━\n"
        f"📊 RSI: {sig['rsi']:.1f}\n"
        f"📈 EMA: {trend}\n"
        f"📉 MACD: {'▲ Bullish' if sig['macd'] > 0 else '▼ Bearish'}\n"
        f"🔥 Vol: {sig['vol_ratio']:.1f}x{'🔥' if sig['vol_spike'] else ''}\n"
        f"💪 Score: {sig['score']}/5\n"
        f"━━━━━━━━━━━━━━\n"
        f"💵 Modal: ${TRADE_SIZE} x{LEVERAGE}x\n"
        f"✅ Pot.Profit: +${pot_profit:.2f} (+Rp{profit_idr:,})\n"
        f"❌ Pot.Loss:   -${pot_loss:.2f} (-Rp{loss_idr:,})\n"
        f"⚖️ R/R: 1:{rr:.1f}\n"
        f"━━━━━━━━━━━━━━\n"
        f"📝 {', '.join(sig['why'])}\n"
        f"⏰ {sig['time']}\n"
        f"⏳ _Memantau TP/SL real-time..._"
    )

# ─────────────────────────────────────────────
# 🔍 MONITOR SEMUA POSISI REAL-TIME
# ─────────────────────────────────────────────
def monitor_positions(price_cache):
    now = datetime.now(WIB)
    for sig_id, d in list(signals.items()):
        if d['status'] != 'OPEN':
            continue

        # Ambil harga dari cache
        symbol = d['symbol']
        if symbol not in price_cache:
            continue

        pc  = price_cache[symbol]
        hit = None

        if d['dir'] == 'LONG':
            if pc['h'] >= d['tp']: hit = 'WIN'
            if pc['l'] <= d['sl']: hit = 'LOSS'
        else:
            if pc['l'] <= d['tp']: hit = 'WIN'
            if pc['h'] >= d['sl']: hit = 'LOSS'

        if hit:
            exit_price = d['tp'] if hit == 'WIN' else d['sl']
            diff = (exit_price - d['entry']) if d['dir'] == 'LONG' else (d['entry'] - exit_price)
            pnl  = (diff / d['entry']) * TRADE_SIZE * LEVERAGE
            pnl_abs = abs(pnl)
            pnl_idr = int(pnl_abs * USD_TO_IDR)

            # Hitung durasi
            dur_sec = int(time.time() - d['open_ts'])
            dur_str = f"{dur_sec//3600}j {(dur_sec%3600)//60}m" if dur_sec >= 3600 else f"{dur_sec//60} menit"

            # Update sinyal
            signals[sig_id]['status']    = hit
            signals[sig_id]['exit']      = exit_price
            signals[sig_id]['pnl']       = pnl
            signals[sig_id]['exit_time'] = now.strftime('%d/%m/%Y %H:%M:%S')

            # Update statistik
            update_stats(hit, pnl, symbol)

            emoji  = '✅' if hit == 'WIN' else '❌'
            pnl_str     = f"+${pnl_abs:.2f}" if pnl >= 0 else f"-${pnl_abs:.2f}"
            pnl_idr_str = f"+Rp{pnl_idr:,}" if pnl >= 0 else f"-Rp{pnl_idr:,}"

            send_tg(
                f"{emoji} *HASIL: {hit}*\n"
                f"━━━━━━━━━━━━━━\n"
                f"🔹 *{symbol}* — {d['dir']}\n"
                f"💹 Harga Sekarang: ${pc['c']:.4f}\n"
                f"📍 Entry: ${d['entry']:.4f}\n"
                f"🚪 Exit: ${exit_price:.4f}\n"
                f"━━━━━━━━━━━━━━\n"
                f"💵 PnL: *{pnl_str}* ({pnl_idr_str})\n"
                f"⏱️ Durasi: {dur_str}\n"
                f"⏰ {signals[sig_id]['exit_time']}\n"
                f"━━━━━━━━━━━━━━\n"
                + ("🎉 _Take profit tercapai!_" if hit == 'WIN' else "⚠️ _Stop loss terkena. Next!_")
            )

# ─────────────────────────────────────────────
# 📊 BUILD PESAN REKAP
# ─────────────────────────────────────────────
def build_rekap(stats, title, since_str=''):
    total  = stats['win'] + stats['loss']
    wr     = f"{stats['win']/total*100:.1f}" if total > 0 else '0'
    profit = stats['total_profit']
    loss   = stats['total_loss']
    pnl    = stats['pnl']
    open_c = sum(1 for s in signals.values() if s['status'] == 'OPEN')

    profit_idr = int(profit * USD_TO_IDR)
    loss_idr   = int(loss   * USD_TO_IDR)
    pnl_idr    = int(abs(pnl) * USD_TO_IDR)
    pnl_str     = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
    pnl_idr_str = f"+Rp{pnl_idr:,}" if pnl >= 0 else f"-Rp{pnl_idr:,}"

    # Best & worst pair
    best = worst = '-'
    best_wr = -1
    worst_wr = 101
    for sym, p in stats['pairs'].items():
        pt = p['win'] + p['loss']
        if pt == 0:
            continue
        pwr = p['win'] / pt * 100
        if pwr > best_wr:
            best_wr = pwr
            best    = f"{sym} ({p['win']}W/{p['loss']}L)"
        if pwr < worst_wr:
            worst_wr = pwr
            worst    = f"{sym} ({p['win']}W/{p['loss']}L)"

    return (
        f"📊 *{title}*\n"
        + (f"_{since_str}_\n" if since_str else '')
        + f"━━━━━━━━━━━━━━\n"
        f"📅 {get_today()}\n"
        f"━━━━━━━━━━━━━━\n"
        f"📡 Total Sinyal: *{stats['total_signal']}*\n"
        f"✅ Win: *{stats['win']}*\n"
        f"❌ Loss: *{stats['loss']}*\n"
        f"⏳ Open: *{open_c}*\n"
        f"📈 Closed: *{total}*\n"
        f"🏆 Win Rate: *{wr}%*\n"
        f"━━━━━━━━━━━━━━\n"
        f"💰 *DETAIL PnL:*\n"
        f"💵 Modal: ${TRADE_SIZE} x{LEVERAGE}x\n"
        f"📈 Profit: *+${profit:.2f}* (+Rp{profit_idr:,})\n"
        f"📉 Loss:   *-${loss:.2f}* (-Rp{loss_idr:,})\n"
        f"💹 Net: *{pnl_str}* ({pnl_idr_str})\n"
        f"━━━━━━━━━━━━━━\n"
        f"🔥 Best: {best}\n"
        f"💀 Worst: {worst}\n"
        f"━━━━━━━━━━━━━━\n"
        f"⚠️ _Simulasi modal Rp{MODAL_IDR:,}, bukan trading real_"
    )

# ─────────────────────────────────────────────
# 📊 REKAP HARIAN JAM 21.00
# ─────────────────────────────────────────────
def rekap_harian():
    today = get_today()
    stats = get_stats()
    send_tg(build_rekap(stats, 'REKAP BACKTEST HARIAN'))
    # Simpan snapshot
    snapshot_21[today] = json.loads(json.dumps(stats))
    print(f'[{datetime.now(WIB)}] Rekap harian terkirim')

# ─────────────────────────────────────────────
# 📊 REKAP MANUAL (data dari jam 21.00 kemarin)
# ─────────────────────────────────────────────
def rekap_manual(chat_id=None):
    today = get_today()
    curr  = get_stats()

    # Cari snapshot kemarin
    yesterday = (datetime.now(WIB) - timedelta(days=1)).strftime('%d/%m/%Y')
    snap = snapshot_21.get(yesterday, {
        'win':0,'loss':0,'total_profit':0,'total_loss':0,
        'pnl':0,'total_signal':0,'pairs':{}
    })

    # Hitung selisih
    delta = {
        'win':          curr['win']          - snap.get('win',0),
        'loss':         curr['loss']         - snap.get('loss',0),
        'total_profit': curr['total_profit'] - snap.get('total_profit',0),
        'total_loss':   curr['total_loss']   - snap.get('total_loss',0),
        'pnl':          curr['pnl']          - snap.get('pnl',0),
        'total_signal': curr['total_signal'] - snap.get('total_signal',0),
        'pairs':        curr['pairs']
    }

    since = f"Sejak {yesterday} 21.00 WIB → Sekarang"
    msg   = build_rekap(delta, 'REKAP SEMENTARA', since)
    send_tg(msg, chat_id)

# ─────────────────────────────────────────────
# 🔁 LOOP UTAMA SCANNER
# ─────────────────────────────────────────────
def scanner_loop():
    price_cache    = {}
    last_rekap_day = None
    pair_idx       = 0
    batch_size     = 6

    print(f'[{datetime.now(WIB)}] Scanner dimulai — {len(PAIRS)} pairs, batch {batch_size}')
    send_tg(
        f"✅ *Scanner Aktif!*\n\n"
        f"⚙️ *Setting:*\n"
        f"• Modal: Rp{MODAL_IDR:,} (~${MODAL_IDR//USD_TO_IDR})\n"
        f"• Per trade: ${TRADE_SIZE} (2%)\n"
        f"• Leverage: {LEVERAGE}x\n"
        f"• Pairs: {len(PAIRS)} pairs\n"
        f"• Interval: {SCAN_INTERVAL} detik\n"
        f"• Data: Binance\n"
        f"• Rekap: Jam 21.00 WIB\n\n"
        f"🚀 Monitoring real-time dimulai!"
    )

    while True:
        try:
            now = datetime.now(WIB)

            # Rekap jam 21.00
            if now.hour == 21 and now.minute == 0:
                today = now.strftime('%d/%m/%Y')
                if last_rekap_day != today:
                    rekap_harian()
                    last_rekap_day = today

            # Scan batch pairs
            for i in range(batch_size):
                pidx = (pair_idx + i) % len(PAIRS)
                pair = PAIRS[pidx]

                try:
                    klines = fetch_klines(pair, TIMEFRAME)
                    last   = klines[-1]

                    # Update price cache
                    price_cache[pair] = {
                        'h': last['h'],
                        'l': last['l'],
                        'c': last['c'],
                        't': time.time()
                    }

                    # Analisis sinyal
                    sig = analyze(klines, pair)
                    if sig['direction']:
                        # Cek apakah sudah ada sinyal open untuk pair ini
                        existing = any(
                            s['symbol'] == pair and s['status'] == 'OPEN'
                            for s in signals.values()
                        )
                        if not existing:
                            print(f"[{now}] SINYAL: {pair} {sig['direction']} score={sig['score']}")
                            send_signal(sig)
                    else:
                        print(f"[{now}] {pair}: no signal score={sig['score']}")

                except Exception as e:
                    print(f"[{now}] {pair} error: {e}")

                time.sleep(0.5)

            pair_idx = (pair_idx + batch_size) % len(PAIRS)

            # Monitor semua posisi dengan price cache
            monitor_positions(price_cache)

        except Exception as e:
            print(f'[{datetime.now(WIB)}] Loop error: {e}')

        time.sleep(SCAN_INTERVAL)

# ─────────────────────────────────────────────
# 🌐 FLASK WEB SERVER (untuk Render + UptimeRobot)
# ─────────────────────────────────────────────
app = Flask(__name__)

# Start scanner saat module diload gunicorn
_scanner_started = False
def start_scanner():
    global _scanner_started
    if not _scanner_started:
        _scanner_started = True
        t = threading.Thread(target=scanner_loop, daemon=True)
        t.start()
        print('[SCANNER] Thread started successfully!')

start_scanner()

@app.route('/')
def home():
    stats  = get_stats()
    open_c = sum(1 for s in signals.values() if s['status'] == 'OPEN')
    total  = stats['win'] + stats['loss']
    wr     = f"{stats['win']/total*100:.1f}%" if total > 0 else '0%'
    return {
        'status':       'running',
        'pairs':        len(PAIRS),
        'time':         datetime.now(WIB).strftime('%d/%m/%Y %H:%M:%S WIB'),
        'today': {
            'signals':  stats['total_signal'],
            'win':      stats['win'],
            'loss':     stats['loss'],
            'open':     open_c,
            'win_rate': wr,
            'pnl_usd':  round(stats['pnl'], 2),
            'pnl_idr':  int(stats['pnl'] * USD_TO_IDR)
        }
    }

@app.route('/rekap')
def rekap_endpoint():
    rekap_manual()
    return {'status': 'sent'}

@app.route('/status')
def status():
    open_sigs = [
        {
            'symbol': s['symbol'],
            'dir':    s['dir'],
            'entry':  s['entry'],
            'sl':     s['sl'],
            'tp':     s['tp'],
            'time':   s['time']
        }
        for s in signals.values()
        if s['status'] == 'OPEN'
    ]
    return {'open_positions': open_sigs, 'count': len(open_sigs)}

# Webhook Telegram untuk command /rekap
@app.route(f'/webhook/{TG_TOKEN}', methods=['POST'])
def webhook():
    data = request.json
    if not data:
        return 'ok'

    msg = data.get('message', {})
    if not msg:
        return 'ok'

    text    = msg.get('text', '')
    chat_id = str(msg.get('chat', {}).get('id', ''))

    if text == '/rekap':
        threading.Thread(target=rekap_manual, args=(chat_id,)).start()
    elif text == '/status':
        open_c = sum(1 for s in signals.values() if s['status'] == 'OPEN')
        stats  = get_stats()
        send_tg(
            f"📈 *STATUS SCANNER*\n\n"
            f"🟢 Scanner: Aktif\n"
            f"⏳ Posisi open: {open_c}\n"
            f"📡 Sinyal hari ini: {stats['total_signal']}\n"
            f"✅ Win: {stats['win']}\n"
            f"❌ Loss: {stats['loss']}\n"
            f"💹 PnL: ${stats['pnl']:.2f}\n"
            f"⏰ {datetime.now(WIB).strftime('%d/%m/%Y %H:%M:%S WIB')}",
            chat_id
        )
    elif text == '/help':
        send_tg(
            "📋 *COMMAND TERSEDIA:*\n\n"
            "/rekap — Rekap sementara sejak 21.00\n"
            "/status — Status scanner sekarang\n"
            "/help — Tampilkan bantuan",
            chat_id
        )

    return 'ok'

# ─────────────────────────────────────────────
# 🚀 MAIN
# ─────────────────────────────────────────────
if __name__ == '__main__':
    # Jalankan scanner di thread terpisah
    scanner_thread = threading.Thread(target=scanner_loop, daemon=True)
    scanner_thread.start()

    # Jalankan Flask web server
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
