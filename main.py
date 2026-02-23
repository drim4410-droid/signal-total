import os
import math
import time
import sqlite3
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional, Tuple

import requests
from dotenv import load_dotenv

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
)

# =========================
# CONFIG
# =========================
load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
ADMIN_ID = int(os.getenv("ADMIN_ID", "0").strip() or "0")

BINGX_BASE_URL = os.getenv("BINGX_BASE_URL", "https://open-api.bingx.com").rstrip("/")
SCAN_INTERVAL_MIN = int(os.getenv("SCAN_INTERVAL_MIN", "15"))
AUTO_SCAN_DEFAULT = os.getenv("AUTO_SCAN", "1").strip() == "1"

# Таймфреймы (твои требования)
TIMEFRAMES = ["5m", "15m", "1h", "4h"]

# Сколько монет:
TOP200 = 200
TOP20 = 20

# Порог “сильного сигнала”
MIN_SCORE_FOR_ALERT = 78  # можно потом подстроить

# =========================
# LOGGING
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("bingxbot")

# =========================
# DB
# =========================
DB_PATH = "bot.db"


def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db() -> None:
    with db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY,
                username TEXT,
                status TEXT NOT NULL DEFAULT 'pending', -- pending/active/blocked
                expires_at TEXT, -- ISO datetime UTC
                auto_scan INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                tf TEXT NOT NULL,
                entry REAL,
                sl REAL,
                tp REAL,
                score INTEGER NOT NULL,
                reason TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS settings (
                k TEXT PRIMARY KEY,
                v TEXT NOT NULL
            )
            """
        )


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def parse_iso(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def upsert_user(user_id: int, username: str) -> None:
    with db() as conn:
        conn.execute(
            """
            INSERT INTO users(user_id, username, status, expires_at, auto_scan, created_at)
            VALUES (?, ?, 'pending', NULL, 1, ?)
            ON CONFLICT(user_id) DO UPDATE SET username=excluded.username
            """,
            (user_id, username or "", iso(now_utc())),
        )


def get_user(user_id: int) -> Optional[Dict[str, Any]]:
    with db() as conn:
        cur = conn.execute(
            "SELECT user_id, username, status, expires_at, auto_scan, created_at FROM users WHERE user_id=?",
            (user_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return {
            "user_id": row[0],
            "username": row[1],
            "status": row[2],
            "expires_at": row[3],
            "auto_scan": int(row[4]),
            "created_at": row[5],
        }


def set_user_access(user_id: int, days: int) -> None:
    exp = now_utc() + timedelta(days=days)
    with db() as conn:
        conn.execute(
            "UPDATE users SET status='active', expires_at=?, auto_scan=1 WHERE user_id=?",
            (iso(exp), user_id),
        )


def block_user(user_id: int) -> None:
    with db() as conn:
        conn.execute("UPDATE users SET status='blocked' WHERE user_id=?", (user_id,))


def is_user_active(u: Dict[str, Any]) -> Tuple[bool, str]:
    if u["status"] != "active":
        return False, "⛔ Доступ не активирован."
    exp = parse_iso(u["expires_at"])
    if not exp:
        return False, "⛔ Доступ не активирован (нет даты окончания)."
    if now_utc() >= exp:
        # протух
        block_user(u["user_id"])
        return False, "⛔ Подписка истекла. Напиши администратору."
    return True, ""


def user_auto_scan_enabled(user_id: int) -> bool:
    with db() as conn:
        cur = conn.execute("SELECT auto_scan FROM users WHERE user_id=?", (user_id,))
        r = cur.fetchone()
        return bool(r and int(r[0]) == 1)


def set_user_auto_scan(user_id: int, enabled: bool) -> None:
    with db() as conn:
        conn.execute("UPDATE users SET auto_scan=? WHERE user_id=?", (1 if enabled else 0, user_id))


# =========================
# BINGX API (public endpoints)
# =========================
# По списку эндпоинтов BingX:
# - 24h change/tickers: /openApi/swap/v2/quote/ticker
# - contracts list:     /openApi/swap/v2/quote/contracts
# - klines:             /openApi/swap/v3/quote/klines
# (источники: статьи/справка BingX Zendesk) 0

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "bingx-signal-bot/1.0"})


def http_get(path: str, params: Optional[Dict[str, Any]] = None, timeout: int = 12) -> Any:
    url = f"{BINGX_BASE_URL}{path}"
    r = SESSION.get(url, params=params or {}, timeout=timeout)
    r.raise_for_status()
    return r.json()


def get_top_usdt_symbols(limit: int = TOP200) -> List[str]:
    """
    Берём 24h тикеры, сортируем по объёму и берём USDT-символы.
    """
    data = http_get("/openApi/swap/v2/quote/ticker")  # 24h stats 1
    # Ожидаем: {"code":0, "data":[{ "symbol":"BTC-USDT", "volume":"...", ...}] } или похожее
    items = data.get("data") or []
    scored = []
    for it in items:
        sym = str(it.get("symbol", "")).strip()
        if not sym:
            continue
        # фильтр USDT
        if "USDT" not in sym:
            continue
        # отсечём странное
        if "UP" in sym or "DOWN" in sym:
            pass
        vol = it.get("volume") or it.get("volume24h") or it.get("quoteVolume") or "0"
        try:
            v = float(vol)
        except Exception:
            v = 0.0
        scored.append((v, sym))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in scored[:limit]]


def get_klines(symbol: str, interval: str, limit: int = 200) -> List[Dict[str, float]]:
    """
    Kline endpoint: /openApi/swap/v3/quote/klines 2
    Возвращаем список свечей: open, high, low, close, volume.
    """
    # Преобразуем наши TF в то, что обычно ждут (часто: "5m","15m","1h","4h")
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
    }
    data = http_get("/openApi/swap/v3/quote/klines", params=params)
    arr = data.get("data") or []
    out: List[Dict[str, float]] = []
    for row in arr:
        # Часто формат: [time, open, high, low, close, volume]
        if isinstance(row, list) and len(row) >= 6:
            try:
                out.append(
                    {
                        "open": float(row[1]),
                        "high": float(row[2]),
                        "low": float(row[3]),
                        "close": float(row[4]),
                        "volume": float(row[5]),
                    }
                )
            except Exception:
                continue
        elif isinstance(row, dict):
            try:
                out.append(
                    {
                        "open": float(row.get("open")),
                        "high": float(row.get("high")),
                        "low": float(row.get("low")),
                        "close": float(row.get("close")),
                        "volume": float(row.get("volume", 0)),
                    }
                )
            except Exception:
                continue
    return out


# =========================
# INDICATORS
# =========================
def ema(values: List[float], period: int) -> List[float]:
    if not values or period <= 1:
        return values[:]
    k = 2 / (period + 1)
    out = []
    e = values[0]
    out.append(e)
    for v in values[1:]:
        e = v * k + e * (1 - k)
        out.append(e)
    return out


def rsi(values: List[float], period: int = 14) -> float:
    if len(values) < period + 1:
        return 50.0
    gains = 0.0
    losses = 0.0
    for i in range(-period, 0):
        diff = values[i] - values[i - 1]
        if diff >= 0:
            gains += diff
        else:
            losses -= diff
    if losses == 0:
        return 100.0
    rs = gains / losses
    return 100 - (100 / (1 + rs))


def atr(candles: List[Dict[str, float]], period: int = 14) -> float:
    if len(candles) < period + 2:
        return 0.0
    trs = []
    for i in range(1, len(candles)):
        h = candles[i]["high"]
        l = candles[i]["low"]
        pc = candles[i - 1]["close"]
        tr = max(h - l, abs(h - pc), abs(l - pc))
        trs.append(tr)
    last = trs[-period:]
    return sum(last) / len(last) if last else 0.0


# =========================
# SIGNAL LOGIC (trend + breakout + retest-ish)
# =========================
@dataclass
class Signal:
    symbol: str
    direction: str  # LONG/SHORT
    tf: str
    entry: float
    sl: float
    tp: float
    score: int
    reason: str


def compute_signal_for_tf(symbol: str, tf: str) -> Optional[Signal]:
    candles = get_klines(symbol, tf, limit=220)
    if len(candles) < 120:
        return None

    closes = [c["close"] for c in candles]
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]
    vols = [c.get("volume", 0.0) for c in candles]

    e50 = ema(closes, 50)[-1]
    e200 = ema(closes, 200)[-1] if len(closes) >= 200 else ema(closes, 100)[-1]
    last = closes[-1]

    trend_up = e50 > e200
    trend_down = e50 < e200

    r = rsi(closes, 14)
    a = atr(candles, 14)
    if a <= 0:
        return None

    # Breakout: пробой максимума/минимума за N свечей (без последних 2)
    N = 55
    recent_high = max(highs[-(N + 2):-2])
    recent_low = min(lows[-(N + 2):-2])

    # “Ретест” грубо: цена вернулась к уровню в пределах 0.35*ATR после пробоя
    retest_band = 0.35 * a

    # Volume impulse:
    v_now = vols[-1]
    v_avg = sum(vols[-30:]) / 30 if len(vols) >= 30 else (sum(vols) / len(vols))
    vol_boost = 1.0 if v_avg <= 0 else (v_now / v_avg)

    score = 0
    reason_parts = []

    # тренд
    if trend_up:
        score += 20
        reason_parts.append("тренд вверх (EMA50>EMA200)")
    elif trend_down:
        score += 20
        reason_parts.append("тренд вниз (EMA50<EMA200)")

    # импульс объёма
    if vol_boost >= 1.3:
        score += 12
        reason_parts.append(f"объём выше среднего x{vol_boost:.2f}")

    # RSI фильтр (не брать в перекуп/перепрод)
    if 40 <= r <= 65:
        score += 10
        reason_parts.append(f"RSI={r:.1f} (норм)")
    elif r < 30 or r > 75:
        score -= 8
        reason_parts.append(f"RSI={r:.1f} (экстр.)")

    # пробой + ретест
    direction = None
    level = None

    # LONG: пробой вверх и ретест уровня
    if trend_up and last > recent_high:
        score += 22
        reason_parts.append("пробой high")
        direction = "LONG"
        level = recent_high

    if direction == "LONG":
        # проверим “ретест”: минимум последних 5 свечей близко к уровню
        min5 = min(lows[-5:])
        if abs(min5 - level) <= retest_band:
            score += 18
            reason_parts.append("есть ретест уровня")

    # SHORT: пробой вниз и ретест
    if trend_down and last < recent_low:
        score += 22
        reason_parts.append("пробой low")
        direction = "SHORT"
        level = recent_low

    if direction == "SHORT":
        max5 = max(highs[-5:])
        if abs(max5 - level) <= retest_band:
            score += 18
            reason_parts.append("есть ретест уровня")

    if direction is None:
        return None

    entry = last
    # SL/TP через ATR (консервативно)
    if direction == "LONG":
        sl = entry - 1.2 * a
        tp = entry + 2.0 * a
    else:
        sl = entry + 1.2 * a
        tp = entry - 2.0 * a

    # нормализуем
    score = max(0, min(100, int(score)))
    reason = "; ".join(reason_parts)

    return Signal(symbol=symbol, direction=direction, tf=tf, entry=entry, sl=sl, tp=tp, score=score, reason=reason)


def best_signal(symbols: List[str]) -> Optional[Signal]:
    best: Optional[Signal] = None
    for sym in symbols:
        for tf in TIMEFRAMES:
            try:
                sig = compute_signal_for_tf(sym, tf)
            except Exception as e:
                log.warning("signal calc failed %s %s: %s", sym, tf, e)
                continue
            if not sig:
                continue
            if (best is None) or (sig.score > best.score):
                best = sig
    return best


def save_signal(sig: Signal) -> None:
    with db() as conn:
        conn.execute(
            """
            INSERT INTO signals(created_at, symbol, direction, tf, entry, sl, tp, score, reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (iso(now_utc()), sig.symbol, sig.direction, sig.tf, sig.entry, sig.sl, sig.tp, sig.score, sig.reason),
        )


# =========================
# UI (BUTTONS)
# =========================
def kb_main(is_admin: bool, auto_on: bool) -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton("🚀 Новый сигнал (Top20)", callback_data="new_signal")],
        [InlineKeyboardButton(f"🧠 Авто-анализ: {'ON ✅' if auto_on else 'OFF ❌'}", callback_data="toggle_auto")],
        [InlineKeyboardButton("📊 Статус", callback_data="status")],
    ]
    if is_admin:
        rows.append([InlineKeyboardButton("👑 Админ-панель", callback_data="admin")])
    return InlineKeyboardMarkup(rows)


def kb_admin() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("📥 Заявки (pending)", callback_data="admin_pending")],
            [InlineKeyboardButton("📦 Активные пользователи", callback_data="admin_active")],
            [InlineKeyboardButton("⬅️ Назад", callback_data="back")],
        ]
    )


def kb_approve(user_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("✅ 7 дней", callback_data=f"approve:{user_id}:7"),
                InlineKeyboardButton("✅ 15 дней", callback_data=f"approve:{user_id}:15"),
                InlineKeyboardButton("✅ 30 дней", callback_data=f"approve:{user_id}:30"),
            ],
            [InlineKeyboardButton("⛔ Отклонить", callback_data=f"deny:{user_id}")],
        ]
    )


def fmt_signal(sig: Signal) -> str:
    return (
        f"📣 <b>{sig.symbol}</b> | <b>{sig.tf}</b>\n"
        f"Сигнал: <b>{sig.direction}</b> | Score: <b>{sig.score}/100</b>\n\n"
        f"Entry: <code>{sig.entry:.6g}</code>\n"
        f"SL: <code>{sig.sl:.6g}</code>\n"
        f"TP: <code>{sig.tp:.6g}</code>\n\n"
        f"Причины: {sig.reason}\n"
        f"Подтверждение+ретест: <b>учтено</b> (упрощённая модель)\n"
    )


# =========================
# HANDLERS
# =========================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not user:
        return
    upsert_user(user.id, user.username or "")

    u = get_user(user.id)
    is_admin = user.id == ADMIN_ID
    auto_on = bool(u and int(u["auto_scan"]) == 1)

    if not is_admin:
        # если pending — отправим заявку админу
        if u and u["status"] == "pending":
            try:
                await context.bot.send_message(
                    chat_id=ADMIN_ID,
                    text=(
                        f"📥 Новая заявка на доступ\n"
                        f"User: @{u['username'] or '—'}\n"
                        f"ID: <code>{u['user_id']}</code>"
                    ),
                    reply_markup=kb_approve(u["user_id"]),
                    parse_mode="HTML",
                )
            except Exception:
                pass

        await update.message.reply_text(
            "✅ Заявка отправлена администратору.\n"
            "Доступ откроется после одобрения (7/15/30 дней).",
        )
        return

    await update.message.reply_text(
        "👋 Панель управления ботом:",
        reply_markup=kb_main(is_admin=True, auto_on=auto_on),
    )


async def cmd_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not user:
        return
    u = get_user(user.id)
    if not u:
        upsert_user(user.id, user.username or "")
        u = get_user(user.id)

    is_admin = user.id == ADMIN_ID
    if not is_admin:
        ok, msg = is_user_active(u)
        if not ok:
            await update.message.reply_text(msg)
            return

    await update.message.reply_text(
        "Меню:",
        reply_markup=kb_main(is_admin=is_admin, auto_on=bool(int(u["auto_scan"]) == 1)),
    )


async def on_cb(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    q = update.callback_query
    if not q:
        return
    await q.answer()

    user = update.effective_user
    if not user:
        return

    u = get_user(user.id)
    if not u:
        upsert_user(user.id, user.username or "")
        u = get_user(user.id)

    is_admin = user.id == ADMIN_ID

    # защита доступа
    if not is_admin:
        ok, msg = is_user_active(u)
        if not ok:
            await q.message.reply_text(msg)
            return

    data = q.data or ""

    if data == "back":
        await q.message.edit_text("Меню:", reply_markup=kb_main(is_admin=is_admin, auto_on=bool(int(u["auto_scan"]) == 1)))
        return

    if data == "status":
        exp = parse_iso(u["expires_at"])
        exp_txt = exp.strftime("%Y-%m-%d %H:%M UTC") if exp else "—"
        await q.message.reply_text(
            f"📊 Статус\n"
            f"• Доступ: {u['status']}\n"
            f"• До: {exp_txt}\n"
            f"• Авто-анализ: {'ON' if int(u['auto_scan']) == 1 else 'OFF'}\n"
            f"• TF: {', '.join(TIMEFRAMES)}\n"
            f"• Скан: каждые {SCAN_INTERVAL_MIN} мин\n"
            f"• Рынок: крипта (BingX USDT)\n"
        )
        return

    if data == "toggle_auto":
        cur = int(u["auto_scan"]) == 1
        set_user_auto_scan(user.id, not cur)
        u2 = get_user(user.id)
        await q.message.edit_text(
            "Меню:",
            reply_markup=kb_main(is_admin=is_admin, auto_on=bool(int(u2["auto_scan"]) == 1)),
        )
        return

    if data == "new_signal":
        await q.message.reply_text("🔎 Ищу самый сильный сигнал в Top20 USDT…")
        try:
            symbols = get_top_usdt_symbols(TOP20)
            sig = best_signal(symbols)
        except Exception as e:
            await q.message.reply_text(f"⚠️ Ошибка получения данных BingX: {e}")
            return

        if not sig:
            await q.message.reply_text("Пока нет сильного сетапа по Top20. Попробуй позже.")
            return

        save_signal(sig)
        await q.message.reply_text(fmt_signal(sig), parse_mode="HTML")
        return

    # ADMIN
    if data == "admin":
        if not is_admin:
            return
        await q.message.edit_text("👑 Админ-панель:", reply_markup=kb_admin())
        return

    if data == "admin_pending":
        if not is_admin:
            return
        with db() as conn:
            cur = conn.execute("SELECT user_id, username, created_at FROM users WHERE status='pending' ORDER BY created_at DESC LIMIT 30")
            rows = cur.fetchall()
        if not rows:
            await q.message.reply_text("Заявок нет.")
            return
        for r in rows:
            uid, uname, created_at = r
            await q.message.reply_text(
                f"📥 Заявка\n@{uname or '—'} | <code>{uid}</code>\n{created_at}",
                reply_markup=kb_approve(int(uid)),
                parse_mode="HTML",
            )
        return

    if data == "admin_active":
        if not is_admin:
            return
        with db() as conn:
            cur = conn.execute("SELECT user_id, username, expires_at FROM users WHERE status='active' ORDER BY expires_at ASC LIMIT 50")
            rows = cur.fetchall()
        if not rows:
            await q.message.reply_text("Активных пользователей нет.")
            return
        text = "📦 Активные пользователи:\n\n"
        for uid, uname, exp in rows:
            exp_dt = parse_iso(exp)
            exp_txt = exp_dt.strftime("%Y-%m-%d %H:%M UTC") if exp_dt else "—"
            text += f"• @{uname or '—'} | <code>{uid}</code> | до {exp_txt}\n"
        await q.message.reply_text(text, parse_mode="HTML")
        return

    # approve/deny
    if data.startswith("approve:") and is_admin:
        _, uid_s, days_s = data.split(":")
        uid = int(uid_s)
        days = int(days_s)
        set_user_access(uid, days)
        await q.message.reply_text(f"✅ Доступ выдан: {uid} на {days} дней.")
        try:
            await context.bot.send_message(
                chat_id=uid,
                text=f"✅ Доступ активирован на {days} дней.\nОткрой меню: /menu",
            )
        except Exception:
            pass
        return

    if data.startswith("deny:") and is_admin:
        _, uid_s = data.split(":")
        uid = int(uid_s)
        block_user(uid)
        await q.message.reply_text(f"⛔ Доступ отклонён: {uid}.")
        try:
            await context.bot.send_message(chat_id=uid, text="⛔ Доступ отклонён администратором.")
        except Exception:
            pass
        return


# =========================
# AUTO SCAN JOB
# =========================
async def autoscan_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Каждые SCAN_INTERVAL_MIN минут:
    - берём Top200
    - ищем лучший сигнал
    - если score >= MIN_SCORE_FOR_ALERT -> рассылаем активным пользователям с auto_scan=1
    """
    log.info("autoscan tick...")
    try:
        symbols = get_top_usdt_symbols(TOP200)
        sig = best_signal(symbols)
    except Exception as e:
        log.warning("autoscan failed: %s", e)
        return

    if not sig:
        return
    if sig.score < MIN_SCORE_FOR_ALERT:
        return

    save_signal(sig)
    text = "🔥 <b>Сильный сигнал найден (Top200)</b>\n\n" + fmt_signal(sig)

    # рассылаем
    with db() as conn:
        cur = conn.execute("SELECT user_id, expires_at, auto_scan, status FROM users WHERE status='active'")
        users = cur.fetchall()

    for (uid, exp, auto_scan, status) in users:
        if int(auto_scan) != 1:
            continue
        exp_dt = parse_iso(exp)
        if not exp_dt or now_utc() >= exp_dt:
            continue
        try:
            await context.bot.send_message(chat_id=int(uid), text=text, parse_mode="HTML")
        except Exception:
            pass


# =========================
# MAIN
# =========================
def main() -> None:
    if not BOT_TOKEN:
        raise RuntimeError("BOT_TOKEN is empty. Set BOT_TOKEN in Railway Variables.")
    if ADMIN_ID == 0:
        raise RuntimeError("ADMIN_ID is empty/0. Set ADMIN_ID in Railway Variables.")

    init_db()

    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("menu", cmd_menu))
    app.add_handler(CallbackQueryHandler(on_cb))

    # авто-скан
    if AUTO_SCAN_DEFAULT:
        app.job_queue.run_repeating(autoscan_job, interval=SCAN_INTERVAL_MIN * 60, first=10)

    log.info("BOT STARTED")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
