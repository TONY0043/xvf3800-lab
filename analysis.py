"""측정 계산과 그림(스펙트로그램, 극좌표). 그림 글자는 폰트 문제를 피하려고 영어로 쓴다."""
import math
import wave

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

SR = 16000


def db(p):
    return 10 * np.log10(np.maximum(p, 1e-12))


def rms_db(x):
    return float(db((np.asarray(x, np.float64) ** 2).mean())) if len(x) else -120.0


def frame_db(x, win=0.02):
    """20ms 단위 구간별 음량(dBFS)."""
    n = int(SR * win)
    m = len(x) // n
    if m == 0:
        return np.array([-120.0])
    return db((x[:m * n].reshape(m, n).astype(np.float64) ** 2).mean(1))


def pct(x, q):
    return float(np.percentile(frame_db(x), q))


def normalize(x):
    """듣기 비교용: 큰 소리 구간이 -20 dBFS 가 되게 맞춘다 (음량 차이로 인한 착각 방지)."""
    g = 10 ** ((-20 - pct(x, 95)) / 20)
    return np.clip(x * min(g, 1000.0), -0.98, 0.98)


def write_wav(path, x):
    x = np.asarray(x)
    ch = 1 if x.ndim == 1 else x.shape[1]
    with wave.open(str(path), "wb") as w:
        w.setnchannels(ch)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


def circ_mean(deg):
    a = np.radians(np.asarray(deg, float))
    return float(np.degrees(math.atan2(np.sin(a).mean(), np.cos(a).mean())) % 360)


def wrap(d):
    return (d + 180) % 360 - 180


def test_signal(seconds=12.0):
    """에코 테스트용 재생음: 말소리처럼 끊기는 대역 잡음 + 멜로디. 스테레오로 돌려준다."""
    rng = np.random.default_rng(1)
    n = int(SR * seconds)
    t = np.arange(n) / SR
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / SR)
    spec *= ((f > 200) & (f < 4000)) / np.sqrt(np.maximum(f, 200))
    noise = np.fft.irfft(spec, n)
    noise /= np.abs(noise).max()
    syll = np.clip(np.sin(2 * np.pi * 3.3 * t) + 0.3, 0, 1) * (np.sin(2 * np.pi * 0.35 * t) > -0.5)
    notes = [262, 330, 392, 523, 392, 330, 294, 349, 440, 349]
    mel = np.zeros(n)
    step = int(SR * 0.3)
    for i in range(0, n, step):
        fr = notes[(i // step) % len(notes)]
        seg = t[:min(step, n - i)]
        mel[i:i + len(seg)] = (np.sin(2 * np.pi * fr * seg) + 0.4 * np.sin(4 * np.pi * fr * seg)) * np.exp(-seg * 6)
    y = 0.22 * noise * syll + 0.16 * mel
    fade = np.minimum(1, np.minimum(t, seconds - t) / 0.05)
    y = (y * fade).astype(np.float32)
    return np.stack([y, y], 1)


def _save(fig, path):
    FigureCanvasAgg(fig)
    fig.savefig(str(path), dpi=100, facecolor="white")


def plot_spectrograms(path, chans, marks=(), timeline=None):
    """chans: [(영문 라벨, 신호)], marks: [(초, 라벨)], timeline: 방향 기록 리스트."""
    rows = len(chans) + (1 if timeline else 0)
    fig = Figure(figsize=(11, 2.6 * rows), constrained_layout=True)
    axes = np.atleast_1d(fig.subplots(rows, 1, sharex=True))
    for ax, (label, x) in zip(axes, chans):
        x = np.asarray(x, np.float64) + np.random.randn(len(x)) * 1e-6
        pxx, _f, _t, im = ax.specgram(x, NFFT=512, Fs=SR, noverlap=384, cmap="magma")
        top = float(db(pxx.max()))
        im.set_clim(top - 75, top)
        ax.set_ylabel("Hz")
        ax.set_title(f"{label}   (RMS {rms_db(x):.1f} dBFS)", loc="left", fontsize=10)
        for sec, text in marks:
            ax.axvline(sec, color="cyan", lw=1)
            ax.text(sec, 7600, " " + text, color="cyan", fontsize=8, va="top")
    if timeline:
        ax = axes[-1]
        ts = np.array([p["t"] for p in timeline])
        sp = np.array([bool(p["speech"]) for p in timeline])
        doa = np.array([p["doa"] if p["doa"] is not None else np.nan for p in timeline], float)
        ax.scatter(ts[sp], doa[sp], s=10, c="k", label="DOA (speech detected)", zorder=3)
        for i, name in enumerate(["beam 1", "beam 2", "free-running", "auto-select"]):
            az = np.array([p["az"][i] if p["az"] and p["az"][i] is not None else np.nan for p in timeline], float)
            ax.plot(ts, az, lw=1.2, label=name)
        ax.set_ylim(0, 360)
        ax.set_yticks(range(0, 361, 90))
        ax.set_ylabel("direction (deg)")
        ax.legend(ncol=5, fontsize=8, loc="upper right")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("time (s)")
    _save(fig, path)


def plot_polar(path, series, title, rmin=None, rmax=None, marks=()):
    """series: [(라벨, 각도[deg], 값, 스타일)], marks: [(각도, 라벨)]"""
    fig = Figure(figsize=(6, 6), constrained_layout=True)
    ax = fig.add_subplot(projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    for label, deg, val, style in series:
        th = np.radians(np.asarray(deg, float))
        if style == "line":
            ax.plot(np.append(th, th[0]), np.append(val, val[0]), "o-", lw=2, label=label)
        else:
            ax.plot(th, val, style, ms=11, label=label)
    for deg, text in marks:
        ax.annotate(text, (math.radians(deg), ax.get_rmax()), fontsize=9, ha="center")
    if rmin is not None:
        ax.set_rlim(rmin, rmax)
    ax.set_title(title, fontsize=11)
    ax.legend(loc="lower right", fontsize=8)
    _save(fig, path)


def plot_gate(path, parts, thr):
    """parts: [(영문 제목, 조각 목록, 길이 초)] — 조각마다 L-R 차이를 막대로, 통과=초록 / 차단=빨강."""
    fig = Figure(figsize=(11, 2.3 * len(parts)), constrained_layout=True)
    axes = np.atleast_1d(fig.subplots(len(parts), 1))
    for ax, (title, cs, secs) in zip(axes, parts):
        for c in cs:
            ax.bar(c["t0"], c["diff"], c["t1"] - c["t0"], align="edge",
                   color="#2e9d57" if c["diff"] >= thr else "#d0453a", edgecolor="white", lw=0.3)
        ax.axhline(thr, color="black", ls="--", lw=1)
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_xlim(0, max(secs, 1))
        ax.set_ylim(-25, 25)
        ax.set_ylabel("L - R (dB)")
        ax.set_title(f"{title}   green = passed, red = blocked, dashed = threshold {thr:+.1f} dB", loc="left", fontsize=10)
    axes[-1].set_xlabel("seconds")
    _save(fig, path)


def plot_levels(path, parts, thr):
    """parts: [(영문 제목, 시간축, {라벨: dB 배열})] — 게이팅 시험용 음량 그래프."""
    fig = Figure(figsize=(11, 2.3 * len(parts)), constrained_layout=True)
    axes = np.atleast_1d(fig.subplots(len(parts), 1))
    for ax, (title, t, series) in zip(axes, parts):
        for (label, y), color in zip(series.items(), ("#888888", "#2e7dd1")):
            ax.plot(t, np.maximum(y, -110), color=color, lw=1, label=label)
        ax.axhline(thr, color="black", ls="--", lw=1)
        ax.set_ylim(-112, -20)
        ax.set_xlim(0, max(float(t[-1]) if len(t) else 1, 1))
        ax.set_ylabel("dB")
        ax.set_title(title + "   (dashed = level counted as 'audible')", loc="left", fontsize=10)
        ax.legend(loc="upper right", fontsize=8, ncol=2)
    axes[-1].set_xlabel("seconds")
    _save(fig, path)
