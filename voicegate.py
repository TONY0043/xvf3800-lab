"""방향 거름망: '내 쪽 빔'(L)과 '상대 쪽 빔'(R)의 크기를 비교해 내 말만 남긴다.

이 파일 하나만 복사하면 다른 기기(예: Jetson)에서도 동작한다. 필요한 것: python3, numpy, arecord(alsa-utils).
장치 설정(고정 빔 2개를 나/상대에게 하나씩, L=빔1, R=빔2, AGC 끔)은 플래시에 저장해 두면 어디에 꽂아도 유지된다.
플래시에 저장하지 않았다면 xvf_ctl.py 와 pyusb 가 있는 곳에서 setup 을 실행한다.

명령:
    python3 voicegate.py calibrate             # 그 자리에서 문턱 다시 맞추기 (터미널 안내를 따라 말하기)
    python3 voicegate.py live [--out 폴더] [--exec '명령 {}']
                                               # 실시간: 내 발화가 끝날 때마다 WAV 한 개로 저장
                                               #   --exec 를 주면 발화마다 {} 자리에 파일 경로를 넣어 실행한다
    python3 voicegate.py file in.wav out.wav   # 스테레오 녹음(L=내 빔, R=상대 빔)에서 내 말만 남기기
    python3 voicegate.py setup                 # 장치 설정 (xvf_ctl.py + pyusb 필요, 재부팅 전까지 유지)

파이썬 코드에서 바로 쓰기 (STT 스크립트 안에서):
    import voicegate
    for samples, start, diff_db in voicegate.utterances():   # samples: 16kHz 모노 float32 numpy 배열
        text = my_stt(samples)

문턱값은 voicegate.json (이 파일과 같은 폴더)에 저장된다.
"""
import json
import math
import sys
import wave
from pathlib import Path

import numpy as np

SR = 16000
WIN, HOP = 512, 256          # 32ms 창, 16ms 간격
CONF = Path(__file__).resolve().parent / "voicegate.json"
DEFAULT = {"thr": 4.0, "floor": None, "me": 90.0, "others": 270.0}


def device_settings(me_deg, others_deg):
    """이 방식에 필요한 장치 설정. 빔1=나, 빔2=상대, 두 빔을 L/R 로 따로 내보낸다."""
    return {
        "AEC_FIXEDBEAMSAZIMUTH_VALUES": [math.radians(me_deg % 360), math.radians(others_deg % 360)],
        "AEC_FIXEDBEAMSONOFF": [1],
        "AEC_FIXEDBEAMSGATING": [0],     # 켜면 한쪽 빔이 무음이 되어 비교할 수 없다
        "PP_AGCONOFF": [0],              # 음량이 저절로 변하면 크기 비교가 틀어진다
        "AUDIO_MGR_OP_L": [6, 0],
        "AUDIO_MGR_OP_R": [6, 1],
    }


def band_power(x, lo=300, hi=3400):
    """목소리 대역(300~3400Hz)의 프레임별 파워."""
    x = np.asarray(x, np.float64)
    n = (len(x) - WIN) // HOP + 1
    if n <= 0:
        return np.zeros(0)
    idx = np.arange(WIN)[None, :] + HOP * np.arange(n)[:, None]
    s = np.abs(np.fft.rfft(x[idx] * np.hanning(WIN), axis=1)) ** 2
    f = np.fft.rfftfreq(WIN, 1 / SR)
    return s[:, (f >= lo) & (f <= hi)].mean(1) / WIN + 1e-14


def db(p):
    return 10 * np.log10(np.maximum(p, 1e-14))


def noise_floor(x):
    """조용한 구간 녹음에서 배경 소음 크기(dB)를 잰다."""
    p = np.maximum(band_power(x[:, 0]), band_power(x[:, 1]))
    return float(np.median(db(p))) if len(p) else -90.0


def chunks(x, floor=None, margin=10.0, max_len=1.0):
    """말 구간을 찾아 최대 max_len 초 조각으로 나누고, 조각마다 L-R 크기 차이(dB)를 잰다.

    돌려주는 값: [{"t0","t1"(초), "diff"(dB), "level"(dB)}]
    """
    pl, pr = band_power(x[:, 0]), band_power(x[:, 1])
    if len(pl) == 0:
        return []
    e = db(np.maximum(pl, pr))
    if floor is None:
        floor = float(np.percentile(e, 10))
    act = e > floor + margin
    fps = SR / HOP
    # 0.3초보다 짧은 끊김은 이어 붙이고, 0.15초보다 짧은 소리는 버린다
    segs, start, gap = [], None, 0
    for i, a in enumerate(np.append(act, False)):
        if a:
            if start is None:
                start = i
            last, gap = i, 0
        elif start is not None:
            gap += 1
            if gap > 0.3 * fps or i == len(act):
                if last - start + 1 >= 0.15 * fps:
                    segs.append((start, last + 1))
                start = None
    out = []
    for a, b in segs:
        k = max(1, int(math.ceil((b - a) / (max_len * fps))))
        edges = np.linspace(a, b, k + 1).astype(int)
        for c0, c1 in zip(edges[:-1], edges[1:]):
            if c1 <= c0:
                continue
            m = act[c0:c1]
            sl, sr_ = pl[c0:c1][m].sum(), pr[c0:c1][m].sum()
            if sl + sr_ <= 0:
                continue
            out.append({"t0": c0 * HOP / SR, "t1": (c1 * HOP + WIN - HOP) / SR,
                        "diff": float(db(sl) - db(sr_)), "level": float(db(max(sl, sr_) / max(m.sum(), 1)))})
    return out


def dur(cs):
    return float(sum(c["t1"] - c["t0"] for c in cs))


def passed(cs, thr):
    return [c for c in cs if c["diff"] >= thr]


def calibrate(me, others):
    """'나만'과 '상대만' 녹음의 조각들로 문턱값을 정한다. (문턱, 내 말 보존율, 남의 말 누설률)"""
    grid = np.arange(-20, 30.01, 0.25)
    dm, do = dur(me), dur(others)

    def rates(t):
        keep = dur(passed(me, t)) / dm if dm else 0.0
        leak = dur(passed(others, t)) / do if do else 0.0
        return keep, leak

    score = np.array([rates(t)[0] - rates(t)[1] for t in grid])
    best = np.flatnonzero(score >= score.max() - 1e-9)
    thr = float(grid[best[len(best) // 2]])      # 점수가 같은 구간의 한가운데
    keep, leak = rates(thr)
    return thr, keep, leak


def apply(x, cs, thr, pad=0.15):
    """통과한 조각의 L 채널만 남기고 나머지는 무음으로 만든다. (신호, 통과한 발화 구간 목록)"""
    n = len(x)
    mask = np.zeros(n)
    for c in passed(cs, thr):
        mask[max(0, int((c["t0"] - pad) * SR)):min(n, int((c["t1"] + pad) * SR))] = 1
    utt, d = [], np.diff(np.concatenate([[0], mask, [0]]))
    for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        utt.append((a / SR, b / SR))
    k = int(0.01 * SR)                                   # 10ms 페이드로 딸깍 소리 방지
    mask = np.convolve(mask, np.ones(k) / k, "same")
    return (x[:, 0] * mask).astype(np.float32), utt


def load_conf():
    c = dict(DEFAULT)
    if CONF.exists():
        c.update(json.loads(CONF.read_text()))
    return c


def _read_wav(path):
    with wave.open(str(path)) as w:
        if w.getnchannels() != 2 or w.getframerate() != SR or w.getsampwidth() != 2:
            raise SystemExit("16kHz, 16bit, 스테레오(L=내 빔, R=상대 빔) WAV 가 필요합니다")
        return np.frombuffer(w.readframes(w.getnframes()), "<i2").reshape(-1, 2).astype(np.float32) / 32768


class Live:
    """블록 단위로 소리를 받아, 말이 끝날 때마다(0.4초 쉼) 그 구간을 거름망에 통과시켜 내 발화만 내보낸다."""
    BLOCK = 1024                      # 64ms

    def __init__(self, conf, on_utt, margin=10.0, min_len=0.3, max_len=15.0):
        self.thr, self.floor = conf["thr"], conf.get("floor")
        if self.floor is None:
            raise SystemExit("먼저 테스트 랩의 '내 목소리만' 탭에서 보정하세요 (voicegate.json 이 없습니다)")
        self.on_utt, self.margin, self.min_len, self.max_len = on_utt, margin, min_len, max_len
        self.buf, self.active, self.gap, self.t = [], False, 0.0, 0.0   # t = 지금까지 받은 길이(초)

    def feed(self, block):
        """block: (BLOCK, 2) float32"""
        dt = len(block) / SR
        self.t += dt
        e = db(max(band_power(block[:, 0]).max(), band_power(block[:, 1]).max()))
        loud = e > self.floor + self.margin
        self.buf.append(block)
        if loud:
            self.active, self.gap = True, 0.0
        elif self.active:
            self.gap += dt
        else:
            self.buf = self.buf[-5:]                      # 말이 없을 때는 직전 0.3초만 들고 있는다
        if self.active and (self.gap >= 0.4 or len(self.buf) * dt >= self.max_len):
            self.flush()

    def flush(self):
        x = np.concatenate(self.buf)
        t0 = self.t - len(x) / SR
        cs = chunks(x, self.floor, self.margin)
        _y, utt = apply(x, cs, self.thr)
        for a, b in utt:
            if b - a >= self.min_len:
                inside = [c["diff"] for c in cs if c["t1"] > a and c["t0"] < b]
                self.on_utt(x[int(a * SR):int(b * SR), 0], t0 + a, float(np.mean(inside)) if inside else 0.0)
        self.buf, self.active, self.gap = self.buf[-5:], False, 0.0


def find_card():
    """XVF3800 의 ALSA 카드 번호 (USB ID 2886:001a). 없으면 None."""
    for f in sorted(Path("/proc/asound").glob("card*/usbid")):
        if f.read_text().strip().lower() == "2886:001a":
            return int(f.parent.name[4:])
    return None


def apply_setup(conf):
    from xvf_ctl import XVF
    dev = XVF()
    for k, v in device_settings(conf["me"], conf["others"]).items():
        dev.write(k, v)


def mic_blocks(seconds=None):
    """마이크에서 (BLOCK, 2) float32 블록을 계속 내놓는다. seconds 를 주면 그만큼만."""
    import subprocess
    card = find_card()
    if card is None:
        raise SystemExit("XVF3800 사운드카드를 찾지 못했습니다 (USB 연결 확인)")
    proc = subprocess.Popen(["arecord", "-q", "-D", f"plughw:{card},0", "-f", "S16_LE", "-r", str(SR), "-c", "2", "-t", "raw"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    nbytes, got = Live.BLOCK * 4, 0
    try:
        while seconds is None or got < seconds * SR:
            raw = proc.stdout.read(nbytes)
            if len(raw) < nbytes:
                raise SystemExit("오디오 입력이 끊겼습니다: " + proc.stderr.read().decode(errors="ignore").strip()
                                 + "  (다른 프로그램이 이 마이크를 쓰고 있으면 먼저 끄세요)")
            got += Live.BLOCK
            yield np.frombuffer(raw, "<i2").reshape(-1, 2).astype(np.float32) / 32768
    finally:
        proc.kill()


def utterances(conf=None, setup=False):
    """내 발화가 끝날 때마다 (samples, 시작 시각(초), L-R 차이(dB)) 를 내놓는 생성기."""
    conf = conf or load_conf()
    if setup:
        apply_setup(conf)
    done = []
    lv = Live(conf, lambda y, t, d: done.append((y, t, d)))
    for block in mic_blocks():
        lv.feed(block)
        while done:
            yield done.pop(0)


def _write_mono(path, y):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(y, -1, 1) * 32767).astype("<i2").tobytes())


def calibrate_cli(argv, conf):
    import argparse
    import time
    ap = argparse.ArgumentParser(prog="voicegate.py calibrate")
    ap.add_argument("--seconds", type=float, default=20, help="나만/상대만 말하는 시간")
    ap.add_argument("--me", type=float, default=conf["me"])
    ap.add_argument("--others", type=float, default=conf["others"])
    ap.add_argument("--setup", action="store_true", help="먼저 장치 설정을 적용 (xvf_ctl.py + pyusb 필요)")
    args = ap.parse_args(argv)
    conf.update(me=args.me, others=args.others)
    if args.setup:
        apply_setup(conf)

    def rec(msg, secs):
        input(f"\n▶ {msg}\n  준비되면 Enter 를 누르세요 ({secs:.0f}초 녹음)")
        print("  녹음 중...", flush=True)
        return np.concatenate(list(mic_blocks(secs)))

    xq = rec("모두 조용히 합니다 (배경 소음 측정)", 3)
    floor = noise_floor(xq)
    cm = chunks(rec("나만 말합니다 — 평소 말하듯 계속", args.seconds), floor)
    co = chunks(rec("상대만 말합니다 — 나는 조용히", args.seconds), floor)
    if dur(cm) < 2 or dur(co) < 2:
        raise SystemExit(f"말소리가 너무 적게 잡혔습니다 (나 {dur(cm):.1f}초, 상대 {dur(co):.1f}초). 더 크게, 계속 말해 주세요.")
    thr, keep, leak = calibrate(cm, co)
    med = lambda cs: float(np.median([c["diff"] for c in cs]))  # noqa: E731
    conf.update(thr=round(thr, 2), floor=round(floor, 1), keep=round(keep, 3), leak=round(leak, 3),
                time=time.strftime("%Y-%m-%d %H:%M:%S"))
    CONF.write_text(json.dumps(conf, indent=1))
    print(f"\n문턱 {thr:+.1f} dB · 내 말 보존 {100 * keep:.0f}% · 남의 말 누설 {100 * leak:.0f}%")
    print(f"L-R 중앙값: 나 {med(cm):+.1f} dB / 상대 {med(co):+.1f} dB (차이가 10 dB 이상이면 넉넉함)")
    if med(cm) - med(co) < 6:
        print("두 빔의 차이가 작습니다. 장치 설정(고정 빔, L=빔1/R=빔2)이 적용됐는지, 나와 상대가 반대편에 있는지 확인하세요.")
    print(f"저장: {CONF}")


def live(argv, conf):
    import argparse
    import subprocess
    import time
    ap = argparse.ArgumentParser(prog="voicegate.py live")
    ap.add_argument("--out", default="utterances", help="발화 WAV 를 저장할 폴더")
    ap.add_argument("--exec", dest="cmd", help="발화마다 실행할 명령. {} 자리에 WAV 경로가 들어간다")
    ap.add_argument("--from-wav", help="(시험용) 장치 대신 스테레오 WAV 를 흘려 넣는다")
    ap.add_argument("--setup", action="store_true", help="먼저 장치 설정을 적용 (xvf_ctl.py + pyusb 필요)")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    count = [0]

    def on_utt(y, t, diff):
        count[0] += 1
        path = out / f"{stamp}_{count[0]:04d}.wav"
        _write_mono(path, y)
        print(json.dumps({"file": str(path), "start": round(t, 2), "seconds": round(len(y) / SR, 2), "diff_db": round(diff, 1)}), flush=True)
        if args.cmd:
            subprocess.run(args.cmd.replace("{}", str(path)), shell=True, check=False)

    lv = Live(conf, on_utt)
    if args.from_wav:
        x = _read_wav(args.from_wav)
        for i in range(0, len(x) - Live.BLOCK + 1, Live.BLOCK):
            lv.feed(x[i:i + Live.BLOCK])
        if lv.active:
            lv.flush()
        return
    if args.setup:
        apply_setup(conf)
    print(f"듣는 중 (문턱 {conf['thr']:+.1f} dB) — Ctrl+C 로 종료", file=sys.stderr)
    try:
        for block in mic_blocks():
            lv.feed(block)
    except KeyboardInterrupt:
        pass


def main(argv):
    conf = load_conf()
    if len(argv) >= 1 and argv[0] == "setup":
        apply_setup(conf)
        print(f"설정했습니다: 빔1(나) {conf['me']:.0f}°, 빔2(상대) {conf['others']:.0f}°, L=빔1, R=빔2, AGC 끔")
        print("재부팅하면 사라집니다. 계속 쓰려면 테스트 랩의 파라미터 탭에서 '플래시에 저장'을 누르세요.")
    elif len(argv) >= 1 and argv[0] == "live":
        live(argv[1:], conf)
    elif len(argv) >= 1 and argv[0] == "calibrate":
        calibrate_cli(argv[1:], conf)
    elif len(argv) == 3 and argv[0] == "file":
        x = _read_wav(argv[1])
        cs = chunks(x, conf.get("floor"))
        y, utt = apply(x, cs, conf["thr"])
        _write_mono(argv[2], y)
        Path(argv[2]).with_suffix(".segments.json").write_text(json.dumps([{"start": round(a, 2), "end": round(b, 2)} for a, b in utt]))
        print(f"문턱 {conf['thr']:+.1f} dB · 말 구간 {dur(cs):.1f}초 중 {dur(passed(cs, conf['thr'])):.1f}초 통과 · 발화 {len(utt)}개 → {argv[2]}")
    else:
        print(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
