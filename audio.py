"""오디오 입출력: arecord/aplay(alsa-utils)만 사용해서 추가 설치가 필요 없다."""
import collections
import math
import os
import re
import subprocess
import threading
import time

import numpy as np

SR = 16000
HOP = 512
BANDS = 64


def find_card():
    """XVF3800 사운드카드를 찾아 (카드번호, 채널수, 장치 샘플레이트) 를 돌려준다."""
    try:
        cards = [d for d in os.listdir("/proc/asound") if re.fullmatch(r"card\d+", d)]
    except OSError:
        return None
    for c in sorted(cards):
        try:
            usbid = open(f"/proc/asound/{c}/usbid").read().strip().lower()
        except OSError:
            continue
        if usbid != "2886:001a":
            continue
        ch, rate = 2, SR
        try:
            txt = open(f"/proc/asound/{c}/stream0").read()
            cap = txt.split("Capture:")[1]
            m = re.search(r"Channels:\s*(\d+)", cap)
            if m:
                ch = int(m.group(1))
            m = re.search(r"Rates:\s*([\d, ]+)", cap)
            if m:
                rate = int(m.group(1).split(",")[0])
        except (OSError, IndexError):
            pass
        return int(c[4:]), ch, rate
    return None


class Capture:
    """arecord 로 계속 받아 링버퍼에 쌓는다. self.n = 지금까지 받은 총 프레임 수."""

    def __init__(self, card, ch, seconds=240):
        self.card, self.ch = card, ch
        self.size = SR * seconds
        self.buf = np.zeros((self.size, ch), np.int16)
        self.n = 0
        self.error = None
        self.alive = True
        self.spec = collections.deque(maxlen=400)  # (seq, [L 밴드], [R 밴드])
        self.seq = 0
        self._win = np.hanning(HOP).astype(np.float32)
        self.proc = None
        threading.Thread(target=self._run, daemon=True).start()

    def _open(self):
        self.proc = subprocess.Popen(
            ["arecord", "-q", "-D", f"plughw:{self.card},0", "-f", "S16_LE", "-c", str(self.ch),
             "-r", str(SR), "-t", "raw", "--buffer-size", "4096"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def _run(self):
        nbytes = HOP * self.ch * 2
        try:
            self._open()
            while self.alive:
                data = self.proc.stdout.read(nbytes)
                if len(data) < nbytes:
                    err = self.proc.stderr.read().decode(errors="ignore").strip()
                    self.error = err or "arecord 가 종료되었습니다"
                    break
                self.push(np.frombuffer(data, np.int16).reshape(-1, self.ch))
        except Exception as e:  # noqa: BLE001
            self.error = str(e)
        self.alive = False

    def push(self, x):
        i = self.n % self.size
        k = len(x)
        if i + k <= self.size:
            self.buf[i:i + k] = x
        else:
            j = self.size - i
            self.buf[i:] = x[:j]
            self.buf[:k - j] = x[j:]
        self.n += k
        cols = []
        for c in range(min(2, self.ch)):
            f = np.fft.rfft(x[:HOP, c].astype(np.float32) / 32768 * self._win)
            p = (np.abs(f[1:257]) ** 2).reshape(BANDS, 4).mean(1) / (HOP * HOP / 8)
            d = 10 * np.log10(p + 1e-12)                      # 대략 dBFS
            cols.append(np.clip((d + 100) * (255 / 90), 0, 255).astype(int).tolist())
        self.seq += 1
        self.spec.append((self.seq, cols))

    def get(self, start, end=None):
        """프레임 구간 [start, end) 을 float32 (-1..1) 로 돌려준다."""
        end = self.n if end is None else min(end, self.n)
        start = max(start, end - self.size, 0)
        idx = np.arange(start, end) % self.size
        return self.buf[idx].astype(np.float32) / 32768

    def levels(self, seconds=0.1):
        x = self.get(self.n - int(SR * seconds))
        if len(x) == 0:
            return [-120.0] * self.ch
        return [round(float(10 * np.log10((x[:, c] ** 2).mean() + 1e-12)), 1) for c in range(self.ch)]

    def close(self):
        self.alive = False
        if self.proc:
            self.proc.kill()


class SimCapture(Capture):
    """가짜 장치용: 라우팅/빔 설정에 따라 그럴듯하게 달라지는 합성 신호."""

    def __init__(self, dev):
        self.dev = dev
        super().__init__(card=-1, ch=2)

    def _chan(self, route, speech, noise, echo):
        cat, src = route
        if cat in (6, 8):
            g = 1.0
            if self.dev.v["AEC_FIXEDBEAMSONOFF"][0] and cat == 6 and src < 2:
                az = self.dev.v["AEC_FIXEDBEAMSAZIMUTH_VALUES"][src]
                g = 0.12 + 0.88 * (0.5 + 0.5 * math.cos(az - math.radians(self.dev.src_deg))) ** 2
                if self.dev.v["AEC_FIXEDBEAMSGATING"][0]:   # 게이팅: 말이 없거나 다른 빔이 더 크면 무음
                    az2 = self.dev.v["AEC_FIXEDBEAMSAZIMUTH_VALUES"][1 - src]
                    g2 = 0.12 + 0.88 * (0.5 + 0.5 * math.cos(az2 - math.radians(self.dev.src_deg))) ** 2
                    if not speech.any() or g < g2:
                        return np.zeros(HOP, np.float32)
            ns = max(self.dev.v["PP_MIN_NS"][0], 0.05)
            return speech * g + noise * ns * g + echo * 0.03
        if cat in (1, 3, 7):
            return speech * 0.5 + noise + echo
        return np.zeros(HOP, np.float32)

    def _run(self):
        t = np.arange(HOP) / SR
        k = 0
        start = time.time()
        while self.alive:
            tt = t + k * HOP / SR
            sp = np.zeros(HOP)
            if self.dev.speech():
                env = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * tt)
                sp = env * sum(np.sin(2 * np.pi * 150 * h * tt) / h for h in range(1, 12)) * 0.08
            noise = np.random.randn(HOP) * 0.02
            echo = np.sin(2 * np.pi * (440 + 200 * np.sin(tt)) * tt) * 0.2 if self.dev.playing else np.zeros(HOP)
            out = np.stack([self._chan(self.dev.v["AUDIO_MGR_OP_L"], sp, noise, echo),
                            self._chan(self.dev.v["AUDIO_MGR_OP_R"], sp, noise, echo)], 1)
            self.push(np.clip(out * 32768, -32768, 32767).astype(np.int16))
            k += 1
            time.sleep(max(0, start + k * HOP / SR - time.time()))


def play(card, path, dev=None):
    """WAV 를 ReSpeaker 의 스피커 출력으로 재생 (AEC 기준 신호가 되려면 반드시 이 출력이어야 함)."""
    if card < 0:  # 시뮬레이션
        dev.playing = True

        class Fake:
            def kill(self_inner):
                dev.playing = False

            def poll(self_inner):
                return None
        return Fake()
    return subprocess.Popen(["aplay", "-q", "-D", f"plughw:{card},0", path],
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
