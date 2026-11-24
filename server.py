#!/usr/bin/env python3
"""reSpeaker XVF3800 테스트 랩 — 브라우저로 보고 듣고 판단하는 테스트 도구.

실행:  python3 server.py          (장치 연결 후)
       python3 server.py --sim    (장치 없이 화면만 미리 보기)
"""
import argparse
import collections
import contextlib
import json
import math
import mimetypes
import threading
import time
import traceback
import wave
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

import analysis as A
import voicegate as VG
import audio
from xvf_ctl import P, SimXVF, XVF, DeviceNotFound

BASE = Path(__file__).resolve().parent
RUNS = BASE / "runs"
SR = audio.SR

# 출력 채널 라우팅 프리셋: (카테고리, 소스) — 공식 host_control/README 의 표 기준
ROUTES = {
    "proc_raw": {"ko": "처리된 출력 ↔ 원본 마이크", "L": (6, 3), "R": (3, 0),
                 "names": [("L: 처리된 출력 (자동선택 빔)", "L: processed (auto-select beam)"),
                           ("R: 원본 마이크 0 (처리 전)", "R: raw mic 0 (unprocessed)")]},
    "beam12": {"ko": "빔 1 ↔ 빔 2", "L": (6, 0), "R": (6, 1),
               "names": [("L: 빔 1", "L: beam 1"), ("R: 빔 2", "R: beam 2")]},
    "auto_free": {"ko": "자동선택 빔 ↔ 탐색(free-running) 빔", "L": (6, 3), "R": (6, 2),
                  "names": [("L: 자동선택 빔", "L: auto-select beam"), ("R: 탐색 빔", "R: free-running beam")]},
    "proc_resid": {"ko": "처리된 출력 ↔ AEC 직후(마이크 0)", "L": (6, 3), "R": (7, 0),
                   "names": [("L: 처리된 출력", "L: processed"), ("R: AEC 직후 마이크 0", "R: AEC residual mic 0")]},
}
SLOW = ["PP_AGCONOFF", "PP_AGCGAIN", "PP_MIN_NS", "PP_MIN_NN", "PP_ECHOONOFF", "AEC_FIXEDBEAMSONOFF",
        "AEC_FIXEDBEAMSAZIMUTH_VALUES", "AEC_FIXEDBEAMSGATING", "AEC_AECCONVERGED", "AEC_RT60",
        "AUDIO_MGR_OP_L", "AUDIO_MGR_OP_R", "LED_EFFECT", "AUDIO_MGR_SYS_DELAY", "GPO_READ_VALUES"]


# 파라미터 탭 목록: (명령, 한글 이름, 종류, 범위/선택지, 설명)
#   종류: bool | sel [(값, 이름)] | num (최소, 최대, 단계) 또는 None | deg (라디안을 도 단위로) | hex
# 범위와 뜻은 공식 xvf_host.py 의 명령 설명을 옮긴 것이다.
PARAMS = [
    ("잡음 제거", [
        ("PP_MIN_NS", "일정한 잡음 남기는 양", "num", (0, 1, 0.01), "선풍기·에어컨처럼 계속 나는 소음. 0 에 가까울수록 강하게 지우고 1 이면 끔. 너무 낮추면 목소리가 로봇처럼 변함."),
        ("PP_MIN_NN", "돌발 잡음 남기는 양", "num", (0, 1, 0.01), "키보드·그릇 소리처럼 갑자기 나는 소음. 0 에 가까울수록 강하게 지움, 1 이면 끔."),
        ("PP_ATTNS_MODE", "말 없을 때 추가로 줄이기", "bool", None, "아무도 말하지 않는 동안 AGC 이득을 더 낮춰 배경을 조용하게 함."),
        ("PP_ATTNS_NOMINAL", "말 없을 때 줄이는 양", "num", (0, 1, 0.01), "보통 음량일 때 말이 없는 구간을 얼마나 더 줄일지."),
        ("PP_ATTNS_SLOPE", "작게 말할수록 더 줄이기", "num", (0, 5, 0.1), "1.0 이면 AGC 가 볼륨을 올려도 배경 소음 크기가 거의 일정하게 유지됨."),
        ("AEC_HPFONOFF", "저음 차단 필터", "sel", [(0, "끔"), (1, "70 Hz"), (2, "125 Hz"), (3, "150 Hz"), (4, "180 Hz")], "웅웅거리는 저음(책상 진동, 바람)을 잘라냄."),
    ]),
    ("자동 음량 (AGC)", [
        ("PP_AGCONOFF", "자동 음량 조절", "bool", None, "멀리서 작게 말해도 볼륨을 자동으로 키워 주는 기능. 끄면 아래 '현재 이득' 값으로 고정됨."),
        ("PP_AGCGAIN", "현재 이득 (배)", "num", (1, 1000, 0.1), "지금 곱해지고 있는 배율. AGC 가 켜져 있으면 계속 변함."),
        ("PP_AGCMAXGAIN", "최대 이득 (배)", "num", (1, 1000, 1), "AGC 가 올릴 수 있는 한계. 낮추면 조용할 때 잡음이 덜 커짐."),
        ("PP_AGCDESIREDLEVEL", "목표 출력 크기 (파워)", "num", (1e-8, 1, None), "출력을 이 크기에 맞추려 함. 파워 값이라 0.01 = -20 dB."),
        ("PP_AGCTIME", "반응 속도 (초)", "num", (0.5, 4, 0.1), "볼륨을 올리고 내리는 속도. 작을수록 빠름."),
        ("PP_AGCFASTTIME", "큰 소리 때 내리는 속도 (초)", "num", (0.05, 4, 0.05), "갑자기 큰 소리가 날 때 볼륨을 빨리 내리는 시간."),
        ("PP_AGCALPHAFASTGAIN", "빠른 모드 진입 이득", "num", (0, 1000, 1), "이득이 이 값을 넘으면 빠른 추적 모드 사용."),
        ("PP_AGCALPHASLOW", "음성 파워 기억 (느림)", "num", (0, 1, 0.01), "전문가용. 음성 크기 추정의 느린 평균 계수."),
        ("PP_AGCALPHAFAST", "음성 파워 기억 (빠름)", "num", (0, 1, 0.01), "전문가용. 음성 크기 추정의 빠른 평균 계수."),
        ("PP_LIMITONOFF", "리미터", "bool", None, "출력이 찢어지지 않게 최대 크기를 제한."),
        ("PP_LIMITPLIMIT", "리미터 최대 파워", "num", (1e-8, 1, None), "리미터가 허용하는 최대 파워."),
    ]),
    ("에코 제거 (AEC)", [
        ("SHF_BYPASS", "AEC 우회", "bool", None, "켜면 에코 제거 처리를 건너뜀 (비교용). 평소에는 꺼 둠."),
        ("PP_ECHOONOFF", "잔여 에코 억제", "bool", None, "AEC 가 빼고 남은 에코 찌꺼기를 한 번 더 눌러 줌."),
        ("PP_NLATTENONOFF", "비선형 에코 감쇠", "bool", None, "스피커가 찌그러지며 생긴 에코를 추가로 줄임."),
        ("PP_GAMMA_E", "에코 더 빼기 (직접음)", "num", (0, 2, 0.05), "클수록 에코는 더 줄지만 동시에 말할 때 내 목소리도 깎임."),
        ("PP_GAMMA_ETAIL", "에코 더 빼기 (잔향 꼬리)", "num", (0, 2, 0.05), "방에 울려 늦게 돌아오는 에코에 대한 강도."),
        ("PP_GAMMA_ENL", "에코 더 빼기 (비선형)", "num", (0, 5, 0.05), "스피커 왜곡 성분에 대한 강도."),
        ("PP_DTSENSITIVE", "동시 말하기 ↔ 에코 억제 균형", "num", (0, 15, 1), "0~5 또는 10~15. 낮을수록 에코를 세게 지우고, 높을수록 동시에 말할 때 목소리가 잘 살아남."),
        ("PP_FMIN_SPEINDEX", "동시 말하기 경계 주파수 (Hz)", "num", (0, 7999, 10), "동시에 말할 때 이 주파수 아래를 더 세게 억제. 기기별 튜닝 값 (Seeed 기본 1300)."),
        ("AUDIO_MGR_SYS_DELAY", "기준 신호 지연 (샘플)", "num", None, "스피커 소리가 마이크에 닿기까지의 지연 보정. 3.5mm 로 외부 스피커를 쓰는데 AEC 가 약하면 조정 (Seeed 기본 12)."),
        ("AUDIO_MGR_REF_GAIN", "기준 신호 이득", "num", None, "AEC 에 넣는 스피커 신호의 이득 (Seeed 기본 8.0)."),
        ("AEC_FAR_EXTGAIN", "외부 앰프 이득 (dB)", "num", None, "장치 밖에서 스피커 볼륨을 키웠다면 그만큼 알려 줌."),
        ("AEC_AECEMPHASISONOFF", "AEC 강조 필터", "sel", [(0, "끔"), (1, "켬"), (2, "켬 (저음 부스트 스피커용)")], "음성 대역을 강조해서 AEC 를 학습시킴."),
        ("AEC_PCD_COUPLINGI", "경로 변화 감지 민감도", "num", None, "스피커나 장치를 옮긴 것을 알아채는 민감도 (0~1). 범위 밖 값(예: -1)이면 끔."),
        ("AEC_PCD_MINTHR", "경로 변화 최소 임계", "num", (0, 0.02, 0.001), "전문가용."),
        ("AEC_PCD_MAXTHR", "경로 변화 최대 임계", "num", (0.025, 0.2, 0.005), "전문가용."),
    ]),
    ("빔포밍", [
        ("AEC_FIXEDBEAMSONOFF", "고정 빔 모드", "bool", None, "켜면 빔 1·2 가 사람을 따라가지 않고 아래 각도만 계속 바라봄 (지정석 마이크처럼)."),
        ("AEC_FIXEDBEAMSAZIMUTH_VALUES", "고정 빔 방위각 (빔1, 빔2)", "deg", None, "고정 빔 모드에서 바라볼 수평 방향."),
        ("AEC_FIXEDBEAMSELEVATION_VALUES", "고정 빔 높이각 (빔1, 빔2)", "deg", None, "고정 빔 모드에서 바라볼 위아래 각도."),
        ("AEC_FIXEDBEAMSGATING", "고정 빔 게이팅", "bool", None, "말소리가 있는 빔만 내보내고 나머지 빔은 무음 처리."),
        ("AEC_FIXEDBEAMNOISETHR", "고정 빔 잡음 학습 임계 (빔1, 빔2)", "num", (0, 1, 0.01), "탐색 빔이 고정 빔과 얼마나 가까울 때까지 잡음 제거기를 갱신할지."),
        ("AEC_ASROUTONOFF", "음성인식(ASR)용 출력", "bool", None, "켜면 카테고리 7 출력이 빔별 ASR 신호가 되고, 끄면 마이크별 AEC 직후 신호가 됨."),
        ("AEC_ASROUTGAIN", "ASR 출력 이득 (배)", "num", (0, 1000, 0.1), "ASR 출력에 곱하는 고정 배율."),
    ]),
    ("입력 · 출력", [
        ("AUDIO_MGR_MIC_GAIN", "마이크 이득", "num", None, "처리 전에 마이크에 곱하는 이득 (Seeed 기본 90). 너무 크면 큰 소리에서 찢어짐."),
        ("AUDIO_MGR_OP_L", "L 채널 신호 (카테고리, 소스)", "num", (0, 12, 1), "예: 6,3 = 자동선택 빔 · 3,0 = 증폭된 마이크 0 · 7,3 = ASR 빔 3 · 8,0 = 기본."),
        ("AUDIO_MGR_OP_R", "R 채널 신호 (카테고리, 소스)", "num", (0, 12, 1), "L 채널과 같은 방식."),
        ("AIC3104_HP_LEVEL", "헤드폰 출력 레벨", "num", (0, 9, 1), "3.5mm 출력 크기 (0~9 dB). 펌웨어 2.1.0 이상."),
        ("AIC3104_LINEOUT_LEVEL", "라인/스피커 출력 레벨", "num", (0, 9, 1), "스피커 단자 출력 크기 (0~9 dB). 펌웨어 2.1.0 이상."),
    ]),
    ("LED", [
        ("LED_EFFECT", "LED 효과", "sel", [(0, "끔"), (1, "숨쉬기"), (2, "무지개"), (3, "단색"), (4, "방향 표시(DOA)"), (5, "링")], "LED 링이 보여 줄 효과."),
        ("LED_BRIGHTNESS", "밝기", "num", (0, 255, 1), "숨쉬기·무지개 효과의 밝기."),
        ("LED_SPEED", "속도", "num", (0, 255, 1), "숨쉬기·무지개 효과의 속도."),
        ("LED_COLOR", "색", "hex", None, "숨쉬기·단색 효과의 색 (RRGGBB)."),
    ]),
]
PARAM_SPEC = {e[0]: e for _g, items in PARAMS for e in items}


class Cancelled(Exception):
    pass


def clean(o, nd=4):
    """JSON 에 못 들어가는 NaN/inf, numpy 타입 정리."""
    if isinstance(o, dict):
        return {k: clean(v, nd) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v, nd) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        o = o.item()
    if isinstance(o, float):
        return round(o, nd) if math.isfinite(o) else None
    return o


class App:
    def __init__(self, sim):
        self.sim = sim
        self.dev = None
        self.cap = None
        self.info = {"sim": sim}
        self.ctl_error = "장치를 찾는 중…"
        self.live = {}
        self.slow = {}
        self.timeline = collections.deque(maxlen=6000)
        self.job = {"running": False}
        self.baseline = {}
        self.cancel = False
        RUNS.mkdir(exist_ok=True)
        threading.Thread(target=self.poll_loop, daemon=True).start()

    # ---------- 연결 / 폴링 ----------
    def connect(self):
        if self.dev is None:
            try:
                dev = SimXVF() if self.sim else XVF()
                ver = dev.read("VERSION")
                self.info.update(version=".".join(map(str, ver)), build=self.try_read("BLD_MSG", dev),
                                 geo=self.try_read("AEC_MIC_ARRAY_GEO", dev))
                self.dev, self.ctl_error, self.baseline = dev, None, {}
            except DeviceNotFound:
                self.ctl_error = "XVF3800 이 USB 에 보이지 않습니다. 보드의 XMOS 쪽 USB-C 포트(3.5mm 잭 옆)에 연결하세요."
            except Exception as e:  # noqa: BLE001
                denied = getattr(e, "errno", None) == 13 or "ccess" in str(e)
                self.ctl_error = ("USB 제어 권한이 없습니다. README 의 udev 규칙을 설치한 뒤 장치를 다시 꽂으세요."
                                  if denied else f"제어 연결 실패: {e}")
        if (self.cap is None or not self.cap.alive) and not self.job.get("running"):
            if self.sim:
                if self.dev:
                    self.cap = audio.SimCapture(self.dev)
            else:
                found = audio.find_card()
                if found:
                    old_err = self.cap.error if self.cap else None
                    self.cap = audio.Capture(found[0], found[1])
                    self.info.update(card=found[0], ch=found[1], rate=found[2], audio_error=old_err)
                else:
                    self.cap = None

    @staticmethod
    def try_read(name, dev):
        try:
            return dev.read(name)
        except Exception:  # noqa: BLE001
            return None

    def poll_loop(self):
        i, fails, last_connect = 0, 0, 0.0
        while True:
            if time.time() - last_connect > 2:
                last_connect = time.time()
                try:
                    self.connect()
                except Exception as e:  # noqa: BLE001
                    self.ctl_error = str(e)
            dev = self.dev
            if dev is None:
                time.sleep(0.2)
                continue
            doa = self.try_read("DOA_VALUE", dev)
            az = self.try_read("AEC_AZIMUTH_VALUES", dev)
            en = self.try_read("AEC_SPENERGY_VALUES", dev)
            if doa is None and az is None:
                fails += 1
                if fails > 15:          # 장치가 뽑힌 것으로 보고 재연결
                    self.dev, fails = None, 0
                    self.ctl_error = "장치와의 제어 연결이 끊겼습니다. 재연결 시도 중…"
                time.sleep(0.2)
                continue
            fails = 0
            azd = [None if (a is None or not math.isfinite(a)) else math.degrees(a) % 360 for a in az] if az else None
            self.live = {"doa": doa[0] if doa else None, "speech": doa[1] if doa else None, "az": azd, "en": en}
            if self.cap:
                self.timeline.append(dict(self.live, n=self.cap.n))
            if i % 8 == 0:
                self.slow = {k: self.try_read(k, dev) for k in SLOW}
            i += 1
            time.sleep(0.07)

    def state(self, seq):
        cap = self.cap
        spec = {"seq": seq, "cols": []}
        if cap:
            cols = [c for s, c in list(cap.spec) if s > seq][-60:]
            spec = {"seq": cap.seq, "cols": cols}
        j = dict(self.job)
        if j.get("running"):
            j["remain"] = max(0.0, j.get("total", 0) - (time.time() - j.get("t0", 0)))
        j.pop("t0", None)
        return clean({
            "ctl_ok": self.dev is not None, "ctl_error": self.ctl_error,
            "audio_ok": bool(cap and cap.alive),
            "audio_error": (cap.error if cap and cap.error else self.info.get("audio_error")) if not (cap and cap.alive) else None,
            "info": self.info, "live": self.live, "slow": self.slow,
            "levels": cap.levels() if cap and cap.alive else [], "spec": spec, "job": j,
        })

    # ---------- 설정 ----------
    def set(self, name, values):
        if name not in P or P[name][3] == "ro":
            raise ValueError(f"쓸 수 없는 명령: {name}")
        if self.dev is None:
            raise RuntimeError("제어 연결이 없습니다")
        if self.job.get("running"):
            raise RuntimeError("테스트가 진행 중일 때는 설정을 바꿀 수 없습니다")
        spec = PARAM_SPEC.get(name)
        if spec and spec[2] == "num" and spec[3]:
            lo, hi = spec[3][0], spec[3][1]
            same_as_base = list(values) == list(self.baseline.get(name) or [])   # 되돌리기는 항상 허용
            if not same_as_base and any(not (lo <= float(v) <= hi) for v in values):
                raise ValueError(f"{spec[1]}: {lo} ~ {hi} 범위로 입력하세요")
        if name in PARAM_SPEC and name not in self.baseline:
            self.baseline[name] = self.try_read(name, self.dev)   # '되돌리기'용으로 바꾸기 직전 값을 기억
        self.dev.write(name, values)

    def params(self):
        """파라미터 탭용: 목록 + 현재 값 + 이 프로그램을 켰을 때의 값."""
        if self.dev is None:
            raise RuntimeError("제어 연결이 없습니다: " + str(self.ctl_error))
        groups = []
        for title, items in PARAMS:
            out = []
            for name, label, kind, spec, desc in items:
                val = self.try_read(name, self.dev)
                if val is not None:
                    self.baseline.setdefault(name, val)
                out.append({"name": name, "label": label, "kind": kind, "spec": spec, "desc": desc,
                            "value": val, "base": self.baseline.get(name)})
            groups.append({"title": title, "items": out})
        return {"groups": groups}

    @contextlib.contextmanager
    def temp(self, settings):
        """테스트 동안만 설정을 바꾸고 끝나면 원래 값으로 되돌린다 (플래시에 저장하지 않음)."""
        old = {}
        try:
            for k, v in settings.items():
                try:
                    old[k] = self.dev.read(k)
                    self.dev.write(k, v)
                except Exception as e:  # noqa: BLE001
                    self.job.setdefault("warnings", []).append(f"{k} 설정 실패: {e}")
            time.sleep(0.4)
            yield
        finally:
            for k, v in old.items():
                with contextlib.suppress(Exception):
                    self.dev.write(k, v)

    # ---------- 작업 실행 ----------
    def start(self, test, params):
        if self.job.get("running"):
            raise RuntimeError("이미 다른 테스트가 진행 중입니다")
        if self.dev is None:
            raise RuntimeError("제어 연결이 없습니다: " + str(self.ctl_error))
        if not (self.cap and self.cap.alive):
            raise RuntimeError("오디오 입력이 없습니다: " + str(self.cap.error if self.cap else "사운드카드를 찾지 못함"))
        fn = getattr(self, "test_" + test)
        self.cancel = False
        self.job = {"running": True, "test": test, "text": "준비 중", "kind": "work", "t0": time.time(),
                    "total": 0, "step": 0, "error": None, "result": None}
        threading.Thread(target=self._run, args=(fn, test, params), daemon=True).start()

    def _run(self, fn, test, params):
        rd = RUNS / (time.strftime("%Y%m%d-%H%M%S") + "_" + test)
        rd.mkdir(parents=True)
        try:
            res = fn(params, rd)
            res.update(test=test, dir=rd.name, time=time.strftime("%Y-%m-%d %H:%M:%S"),
                       warnings=self.job.get("warnings", []),
                       firmware=self.info.get("version"))
            res = clean(res)
            (rd / "result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
            self.job.update(result=res)
        except Cancelled:
            self.job.update(error="취소했습니다")
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self.job.update(error=f"{type(e).__name__}: {e}")
        self.job.update(running=False)

    def phase(self, text, seconds, kind):
        """화면에 안내를 띄우고 seconds 동안 기다린다. 그 구간의 (시작, 끝) 프레임을 돌려준다."""
        self.job.update(text=text, kind=kind, t0=time.time(), total=seconds, step=self.job["step"] + 1)
        f0 = self.cap.n
        end = time.time() + seconds
        while time.time() < end:
            if self.cancel:
                raise Cancelled
            if not self.cap.alive:
                raise RuntimeError("오디오 입력이 끊겼습니다: " + str(self.cap.error))
            time.sleep(0.05)
        return f0, self.cap.n

    def tl(self, f0, f1):
        return [dict(p, t=(p["n"] - f0) / SR) for p in list(self.timeline) if f0 <= p["n"] <= f1]

    def save_audio(self, rd, x, names, f0, f1, marks=()):
        A.write_wav(rd / "stereo_original.wav", x[:, :2])
        files = []
        for i, (ko, _en) in enumerate(names):
            A.write_wav(rd / f"ch{i}.wav", A.normalize(x[:, i]))
            files.append({"label": ko, "file": f"ch{i}.wav"})
        A.plot_spectrograms(rd / "spec.png", [(en, x[:, i]) for i, (_ko, en) in enumerate(names)],
                            marks, self.tl(f0, f1))
        return files

    def play(self, path, seconds):
        if self.sim:
            return audio.play(-1, path, self.dev)
        p = audio.subprocess.Popen(["aplay", "-q", "-D", f"plughw:{self.cap.card},0", "-d", str(int(seconds) + 1), str(path)],
                                   stdout=audio.subprocess.DEVNULL, stderr=audio.subprocess.PIPE)
        time.sleep(0.25)
        if p.poll() not in (None, 0):
            raise RuntimeError("스피커 재생 실패: " + p.stderr.read().decode(errors="ignore").strip())
        return p

    # ---------- 테스트 1: 자유 녹음 A/B ----------
    def test_record(self, p, rd):
        r = ROUTES[p.get("preset", "proc_raw")]
        secs = float(p.get("seconds", 10))
        with self.temp({"AUDIO_MGR_OP_L": r["L"], "AUDIO_MGR_OP_R": r["R"]}):
            self.phase("곧 녹음을 시작합니다", 3, "wait")
            f0, f1 = self.phase("녹음 중 — 말하거나 소리를 내보세요", secs, "do")
        x = self.cap.get(f0, f1)
        rows = []
        for i, (ko, _) in enumerate(r["names"]):
            lo, hi = A.pct(x[:, i], 10), A.pct(x[:, i], 90)
            rows.append([ko, f"{A.rms_db(x[:, i]):.1f}", f"{lo:.1f}", f"{hi:.1f}", f"{hi - lo:.1f}"])
        return {"title": "자유 녹음: " + r["ko"],
                "table": {"head": ["채널", "평균 음량 dBFS", "조용한 구간", "큰 소리 구간", "차이(클수록 또렷)"], "rows": rows},
                "audio": self.save_audio(rd, x, r["names"], f0, f1), "images": ["spec.png"],
                "notes": ["듣기용 파일은 두 채널의 음량을 똑같이 맞춘 것입니다 (원본은 stereo_original.wav).",
                          "맨 아래 그래프는 녹음 중 장치가 보고한 방향(DOA)과 각 빔이 향한 방향입니다."]}

    # ---------- 테스트 2: 방향 탐지(DOA) 정확도 ----------
    def test_doa(self, p, rd):
        targets = [float(a) for a in p.get("angles", [0, 90, 180, 270])]
        speak = float(p.get("speak", 4))
        meas = []
        for a in targets:
            self.phase(f"{a:.0f}° 위치로 이동하세요 (아직 조용히)", 4, "wait")
            if self.sim:
                self.dev.src_deg = (37 - a) % 360
            f0, f1 = self.phase(f"{a:.0f}° 위치에서 계속 말하세요", speak, "do")
            d = [q["doa"] for q in self.tl(f0, f1) if q["speech"] and q["doa"] is not None]
            meas.append((A.circ_mean(d) if len(d) >= 5 else None, len(d)))
        ok = [(t, m) for t, (m, _n) in zip(targets, meas) if m is not None]
        if len(ok) < 2:
            raise RuntimeError("음성이 거의 감지되지 않았습니다. 더 크게, 계속 말해 주세요.")
        best = None
        for sign in (1, -1):   # 장치의 각도 증가 방향이 내 기준과 반대일 수도 있다
            diff = [m - sign * t for t, m in ok]
            off = A.circ_mean(diff)
            err = [abs(A.wrap(d - off)) for d in diff]
            if best is None or np.mean(err) < best[0]:
                best = (float(np.mean(err)), sign, off)
        mean_err, sign, off = best
        rows = []
        for t, (m, n) in zip(targets, meas):
            exp = (sign * t + off) % 360
            rows.append([f"{t:.0f}°", "감지 안 됨" if m is None else f"{m:.0f}°", f"{exp:.0f}°",
                         "-" if m is None else f"{A.wrap(m - exp):+.0f}°", n])
        A.plot_polar(rd / "doa.png",
                     [("expected (after offset fit)", [(sign * t + off) % 360 for t in targets], [1] * len(targets), "s"),
                      ("measured DOA", [m for _t, m in ok], [1] * len(ok), "x")],
                     "DOA accuracy (device angle, deg)", 0, 1.15)
        verdict = ("매우 정확" if mean_err <= 10 else "양호" if mean_err <= 20 else "보통" if mean_err <= 35 else "부정확")
        return {"title": "방향 탐지(DOA) 정확도", "verdict": f"{verdict} — 평균 오차 {mean_err:.0f}°",
                "summary": [["평균 오차", f"{mean_err:.1f}°"], ["내 0° ↔ 장치 각도 차이", f"{off:.0f}°"],
                            ["각도 증가 방향", "내 기준과 같음" if sign == 1 else "내 기준과 반대(거울)"]],
                "table": {"head": ["내가 선 위치", "장치가 잰 방향", "기대값(보정 후)", "오차", "음성 샘플 수"], "rows": rows},
                "images": ["doa.png"],
                "notes": ["보드의 0° 기준과 내가 정한 0° 는 다를 수 있어서, 일정한 차이(offset)는 빼고 오차를 계산합니다.",
                          "판정 기준(10°/20°/35°)은 이 도구가 정한 경험적 기준입니다. 4-마이크 배열은 보통 10~20° 수준입니다."]}

    # ---------- 테스트 3: 빔 패턴 (빔을 돌려가며 측정) ----------
    def test_beamscan(self, p, rd):
        step = int(p.get("step", 20))
        angles = list(range(0, 360, step))
        cur_az = self.dev.read("AEC_FIXEDBEAMSAZIMUTH_VALUES")
        rel, raw = [], []
        with self.temp({"AEC_FIXEDBEAMSAZIMUTH_VALUES": cur_az, "AEC_FIXEDBEAMSONOFF": [1], "AEC_FIXEDBEAMSGATING": [0],
                        "PP_AGCONOFF": [0], "PP_MIN_NS": [1.0], "PP_MIN_NN": [1.0],
                        "AUDIO_MGR_OP_L": (6, 0), "AUDIO_MGR_OP_R": (3, 0)}):
            g0, _ = self.phase("소리(잡음/음악/말소리)를 한 자리에서 계속 틀어 두세요", 4, "wait")
            for a in angles:
                r = math.radians(a)
                self.dev.write("AEC_FIXEDBEAMSAZIMUTH_VALUES", [r, (r + math.pi) % (2 * math.pi)])
                self.phase(f"빔을 {a}° 로 돌리는 중", 1.5, "work")
                f0, f1 = self.phase(f"빔 {a}° 측정 중 — 소리를 유지하세요", 1.5, "do")
                x = self.cap.get(f0, f1)
                rel.append(A.rms_db(x[:, 0]) - A.rms_db(x[:, 1]))   # 원본 마이크로 나눠 소리 크기 변동 제거
                raw.append(A.rms_db(x[:, 1]))
            g1 = self.cap.n
        rel = np.array(rel)
        norm = rel - rel.max()
        k = int(rel.argmax())
        back = norm[(k + len(angles) // 2) % len(angles)]
        width = int((norm >= -3).sum()) * step
        rng = float(-norm.min())
        doa = [q["doa"] for q in self.tl(g0, g1) if q["speech"] and q["doa"] is not None]
        marks = [(A.circ_mean(doa), "DOA")] if len(doa) > 10 else []
        A.plot_polar(rd / "beam.png", [("beam 1 response (dB, 0 = max)", angles, norm, "line")],
                     "Fixed beam steering response", min(-20.0, float(norm.min()) - 2), 1, marks)
        verdict = ("빔 방향성이 뚜렷함" if rng >= 10 else "방향성 보통" if rng >= 5 else "방향성 약함 (소리 크기·잔향·설정 확인)")
        return {"title": "빔 패턴 (고정 빔 조향)", "verdict": f"{verdict} — 최대/최소 차이 {rng:.1f} dB",
                "summary": [["가장 잘 들리는 빔 각도", f"{angles[k]}°"], ["정반대 방향 감쇠", f"{-back:.1f} dB"],
                            ["-3 dB 폭(대략)", f"{width}°"], ["최대/최소 차이", f"{rng:.1f} dB"],
                            ["측정 중 소리 크기 변동", f"±{np.std(raw):.1f} dB"],
                            ["장치 DOA 가 본 소리 방향", f"{marks[0][0]:.0f}°" if marks else "음성 미감지"]],
                "table": {"head": ["빔 각도", "상대 응답 dB"], "rows": [[f"{a}°", f"{v:.1f}"] for a, v in zip(angles, norm)]},
                "images": ["beam.png"],
                "notes": ["소리는 가만히 두고 '귀(빔)'만 전자적으로 한 바퀴 돌린 결과입니다. 소리가 있는 쪽에서 봉우리가 생겨야 정상입니다.",
                          "측정 동안 AGC·잡음억제를 끄고 고정 빔 모드를 켭니다. 끝나면 원래 설정으로 되돌립니다.",
                          "마이크 4개짜리 배열이라 빔이 넓습니다(수십 도). 잔향이 큰 방에서는 차이가 줄어듭니다."]}

    # ---------- 테스트 4: 잡음 제거 ----------
    def test_ns(self, p, rd):
        r = ROUTES["proc_raw"]
        s = {"AUDIO_MGR_OP_L": r["L"], "AUDIO_MGR_OP_R": r["R"]}
        if p.get("freeze_agc", True):
            s["PP_AGCONOFF"] = [0]
        with self.temp(s):
            self.phase("소음원(선풍기, 화이트노이즈 영상 등)을 켜 두세요", 4, "wait")
            f0, f1 = self.phase("말하지 마세요 — 소음만 녹음 중", float(p.get("noise", 6)), "do")
            _, f2 = self.phase("소음 속에서 또박또박 계속 말하세요", float(p.get("speech", 8)), "do")
        x = self.cap.get(f0, f2)
        cut = f1 - f0
        res, rows = [], []
        for i, (ko, _) in enumerate(r["names"]):
            noise = float(np.median(A.frame_db(x[SR:cut, i])))
            speech = A.pct(x[cut:, i], 90)
            res.append(speech - noise)
            rows.append([ko, f"{noise:.1f}", f"{speech:.1f}", f"{speech - noise:.1f}"])
        gain = res[0] - res[1]
        verdict = ("매우 좋음" if gain >= 15 else "좋음" if gain >= 8 else "보통" if gain >= 3 else "효과 미미")
        return {"title": "잡음 제거", "verdict": f"{verdict} — 말소리 대비 잡음이 {gain:+.1f} dB 개선",
                "summary": [["SNR 개선량 (처리 후 − 처리 전)", f"{gain:+.1f} dB"],
                            ["처리 후 SNR", f"{res[0]:.1f} dB"], ["처리 전 SNR", f"{res[1]:.1f} dB"]],
                "table": {"head": ["채널", "소음 구간 dBFS", "말소리 구간 dBFS", "SNR(dB)"], "rows": rows},
                "audio": self.save_audio(rd, x, r["names"], f0, f2, [(0, "noise only"), (cut / SR, "speech + noise")]),
                "images": ["spec.png"],
                "notes": ["SNR = 말소리 크기 − 소음 크기. 채널마다 볼륨이 달라도 공정하게 비교됩니다.",
                          "귀로 확인: 두 파일을 번갈아 들으면서 (1) 배경 '쉬~' 소리가 줄었는지 (2) 목소리가 로봇처럼 뭉개지지 않았는지 보세요.",
                          "스펙트로그램에서는 말소리 줄무늬 사이의 배경이 어두울수록 잡음이 잘 지워진 것입니다.",
                          "판정 기준(3/8/15 dB)은 이 도구의 경험적 기준입니다."]}

    # ---------- 테스트 5: 에코 제거 ----------
    def test_aec(self, p, rd):
        r = ROUTES["proc_raw"]
        wav = (p.get("wav") or "").strip()
        if wav:
            with wave.open(wav) as w:
                if w.getnframes() / w.getframerate() < 10:
                    raise RuntimeError("WAV 파일은 10초 이상이어야 합니다")
        else:
            wav = rd / "play.wav"
            A.write_wav(wav, A.test_signal(14))
        s = {"AUDIO_MGR_OP_L": r["L"], "AUDIO_MGR_OP_R": r["R"]}
        if p.get("freeze_agc", True):
            s["PP_AGCONOFF"] = [0]
        player = None
        try:
            with self.temp(s):
                self.phase("조용히 해 주세요. 스피커는 ReSpeaker 출력에 연결돼 있어야 합니다", 3, "wait")
                f0, f1 = self.phase("조용히 — 배경 소음 측정 중", 2, "do")
                player = self.play(wav, 10)
                _, f2 = self.phase("스피커 재생 중 — 말하지 마세요 (에코만 녹음)", 10, "do")
                conv = self.try_read("AEC_AECCONVERGED", self.dev)
                player.kill()
                _, f3 = self.phase("재생 멈춤 — 잔향이 사라지길 기다리는 중", 1.5, "work")
                _, f4 = self.phase("스피커 없이 — 지금 계속 말하세요", 6, "do")
                player = self.play(wav, 8)
                _, f5 = self.phase("스피커 재생 중 + 동시에 계속 말하세요", 8, "do")
        finally:
            if player:
                player.kill()
        x = self.cap.get(f0, f5)
        a, b, c, d = (f - f0 for f in (f1, f2, f3, f4))
        echo_from = b - int(6 * SR)   # 처음 4초는 AEC 가 방을 '학습'하는 시간이라 뺀다
        rows, rel, base_l, echo_l = [], [], [], []
        for i, (ko, _) in enumerate(r["names"]):
            base = float(np.median(A.frame_db(x[:a, i])))
            echo = A.rms_db(x[echo_from:b, i])
            speech = A.pct(x[c:d, i], 90)
            rel.append(echo - speech)
            base_l.append(base)
            echo_l.append(echo)
            rows.append([ko, f"{base:.1f}", f"{echo:.1f}", f"{speech:.1f}", f"{echo - speech:.1f}"])
        red = rel[1] - rel[0]
        notes = ["에코 감소량 = (원본에서 에코가 말소리보다 얼마나 큰가) − (처리 후 얼마나 큰가). 클수록 좋습니다.",
                 "귀로 확인: 마지막 구간(재생+말하기)에서 '처리된 출력'에 내 목소리만 남고 스피커 소리가 사라졌는지 들어보세요. 목소리가 끊기거나 물속처럼 들리면 더블토크 성능이 약한 것입니다.",
                 "측정 구간은 재생 시작 4초 뒤부터입니다 (AEC 가 스피커→마이크 경로를 학습하는 시간).",
                 "판정 기준(10/20/30 dB)은 이 도구의 경험적 기준입니다."]
        if echo_l[1] - base_l[1] < 6:
            notes.insert(0, "⚠ 원본 마이크에 스피커 소리가 거의 안 잡혔습니다. 스피커가 ReSpeaker 의 3.5mm/JST 출력에 연결됐는지, 볼륨(alsamixer 의 PCM)이 충분한지 확인하세요.")
        if echo_l[0] - base_l[0] < 3:
            notes.insert(0, "처리 후 에코가 배경 소음 수준까지 내려갔습니다. 실제 감소량은 표시된 값 이상입니다.")
        verdict = ("매우 좋음" if red >= 30 else "좋음" if red >= 20 else "보통" if red >= 10 else "약함")
        return {"title": "에코 제거(AEC)", "verdict": f"{verdict} — 에코가 {red:.1f} dB 줄어듦",
                "summary": [["에코 감소량", f"{red:.1f} dB"],
                            ["AEC 수렴 플래그(AEC_AECCONVERGED)", "수렴함" if conv and conv[0] else "수렴 안 됨"],
                            ["처리 후: 에코 − 말소리", f"{rel[0]:.1f} dB"], ["처리 전: 에코 − 말소리", f"{rel[1]:.1f} dB"]],
                "table": {"head": ["채널", "배경 dBFS", "에코만 dBFS", "말소리만 dBFS", "에코−말소리"], "rows": rows},
                "audio": self.save_audio(rd, x, r["names"], f0, f5,
                                         [(0, "quiet"), (a / SR, "echo only"), (b / SR, ""), (c / SR, "speech only"),
                                          (d / SR, "double talk")]),
                "images": ["spec.png"], "notes": notes}

    # ---------- 기타 ----------
    # ---------- 테스트 6: 내 목소리만 남기기 (방향 거름망 보정) ----------
    def test_voicegate(self, p, rd):
        me, ot = float(p.get("me", 90)), float(p.get("others", 270))
        t_me, t_ot, t_mix = float(p.get("t_me", 20)), float(p.get("t_others", 20)), float(p.get("t_mix", 30))
        flip = None
        with self.temp(VG.device_settings(me, ot)):
            self.phase("빔을 맞추는 중 — 모두 조용히", 2, "wait")
            fq = self.phase("모두 조용히 — 배경 소음을 잽니다", 3, "wait")
            if self.sim:
                self.dev.src_deg = me
            fm = self.phase("나만 말하세요 — 평소 말하듯 계속 (다른 사람은 조용히)", t_me, "do")
            if self.sim:
                self.dev.src_deg = ot
            self.phase("이제 나는 조용히", 2, "wait")
            fo = self.phase("상대(두 사람)만 말합니다 — 나는 조용히", t_ot, "do")
            if self.sim:   # 가짜 장치: 4초마다 말하는 사람을 바꾼다
                def flip_loop():
                    k = 0
                    while not stop.is_set():
                        self.dev.src_deg = me if k % 2 == 0 else ot
                        k += 1
                        stop.wait(4)
                stop = threading.Event()
                flip = threading.Thread(target=flip_loop, daemon=True)
                flip.start()
            try:
                fx = self.phase("모두 자유롭게 대화하세요 — 번갈아, 가끔은 겹쳐서", t_mix, "do")
            finally:
                if flip:
                    stop.set()
        xq, xm, xo, xx = (self.cap.get(*f) for f in (fq, fm, fo, fx))
        floor = VG.noise_floor(xq)
        cm, co, cx = (VG.chunks(x, floor) for x in (xm, xo, xx))
        if VG.dur(cm) < 2:
            raise RuntimeError("'나만 말하기' 구간에서 말소리가 거의 잡히지 않았습니다. 더 크게, 계속 말해 주세요.")
        if VG.dur(co) < 2:
            raise RuntimeError("'상대만 말하기' 구간에서 말소리가 거의 잡히지 않았습니다.")
        thr, keep, leak = VG.calibrate(cm, co)
        med_m = float(np.median([c["diff"] for c in cm]))
        med_o = float(np.median([c["diff"] for c in co]))
        if not self.sim:   # 가짜 장치의 결과로 실제 문턱값을 덮어쓰지 않는다
          (BASE / "voicegate.json").write_text(json.dumps({
            "thr": round(thr, 2), "floor": round(floor, 1), "me": me, "others": ot,
            "keep": round(keep, 3), "leak": round(leak, 3), "time": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))
        files = []
        for tag, ko, x, cs in (("mix", "섞인 대화", xx, cx), ("others", "상대만", xo, co), ("me", "나만", xm, cm)):
            y, _utt = VG.apply(x, cs, thr)
            g = 10 ** ((-20 - A.pct(x[:, 0], 95)) / 20)   # 전/후를 같은 배율로 키워야 공정하게 비교된다
            g = min(g, 1000.0)
            A.write_wav(rd / f"{tag}_before.wav", np.clip(x[:, 0] * g, -0.98, 0.98))
            A.write_wav(rd / f"{tag}_after.wav", np.clip(y * g, -0.98, 0.98))
            A.write_wav(rd / f"{tag}_stereo.wav", x[:, :2])
            files += [{"label": f"{ko} — 거르기 전 (내 빔)", "file": f"{tag}_before.wav"},
                      {"label": f"{ko} — 거른 후 (STT 로 넘길 소리)", "file": f"{tag}_after.wav"}]
        A.plot_gate(rd / "gate.png", [("me only", cm, t_me), ("others only", co, t_ot), ("mixed talk", cx, t_mix)], thr)
        rows = []
        for ko, cs in (("나만 말하기", cm), ("상대만 말하기", co), ("섞인 대화", cx)):
            d, pd = VG.dur(cs), VG.dur(VG.passed(cs, thr))
            rows.append([ko, f"{d:.1f}초", f"{pd:.1f}초", f"{100 * pd / d:.0f}%" if d else "-",
                         f"{np.median([c['diff'] for c in cs]):+.1f} dB" if cs else "-"])
        sep = med_m - med_o
        if keep >= 0.95 and leak <= 0.05:
            verdict = "방향만으로 잘 갈립니다"
        elif keep >= 0.85 and leak <= 0.2:
            verdict = "대체로 갈리지만 새는 구간이 있습니다 — 화자 확인을 추가하면 좋습니다"
        else:
            verdict = "방향만으로는 부족합니다 — 배치를 바꾸거나 화자 확인이 필요합니다"
        return {"title": "내 목소리만 남기기 (방향 거름망)",
                "verdict": f"{verdict} — 내 말 보존 {100 * keep:.0f}%, 남의 말 누설 {100 * leak:.0f}%",
                "summary": [["정해진 문턱 (L−R)", f"{thr:+.1f} dB"], ["내 말 보존율", f"{100 * keep:.0f}%"],
                            ["남의 말 누설률", f"{100 * leak:.0f}%"],
                            ["두 빔의 구분 폭", f"{sep:.1f} dB (나 {med_m:+.1f} / 상대 {med_o:+.1f})"],
                            ["배경 소음", f"{floor:.0f} dB"]],
                "table": {"head": ["녹음", "말 구간", "통과", "통과 비율", "L−R 중앙값"], "rows": rows},
                "audio": files, "images": ["gate.png"],
                "notes": ["L−R = 내 쪽 빔이 상대 쪽 빔보다 얼마나 큰가. 내가 말하면 +, 상대가 말하면 − 쪽으로 갑니다.",
                          "보존율·누설률은 '나만'/'상대만' 녹음으로 잰 값이고, 그 녹음으로 문턱도 정했기 때문에 실제보다 좋게 나올 수 있습니다. 섞인 대화는 정답이 없으므로 직접 들어서 판단하세요.",
                          "문턱은 voicegate.json 에 저장했습니다. 테스트가 끝나면 장치 설정은 원래대로 돌아갑니다.",
                          "실제로 쓸 때: python3 voicegate.py setup 으로 장치를 설정하고, 녹음 파일은 python3 voicegate.py file in.wav out.wav 로 거릅니다."]}

    def snapshot(self, seconds):
        d = RUNS / "_snapshot"
        d.mkdir(exist_ok=True)
        x = self.cap.get(self.cap.n - int(SR * seconds))
        out = []
        for i in range(min(2, x.shape[1])):
            A.write_wav(d / f"ch{i}.wav", A.normalize(x[:, i]))
            out.append(f"/runs/_snapshot/ch{i}.wav?t={time.time():.0f}")
        return out

    def runs(self):
        out = []
        for f in sorted(RUNS.glob("*/result.json"), reverse=True)[:60]:
            with contextlib.suppress(Exception):
                out.append(json.loads(f.read_text()))
        return out


class Handler(BaseHTTPRequestHandler):
    app = None

    def log_message(self, *a):
        pass

    def send_json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path, _, query = self.path.partition("?")
        if path == "/api/state":
            seq = int(query.split("seq=")[1].split("&")[0]) if "seq=" in query else 0
            return self.send_json(self.app.state(seq))
        if path == "/api/params":
            try:
                return self.send_json(clean(self.app.params(), 10))
            except Exception as e:  # noqa: BLE001
                return self.send_json({"error": str(e)}, 400)
        if path == "/api/runs":
            return self.send_json({"runs": self.app.runs(), "routes": {k: v["ko"] for k, v in ROUTES.items()}})
        f = (BASE / "static" / "index.html") if path == "/" else (BASE / path.lstrip("/")).resolve()
        if path != "/" and not (f.is_relative_to(RUNS) or f.is_relative_to(BASE / "static")):
            return self.send_error(403)
        if not f.is_file():
            return self.send_error(404)
        data = f.read_bytes()
        start, end, code = 0, len(data) - 1, 200
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):   # 오디오 탐색(seek)용
            s, _, e = rng[6:].partition("-")
            start = int(s or 0)
            end = min(int(e), end) if e else end
            code = 206
        self.send_response(code)
        self.send_header("Content-Type", mimetypes.guess_type(f.name)[0] or "application/octet-stream")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        if code == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        self.wfile.write(data[start:end + 1])

    def do_POST(self):
        app = self.app
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)) or 0) or b"{}")
            if self.path == "/api/run":
                app.start(body["test"], body.get("params", {}))
            elif self.path == "/api/cancel":
                app.cancel = True
            elif self.path == "/api/set":
                app.set(body["name"], body["values"])
            elif self.path == "/api/route":
                r = ROUTES[body["preset"]]
                app.set("AUDIO_MGR_OP_L", r["L"])
                app.set("AUDIO_MGR_OP_R", r["R"])
            elif self.path == "/api/reboot":
                app.set("REBOOT", [1])
                app.baseline = {}
            elif self.path == "/api/save":      # 사용자가 버튼으로 확인했을 때만 호출된다
                app.set("SAVE_CONFIGURATION", [1])
            elif self.path == "/api/clear":
                app.set("CLEAR_CONFIGURATION", [1])
            elif self.path == "/api/snapshot":
                return self.send_json({"files": app.snapshot(float(body.get("seconds", 10)))})
            else:
                return self.send_error(404)
            self.send_json({"ok": True})
        except Exception as e:  # noqa: BLE001
            self.send_json({"error": str(e)}, 400)


def main():
    ap = argparse.ArgumentParser(description="reSpeaker XVF3800 테스트 랩")
    ap.add_argument("--sim", action="store_true", help="장치 없이 가짜 신호로 화면 미리 보기")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    Handler.app = App(args.sim)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"XVF3800 테스트 랩: {url}  (종료: Ctrl+C)")
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, [url]).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
