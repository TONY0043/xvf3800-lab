"""reSpeaker XVF3800 USB 제어 (공식 python_control/xvf_host.py 의 명령표를 그대로 사용).

명령표 출처: https://github.com/respeaker/reSpeaker_XVF3800_USB_4MIC_ARRAY
"""
import math
import random
import struct
import threading
import time

VID, PID = 0x2886, 0x001A

# 이름: (resid, cmdid, 값 개수, 접근, 타입)
P = {
    "VERSION": (48, 0, 3, "ro", "uint8"),
    "BLD_MSG": (48, 1, 50, "ro", "char"),
    "REBOOT": (48, 7, 1, "wo", "uint8"),
    "SHF_BYPASS": (33, 70, 1, "rw", "uint8"),
    "AEC_MIC_ARRAY_GEO": (33, 74, 12, "ro", "float"),
    "AEC_AZIMUTH_VALUES": (33, 75, 4, "ro", "radians"),
    "AEC_SPENERGY_VALUES": (33, 80, 4, "ro", "float"),
    "AEC_FIXEDBEAMSAZIMUTH_VALUES": (33, 81, 2, "rw", "radians"),
    "AEC_FIXEDBEAMSELEVATION_VALUES": (33, 82, 2, "rw", "radians"),
    "AEC_FIXEDBEAMSGATING": (33, 83, 1, "rw", "uint8"),
    "AEC_AECPATHCHANGE": (33, 0, 1, "ro", "int32"),
    "AEC_HPFONOFF": (33, 1, 1, "rw", "int32"),
    "AEC_AECCONVERGED": (33, 3, 1, "ro", "int32"),
    "AEC_RT60": (33, 9, 1, "ro", "float"),
    "AEC_ASROUTONOFF": (33, 35, 1, "rw", "int32"),
    "AEC_FIXEDBEAMSONOFF": (33, 37, 1, "rw", "int32"),
    "AUDIO_MGR_MIC_GAIN": (35, 0, 1, "rw", "float"),
    "AUDIO_MGR_REF_GAIN": (35, 1, 1, "rw", "float"),
    "AUDIO_MGR_SELECTED_AZIMUTHS": (35, 11, 2, "ro", "radians"),
    "AUDIO_MGR_OP_L": (35, 15, 2, "rw", "uint8"),
    "AUDIO_MGR_OP_R": (35, 19, 2, "rw", "uint8"),
    "AUDIO_MGR_SYS_DELAY": (35, 26, 1, "rw", "int32"),
    "AUDIO_MGR_OP_CH3": (35, 28, 2, "rw", "uint8"),
    "AUDIO_MGR_OP_CH4": (35, 29, 2, "rw", "uint8"),
    "AUDIO_MGR_OP_CH5": (35, 30, 2, "rw", "uint8"),
    "AUDIO_MGR_OP_CH6": (35, 31, 2, "rw", "uint8"),
    "GPO_READ_VALUES": (20, 0, 5, "ro", "uint8"),
    "LED_EFFECT": (20, 12, 1, "rw", "uint8"),
    "DOA_VALUE": (20, 18, 2, "ro", "uint16"),
    "PP_AGCONOFF": (17, 10, 1, "rw", "int32"),
    "PP_AGCMAXGAIN": (17, 11, 1, "rw", "float"),
    "PP_AGCDESIREDLEVEL": (17, 12, 1, "rw", "float"),
    "PP_AGCGAIN": (17, 13, 1, "rw", "float"),
    "PP_LIMITONOFF": (17, 19, 1, "rw", "int32"),
    "PP_MIN_NS": (17, 21, 1, "rw", "float"),
    "PP_MIN_NN": (17, 22, 1, "rw", "float"),
    "PP_ECHOONOFF": (17, 23, 1, "rw", "int32"),
    "PP_NLATTENONOFF": (17, 27, 1, "rw", "int32"),
    "PP_DTSENSITIVE": (17, 31, 1, "rw", "int32"),
    # --- 파라미터 탭에서 쓰는 나머지 조정 항목 ---
    "SAVE_CONFIGURATION": (48, 9, 1, "wo", "uint8"),
    "CLEAR_CONFIGURATION": (48, 10, 1, "wo", "uint8"),
    "AIC3104_HP_LEVEL": (48, 11, 1, "rw", "uint8"),
    "AIC3104_LINEOUT_LEVEL": (48, 12, 1, "rw", "uint8"),
    "AEC_AECEMPHASISONOFF": (33, 4, 1, "rw", "int32"),
    "AEC_FAR_EXTGAIN": (33, 5, 1, "rw", "float"),
    "AEC_PCD_COUPLINGI": (33, 6, 1, "rw", "float"),
    "AEC_PCD_MINTHR": (33, 7, 1, "rw", "float"),
    "AEC_PCD_MAXTHR": (33, 8, 1, "rw", "float"),
    "AEC_ASROUTGAIN": (33, 36, 1, "rw", "float"),
    "AEC_FIXEDBEAMNOISETHR": (33, 38, 2, "rw", "float"),
    "LED_BRIGHTNESS": (20, 13, 1, "rw", "uint8"),
    "LED_SPEED": (20, 15, 1, "rw", "uint8"),
    "LED_COLOR": (20, 16, 1, "rw", "uint32"),
    "PP_AGCTIME": (17, 14, 1, "rw", "float"),
    "PP_AGCFASTTIME": (17, 15, 1, "rw", "float"),
    "PP_AGCALPHAFASTGAIN": (17, 16, 1, "rw", "float"),
    "PP_AGCALPHASLOW": (17, 17, 1, "rw", "float"),
    "PP_AGCALPHAFAST": (17, 18, 1, "rw", "float"),
    "PP_LIMITPLIMIT": (17, 20, 1, "rw", "float"),
    "PP_GAMMA_E": (17, 24, 1, "rw", "float"),
    "PP_GAMMA_ETAIL": (17, 25, 1, "rw", "float"),
    "PP_GAMMA_ENL": (17, 26, 1, "rw", "float"),
    "PP_FMIN_SPEINDEX": (17, 30, 1, "rw", "float"),
    "PP_ATTNS_MODE": (17, 32, 1, "rw", "int32"),
    "PP_ATTNS_NOMINAL": (17, 33, 1, "rw", "float"),
    "PP_ATTNS_SLOPE": (17, 34, 1, "rw", "float"),
}
FMT = {"uint8": "B", "char": "B", "uint16": "H", "int32": "i", "uint32": "I",
       "float": "f", "radians": "f"}
SIZE = {"B": 1, "H": 2, "i": 4, "I": 4, "f": 4}


class DeviceNotFound(Exception):
    pass


class XVF:
    """실제 장치. 모든 접근은 lock 으로 직렬화한다."""
    sim = False

    def __init__(self):
        import usb.core
        self.dev = usb.core.find(idVendor=VID, idProduct=PID)
        if self.dev is None:
            raise DeviceNotFound("USB 장치 2886:001a 를 찾을 수 없습니다")
        self.lock = threading.RLock()

    def read(self, name):
        resid, cmd, n, _acc, typ = P[name]
        f = FMT[typ]
        length = n * SIZE[f] + 1  # 첫 바이트는 상태 코드
        with self.lock:
            for _ in range(100):
                r = self.dev.ctrl_transfer(0xC0, 0, 0x80 | cmd, resid, length, 2000)
                if r[0] == 0:
                    break
                if r[0] != 64:  # 64 = 재시도 요청
                    raise IOError(f"{name}: 상태 코드 {r[0]}")
                time.sleep(0.01)
            else:
                raise IOError(f"{name}: 재시도 100회 초과")
        body = bytes(r[1:1 + n * SIZE[f]])
        if typ == "char":
            return body.rstrip(b"\x00").decode("utf-8", errors="ignore")
        return list(struct.unpack("<" + f * n, body))

    def write(self, name, values):
        resid, cmd, n, acc, typ = P[name]
        if acc == "ro":
            raise ValueError(f"{name} 은 읽기 전용")
        if len(values) != n:
            raise ValueError(f"{name}: 값 {n}개 필요")
        f = FMT[typ]
        vals = [float(v) if f == "f" else int(v) for v in values]
        payload = struct.pack("<" + f * n, *vals)
        with self.lock:
            self.dev.ctrl_transfer(0x40, 0, cmd, resid, payload, 2000)


class SimXVF:
    """장치 없이 화면과 분석 코드를 확인하기 위한 가짜 장치."""
    sim = True

    def __init__(self):
        self.lock = threading.RLock()
        self.t0 = time.time()
        self.src_deg = 120.0   # 가상 화자 위치
        self.playing = False   # 가상 스피커 재생 중
        self.v = {
            "VERSION": [2, 1, 1], "BLD_MSG": "SIMULATION",
            "AEC_MIC_ARRAY_GEO": [0.033, -0.033, 0, 0.033, 0.033, 0, -0.033, 0.033, 0, -0.033, -0.033, 0],
            "AEC_FIXEDBEAMSAZIMUTH_VALUES": [0.0, 3.14], "AEC_FIXEDBEAMSELEVATION_VALUES": [0.0, 0.0],
            "AEC_FIXEDBEAMSGATING": [0], "AEC_FIXEDBEAMSONOFF": [0], "AEC_AECCONVERGED": [1],
            "AEC_RT60": [0.41], "AEC_ASROUTONOFF": [0], "AEC_HPFONOFF": [1], "AEC_AECPATHCHANGE": [0],
            "AUDIO_MGR_MIC_GAIN": [90.0], "AUDIO_MGR_REF_GAIN": [8.0], "AUDIO_MGR_SYS_DELAY": [12],
            "AUDIO_MGR_OP_L": [8, 0], "AUDIO_MGR_OP_R": [0, 0], "LED_EFFECT": [4],
            "PP_AGCONOFF": [1], "PP_AGCMAXGAIN": [64.0], "PP_AGCGAIN": [2.0], "PP_MIN_NS": [0.15],
            "PP_MIN_NN": [0.51], "PP_ECHOONOFF": [1], "PP_DTSENSITIVE": [0], "SHF_BYPASS": [0],
        }

    def speech(self):
        return (time.time() - self.t0) % 4 < 2.8

    def read(self, name):
        t = time.time() - self.t0
        if name == "DOA_VALUE":
            return [int((self.src_deg + random.gauss(0, 4)) % 360), int(self.speech())]
        if name == "AEC_AZIMUTH_VALUES":
            r = math.radians
            return [r(self.src_deg), r(300), r((t * 40) % 360), r(self.src_deg)]
        if name == "AEC_SPENERGY_VALUES":
            s = self.speech()
            return [3e5 * s * random.random(), 0.0, 2e5 * s * random.random(), 3e5 * s * random.random()]
        if name == "PP_AGCGAIN" and self.v["PP_AGCONOFF"][0]:
            return [2.0 + 3 * abs(math.sin(t / 5))]
        v = self.v.setdefault(name, [0] * P[name][2])
        return v if isinstance(v, str) else list(v)

    def write(self, name, values):
        if name in ("REBOOT", "CLEAR_CONFIGURATION"):
            self.__init__()
            return
        if name == "SAVE_CONFIGURATION":
            return
        self.v[name] = list(values)
