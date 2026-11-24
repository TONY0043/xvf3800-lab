# reSpeaker XVF3800 테스트 랩

브라우저에서 **보고(방향 다이얼·스펙트로그램·빔 패턴), 듣고(처리 전/후 A/B), 숫자로 판단**하는 테스트 도구.
추가 설치 없이 동작한다 (python3 + numpy + matplotlib + pyusb + alsa-utils).

"내 목소리만 STT 에 넘기기" 방식의 개념과 숫자는 [HANDOFF.md](HANDOFF.md) 에 정리되어 있다.

## 준비 (한 번만)

```bash
# sudo 없이 USB 제어 명령을 보내기 위한 udev 규칙
sudo cp 99-respeaker.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
# 그 다음 장치를 뽑았다 다시 꽂는다
```

- 보드의 **XMOS 쪽 USB-C 포트(3.5mm 잭 옆)** 에 연결한다.
- 에코 제거 테스트를 하려면 스피커를 ReSpeaker 의 **3.5mm 잭 또는 JST 단자**에 연결한다.
- 마이크 구멍(Seeed 로고가 있는 면)이 소리 쪽을 향하게 둔다.

## 실행

```bash
python3 server.py          # 브라우저가 http://127.0.0.1:8765 로 열린다
python3 server.py --sim    # 장치 없이 가짜 신호로 화면 미리 보기
```

## 탭별 테스트

| 탭 | 무엇을 확인하나 | 판단 방법 |
| --- | --- | --- |
| 실시간 | DOA, 빔 4개의 방향·에너지, 음량, 스펙트로그램, 설정 토글 | 말하는 쪽으로 바늘/점이 따라오는지 |
| 방향 탐지 | 0/90/180/270° 에서 말하고 오차 계산 | 평균 오차(°) |
| 빔 패턴·멀티빔 | 고정 빔을 한 바퀴 돌리며 응답 측정 | 극좌표 그림의 봉우리, 앞/뒤 차이(dB) |
| 잡음 제거 | 처리 전/후 동시 녹음 | SNR 개선량(dB) + 직접 듣기 |
| 에코 제거 | 장치 스피커로 재생하며 녹음 | 에코 감소량(dB) + 더블토크 듣기 |
| 자유 녹음 | 빔1↔빔2 등 원하는 두 신호 A/B | 직접 듣기 |
| 내 목소리만 | 빔1=나, 빔2=상대로 두고 두 빔의 크기 차이로 내 말만 남기기 (문턱 보정) | 내 말 보존율·남의 말 누설률(%) + 거르기 전/후 듣기 |
| 파라미터 | 잡음 제거·AGC·AEC·빔포밍·입출력·LED 값 조정, 되돌리기, 플래시 저장 | 값을 바꾼 뒤 위 테스트를 다시 돌려 전/후 비교 |

결과(WAV·PNG·result.json)는 `runs/<시각>_<테스트>/` 에 남는다.

## 내 말만 STT 로 넘기기 (voicegate.py)

"내 목소리만" 탭에서 보정하면 문턱이 `voicegate.json` 에 저장된다.

```bash
python3 voicegate.py setup                 # 장치를 빔1=나 / 빔2=상대, L=빔1, R=빔2, AGC 끔으로 설정
python3 voicegate.py file in.wav out.wav   # 스테레오 녹음에서 내 말 구간만 남김 (+ out.segments.json)
```

## 알아둘 점

- 테스트는 설정을 **임시로만** 바꾸고 끝나면 되돌린다. 플래시 저장은 파라미터 탭의 "플래시에 저장" 버튼을 눌렀을 때만 한다.
  꼬였다면 실시간 탭의 "장치 재부팅" 또는 보드의 RST 버튼.
- 다른 프로그램(화상회의, 녹음 앱)이 XVF3800 마이크를 쓰고 있으면 "오디오 없음"이 뜰 수 있다. 그 프로그램을 닫는다.
- 판정 문구의 기준값(dB, °)은 이 도구가 정한 경험적 기준이다. 절대 평가보다 **설정을 바꿔가며 상대 비교**하는 데 쓰는 것이 정확하다.
- 6채널 펌웨어에서 3~6번 채널이 무음이면 위키대로 `amixer -c <카드> cset numid=8 on,on,on,on,on,on`.

## 참고 (공식 자료)

- 위키: https://wiki.seeedstudio.com/respeaker_xvf3800_introduction/
- 펌웨어·제어 도구: https://github.com/respeaker/reSpeaker_XVF3800_USB_4MIC_ARRAY
- 공식 GUI 앱 reSpeaker Console: https://github.com/respeaker/respeaker-console/releases
- XMOS 튜닝 가이드: https://www.xmos.com/documentation/XM-014888-PC/html/modules/fwk_xvf/doc/user_guide/04_tuning_the_application.html
