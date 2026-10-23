# Tesseract 오버레이 코어 — 아키텍처 & ISA 스펙 v0.2

Oct 22, 2026 · @Jiwoo KIm

## 개요

Tesseract는 llama.cpp가 넘기는 ggml 그래프를 명령어 스트림으로 바꿔 실행하는 FPGA 오버레이 코어로, 바닥부터 직접 설계한다. v1 목표는 Alveo U200에서 32B Q4\_K\_M GGUF 모델을 변환 없이 llama.cpp로 끝까지 돌리는 것이고, 장기 목표는 MoE 지원과 호스트가 거의 끼어들지 않는 디바이스 상주 디코드다.

**v1 목표**

- 실제 배포되는 GGUF 파일을 그대로 로드: 하드웨어는 8비트 Q8\_0과 4비트 Q4\_K 두 포맷만 지원하고, 그 밖의 타입으로 된 가중치는 로드 시 Q8\_0으로 변환한다 (Q4\_K\_M 파일에 섞인 Q6\_K 포함)
- llama.cpp 백엔드 구조에 맞는 ggml 백엔드 `ggml-tesseract`
- 코어 RTL은 벤더 독립 SystemVerilog, 기술 래퍼와 플랫폼 어댑터만 타깃별 구현
- 하드웨어와 소프트웨어 사이의 유일한 계약은 이 문서의 ISA와 메모리 레이아웃
- ISA는 처음부터 정수 레지스터와 "레지스터 + 즉시값" 주소로 정의해, 장기 목표를 인코딩 변경 없이 수용
- 성능 기준: 정확하게 끝까지 도는 것이 우선, 32B Q4\_K\_M 디코드 1 tok/s 이상 (4채널)

**v1 비목표**

- 5·6비트(Q5\_K, Q6\_K)와 I-quant의 하드웨어 경로: 계획 없음, 로드 시 Q8\_0 변환으로 대신한다. 4비트 계열 Q4\_0, MXFP4는 보류
- 64GB에 들어가지 않는 모델 (gpt-oss-120b, GLM-4.5-Air급)
- 학습, 다중 카드, 다중 세션 서빙

**장기 목표 (v1 이후, 마일스톤 M5\~M9)**

- **MoE**: `MUL_MAT_ID`를 호스트 보조 방식으로 먼저 지원하고(M5), 디바이스가 라우팅까지 하는 방식으로 옮긴다(M8). 대역폭은 낮고 용량은 큰 U200에 가장 잘 맞는 워크로드다.
- **호스트 독립**: 전체 그래프 디바이스화(단계 A, M7) → 위치 독립 프로그램(단계 B, M8) → 디바이스 자체 디코드 루프(단계 C, M9). 자세한 내용은 "호스트 독립 로드맵" 절.

**용어**

| 용어 | 뜻 |
| --- | --- |
| 채널 | 독립 DRAM 포트 1개. U200에서는 DDR4 뱅크 1개 |
| 클러스터 | 채널 1개를 전담하는 MVM 엔진 |
| beat | 메모리 인터페이스 1회 전송 단위. 512비트 = 64B |
| SPM | 온칩 스크래치패드. 활성값과 중간값 저장 |
| 슈퍼블록 (sb) | K-quant의 가중치 256개 단위 |
| QA | 양자화 활성값. int8 데이터 + 스케일 + bsums |
| N, K | 가중치 행렬 W의 행 수(출력 차원), 열 수(입력 차원) |
| T | MVM 한 번이 처리하는 토큰(활성값 벡터) 수 |
| R0\~R15 | 정수 레지스터 (int32, R0은 항상 0). 주소, 루프 횟수, 인자 |
| s0\~s15 | f32 스칼라 레지스터. 리덕션 결과와 상수 |
| 단계 A·B·C | 호스트 독립의 세 단계 ("호스트 독립 로드맵" 절) |

## 시스템 구조

시스템은 4계층으로 나뉘고, 타깃이 바뀌면 아래 두 계층(기술 래퍼, 플랫폼 어댑터)만 새로 쓴다. 소프트웨어와 코어는 파라미터만 바뀐다.

| 계층 | 구성 요소 | 타깃 의존성 |
| --- | --- | --- |
| 소프트웨어 | `ggml-tesseract` 백엔드, 가중치 리패커, 그래프 lowering, ISA 시뮬레이터, 전송 계층 | 전송 계층만 (XRT, Zynq AXI, SoC 드라이버) |
| 코어 | 커맨드 프로세서(정수 레지스터 R0\~R15 포함), MVM 클러스터 × C, 벡터 유닛, f32 스칼라 유닛, SPM, DMA | 파라미터만 |
| 기술 래퍼 | `tsr_sram_*`, `tsr_mac_lane`, 클럭·리셋, CDC 동기화기 | 타깃별 구현 |
| 플랫폼 어댑터 | 제어 레지스터 브리지, 메모리 포트 → AXI4 등 변환 | 타깃별 구현 |

**U200 구성: 가중치 스트림은 채널마다 독립된 클러스터로만 흐른다** (그림 `docs/fig/u200-system.jpg`)

- 호스트 (llama.cpp + ggml-tesseract): 그래프 → 명령 버퍼 lowering, 가중치 리패킹, 전송 계층(XRT 또는 시뮬레이터). PCIe로 명령 버퍼, 가중치, 활성값을 보낸다.
- 플랫폼 어댑터 (Alveo XRT): 제어 브리지(`s_axi_control` ↔ CSR, `ap_done` ↔ `done`), 채널별 AXI4 메모리 브리지. 코어와는 CSR 포트와 `done`으로 이어진다.
- 코어 (벤더 독립 SystemVerilog): 커맨드 프로세서(명령어 fetch·decode, 순차 실행, R0\~R15), SPM(활성값·중간값·QA, line 64B 단위, 4 MiB), 벡터·스칼라 유닛(원소 연산, 리덕션, RoPE, VL 16, s0\~s15), 클러스터 0\~3(스트림·언팩, MAC 128, fp 누산). SPM과 클러스터 사이는 x 브로드캐스트·y 수집 버스다.
- DDR4 ch0\~ch3: 클러스터 c는 채널 c에서만 가중치 스트림을 받는다(채널당 최대 19.2 GB/s = 300 MHz × 64B, AXI4 메모리 브리지 경유). ch0은 가중치 행 1/4과 CMD·IO, ch1\~ch3은 가중치 행 1/4씩이다.

무거운 가중치 트래픽은 각 채널과 그 채널의 클러스터 사이에서만 오가고, SLR을 넘는 것은 수 KB짜리 활성값 브로드캐스트와 결과 수집뿐이다.

**코어 경계 인터페이스**

코어는 AXI를 직접 쓰지 않는다. 아래 세 가지만 밖으로 낸다.

- **CSR 포트**: 32비트 레지스터 읽기·쓰기 (`addr`, `wdata`, `we`, `rdata`, `valid/ready`). 어댑터가 `s_axi_control` 등으로 변환한다.
- **메모리 포트 × C**: 채널마다 1개. 읽기 요청 `{addr_beat, len_beats, id}`, 읽기 응답 `{data[511:0], id, last}`, 쓰기 요청·데이터·응답 채널. 모두 valid/ready.
- **완료 신호**: `done` 레벨 출력. 어댑터가 `ap_done`이나 인터럽트로 변환한다.

메모리 포트 규칙:

- 주소 단위는 beat(64B). 코어는 4KB 경계를 넘는 버스트를 내지 않는다 (최대 64 beat, 4KB 정렬 내).
- 읽기 응답은 같은 `id` 안에서만 순서 보장. 코어는 채널당 최대 `RD_OUTSTANDING`개 요청을 미처리 상태로 둘 수 있다.
- 채널 포트 하나를 클러스터 가중치 스트림과 DMA가 공유하며, 코어 내부 중재기가 나눈다. 명령어 fetch는 채널 0 포트를 쓴다.

## 구성 파라미터

모든 크기는 엘라보레이션 파라미터로 받고, 소프트웨어는 시작 시 CSR `CFG_*` 레지스터에서 같은 값을 읽는다. U200 v1 값은 아래와 같다.

| 파라미터 | 의미 | U200 v1 |
| --- | --- | --- |
| `NUM_CH` (C) | 채널 수 = 클러스터 수 (1\~16, `LD`·`ST`의 4비트 채널 필드가 상한) | 4 |
| `BEAT_BITS` | 메모리 포트 데이터 폭 | 512 |
| `F_CORE` | 코어 클럭 목표 | 300 MHz |
| `LANES` | 클러스터당 MAC 레인 수 (beat당 가중치 처리량). `BEAT_BITS` / 4(Q4\_K beat 하나의 가중치 수)로 정해지며 독립 파라미터가 아니다 | 128 |
| `T_MAX` | MVM 한 번의 최대 토큰 수 | 합성 1, 아키텍처 상한 8 |
| `K_MAX` | 지원 최대 K | 32768 |
| `SPM_BYTES` | 스크래치패드 크기 | 4 MiB |
| `VL` | 벡터 유닛 fp32 레인 수 | 16 |
| `NIREG` | 정수 레지스터 수 (R0 = 0 고정) | 16 |
| `NSREG` | f32 스칼라 레지스터 수 | 16 |
| `LOOP_DEPTH` | LOOP 중첩 깊이 (M8부터) | 4 |
| `RD_OUTSTANDING` | 채널당 미처리 읽기 요청 수 | 16 |
| `CH_BYTES` | 채널당 DRAM 용량 | 16 GiB |
| `ADDR_BEAT_W` | beat 단위 채널 주소 폭 | 28 |

`K_MAX`는 Qwen2.5-32B FFN 중간 차원 27648, Llama 70B급 28672를 덮는 값으로 잡았다. 클러스터의 활성값 버퍼는 토큰마다 QA 벡터 하나 전체(QS + D + BS, "양자화 활성값(QA) 레이아웃")를 담으므로 `QA_LINES(A256, K_MAX) × 64 × T_MAX` 바이트가 필요하다. K\_MAX = 32768에서 토큰당 584 line = 37,376 B다 (A32는 576 line = 36,864 B, A256이 더 크다). v0.2.1 정정: 이전 판은 `K_MAX × T_MAX` 바이트(QS만, 32 KiB)라 적어 Dx·BS 자리가 빠졌고, 그 크기로는 K > 28672(A256)·K > 29120(A32)인 MVM의 QA가 들어가지 않는다 (예: Qwen2.5-72B FFN K = 29568, Q8\_0 → A32 520 line = 33,280 B).

이식 예: HBM 카드(U280, U55C)는 `NUM_CH`가 크고 `BEAT_BITS`가 256으로 줄어든다. Zynq 계열은 `NUM_CH` 1\~2, `BEAT_BITS` 128 수준이다. 어느 경우든 클러스터 구조와 ISA는 그대로다. 단 `NUM_CH`는 16 이하다 (v0.2.1 명확화): `LD`·`ST`는 채널을 w0의 4비트 sub 필드로 지정하므로 채널 16 이상에는 닿지 못한다. CFG0의 NUM\_CH 필드는 8비트지만 16을 넘는 구성은 정의하지 않으며, 시뮬레이터와 도구는 거부한다. 의사 채널이 16개를 넘는 HBM(U280·U55C는 32개)은 의사 채널 여러 개를 채널 하나로 묶어 `NUM_CH` ≤ 16으로 맞춘다. 나중에 채널을 더 늘려야 하면 `LD`·`ST`의 flags\[15:12\](지금은 예약, 0)를 채널 상위 비트로 넓힐 수 있고, 기존 프로그램은 그대로 유효하다.

## 메모리 모델

코어가 보는 주소 공간은 DRAM(채널별)과 SPM 두 개뿐이다. 호스트는 DRAM에만 접근하고 SPM은 직접 보지 못한다.

**DRAM 주소**: `(ch, addr_beat)`. 채널 번호와 그 채널 안의 beat(64B) 단위 오프셋이다.

**대칭 할당**: 양자화 가중치 텐서 하나는 모든 채널의 같은 오프셋 `W_OFF`에 놓인다. 행 N개를 연속 구간으로 C등분한다.

- R = ceil(N / C), 채널 c는 행 \[c·R, min((c+1)·R, N))을 가진다. 빈 구간도 허용한다.
- 채널 안에서 행 r의 시작 = `W_OFF + (r − c·R) × ROW_BEATS(fmt, K)`. `ROW_BEATS`는 다음 절에서 정의한다.
- 덕분에 MVM 명령어는 텐서 주소를 오프셋 하나로 표현하고, 클러스터는 자기 채널 구간만 독립적으로 처리한다.

**전문가 텐서 (MoE)**: ggml의 3D 전문가 텐서 `[K, N, n_expert]`는 전문가마다 N×K 행렬 하나로 보고, 각 전문가를 위와 같이 4채널에 행 분할한다. 전문가 e는 `W_OFF + e × EXP_STRIDE`에 놓이며, `EXP_STRIDE = R × ROW_BEATS`(채널당 beat)다. 덕분에 전문가 MVM 하나도 4채널 대역폭을 모두 쓰고, 디바이스가 계산한 전문가 번호로 주소를 만들 때 곱셈 한 번이면 된다.

**채널별 DRAM 맵** (모든 채널이 같은 맵을 쓰고, v1에서는 채널 1\~3의 하위 영역은 비워 둔다)

| 영역 | 범위 (채널 내) | 용도 |
| --- | --- | --- |
| CMD | 0 \~ 64 MiB (ch0) | 명령 버퍼, 더블 버퍼링. M9부터 상주 프로그램 |
| IO | 64 \~ 320 MiB (ch0) | 호스트 ↔ 코어 입출력: 활성값, 입력 텐서(토큰 id, 위치, 마스크, KV 인덱스), 출력 토큰 링버퍼. v1 소프트웨어는 아래쪽을 compute 버퍼 창, 맨 위를 스테이징 영역으로 쓴다 ("v1 lowering 제약") |
| AUX | 320 \~ 512 MiB (ch0) | 작은 F32 텐서: norm 가중치, RoPE 각속도 테이블 |
| WEIGHT | 512 MiB \~ 끝 (전 채널) | 리패킹된 가중치와 전문가 텐서, 대칭 할당 |
| KV | 미정 (M7) | 디바이스 KV 캐시. 위치는 미결정 사항 |

32B Q4\_K\_M(약 20GB)은 채널당 약 5GB, Q8\_0(약 35GB)도 채널당 약 9GB라 여유가 있다.

**SPM 규칙**

- 선형 주소, 단위는 line(64B). 모든 피연산자는 line 경계에서 시작한다.
- SPM 데이터에는 타입 태그가 없다. 해석은 명령어가 정한다 (F32 벡터, QA 블록 등).
- 쓰기는 line 단위로 이뤄진다. 출력 벡터의 마지막 line에서 유효 원소 뒤 꼬리 바이트는 정의되지 않는다. lowering은 서로 다른 피연산자가 line을 공유하지 않게 배치해야 한다.

## 디바이스 가중치 레이아웃

가중치는 업로드 시 청크 단위로 재배열된다. 청크 = 메타 beat 1개 + 양자 beat 여러 개이고, 비트 의미는 ggml과 같고 바이트 위치만 바뀐다. GGUF 대비 대역폭 오버헤드는 Q8\_0과 Q4\_K 모두 0%다.

| fmt | 청크당 블록 | 청크당 가중치 | 메타 beat 내용 | 양자 beat | 청크 크기 | GGUF 대비 |
| --- | --- | --- | --- | --- | --- | --- |
| Q8\_0 | 32 블록 | 1024 | 블록별 `d` fp16 × 32 = 64B | 16 | 1088B | 0% |
| Q4\_K | 4 sb | 1024 | sb별 16B: `d`, `dmin`, `scales[12]` | 8 | 576B | 0% |

**포맷별 바이트 배치** (청크 안 블록 번호 b 또는 sb 번호 s 기준, 모든 다중 바이트 값은 little-endian)

- **Q8\_0**: 메타 바이트 \[2b, 2b+2) = `d[b]`. 블록 b의 `qs[32]`는 양자 beat ⌊b/2⌋의 바이트 \[32·(b mod 2), +32). 양자 beat j는 청크 가중치 \[64j, 64j+64)를 담는다.
- **Q4\_K**: 메타 바이트 \[16s, 16s+16) = 원본 블록의 앞 16바이트 (`d`, `dmin`, `scales[12]`) 그대로. 원본 `qs[128]`을 그대로 양자 beat 2s, 2s+1에 둔다. beat 2s는 sb 가중치 \[0, 128), beat 2s+1은 \[128, 256)을 담는다.

**로드 시 변환**: 하드웨어 포맷은 Q8\_0과 Q4\_K 두 개뿐이다. 그 밖의 타입으로 된 2D 가중치(Q6\_K, Q5\_K, Q5\_0, Q4\_0, F16, BF16, F32, I-quant 등)는 `set_tensor`에서 ggml의 `to_float`로 푼 뒤 `quantize_row_q8_0_ref`로 다시 양자화해 Q8\_0으로 저장한다. 그래서 어떤 GGUF든 로드는 되고, 8·4비트가 아닌 텐서만 8비트 대역폭을 쓴다.

- Q4\_K\_M 파일은 attn\_v와 ffn\_down의 절반, output이 Q6\_K라 이 경로를 탄다. Qwen2.5-32B Q4\_K\_M 기준 디바이스 용량과 토큰당 읽는 양이 약 1.3GB(약 6%) 늘고, 디코드도 그만큼 느려진다.
- 정확도 (실측, `docs/numcheck.md` B3): fp64 정확값 대비 상대오차를 CPU와 비교하면 Q6\_K → Q8\_0은 CPU보다 오히려 작다(32B 중앙값 0.62배). CPU는 Q6\_K를 Q8\_K 활성값(256개당 스케일 1개)과 곱하고, 변환 경로는 A32(32개당 1개)를 쓰기 때문이다. Q5\_0 → Q8\_0은 CPU의 약 1.25배다. Q5\_0 격자점이 Q8\_0 격자 위에 있지 않아 재양자화가 가중치마다 최대 Q8\_0 반 칸을 더한다. 둘 다 이 설계에서 예상되는 값이고 결함이 아니다. K가 256의 배수가 아니면 llama-quantize가 Q4\_K 대신 Q5\_0을 쓰므로 작은 모델에서 나타난다.
- 6비트 값을 8비트로 다시 양자화하는 오차는 원래 6비트 양자화 오차보다 훨씬 작다. 모델 단위 KL 기준으로 확인한다.
- MoE 라우터(`ffn_gate_inp`)는 `llama-quantize`가 F32로 남기므로 역시 Q8\_0으로 바뀐다. 라우팅이 바뀌는지 M5에서 KL로 확인한다.
- 1D 텐서(norm 가중치 등)는 변환하지 않고 F32 그대로 AUX에 둔다.

nibble과 6비트 스케일 패킹 같은 비트 수준 해석은 `ggml-quants.c`의 `dequantize_row_q4_K`, `get_scale_min_k4`, `dequantize_row_q8_0`을 따른다. 패킹 풀기는 RTL에서 비트 슬라이싱으로 처리한다.

**행 크기**: 블록 수 nb = K / 블록크기 (Q8\_0 32, K-quant 256), 청크당 블록 S, 블록당 양자 바이트 QB (Q8\_0 32, Q4\_K 128)일 때, full = ⌊nb/S⌋, rem = nb mod S로 둔다. 마지막 부분 청크는 메타 beat(빈 항목은 0) + 실제 블록만큼의 양자 beat를 가진다. 모든 행은 beat 경계에서 시작한다.

```latex
\mathrm{ROW\_BEATS} = \mathrm{full}\left(1 + \left\lceil \tfrac{S \cdot QB}{64} \right\rceil\right) + [\mathrm{rem} > 0]\left(1 + \left\lceil \tfrac{\mathrm{rem} \cdot QB}{64} \right\rceil\right)
```

예: Qwen2.5-32B `ffn_down`(Q4\_K, K = 27648)은 nb = 108, full = 27, rem = 0이라 행당 243 beat(15552B)로 GGUF와 같다. K = 5120인 Q8\_0 행은 nb = 160, full = 5, rem = 0이라 85 beat(5440B)로 역시 GGUF와 같다.

K는 블록 크기의 배수여야 한다 (ggml이 양자화 텐서에 대해 보장). 리패킹은 `ggml-tesseract`의 `set_tensor`에서 하고, 레퍼런스 구현은 ISA 시뮬레이터와 같은 코드를 쓴다.

## 양자화 활성값(QA) 레이아웃

MVM의 입력 활성값은 int8로 양자화된 QA 형식이며, ggml CPU 백엔드의 활성값 형식(`block_q8_0`, `block_q8_K`)에 대응하고 값은 ggml 레퍼런스 양자화 함수(`quantize_row_q8_0_ref`, `quantize_row_q8_K_ref`)와 같은 규칙으로 만든다. 가중치 fmt가 QA 종류를 정한다: Q8\_0 → A32, Q4\_K → A256. 종류가 맞지 않으면 MVM은 오류로 멈춘다.

- v0.2.1 명확화: CPU 백엔드가 실제로 만드는 활성값과 항상 같지는 않다. x86 CPU 백엔드의 Q8\_K(`quantize_row_q8_K`)는 레퍼런스 함수를 그대로 부르지만, Q8\_0 활성값은 SIMD 구현 `quantize_row_q8_0`(AVX 경로: id = 127 / amax를 한 번에 반올림, q는 짝수 쪽 반올림)으로 만든다. 그래서 A32는 x86 CPU 백엔드의 Q8\_0 활성값과 드물게 다르다 (예: x = {127, 2.5, −0.5, 0, …}에서 CPU는 q = {127, 2, 0}, A32와 `quantize_row_q8_0_ref`는 {127, 3, −1}. 무작위 입력에서 q 수십만 개 중 1개꼴). CPU 백엔드의 Q8\_0 활성값이나 출력을 QACT A32의 비트 오라클로 쓰면 안 되며, 비트 기준은 레퍼런스 함수다.

| 종류 | ggml 대응 | 블록 크기 | 스케일 | bsums |
| --- | --- | --- | --- | --- |
| A32 | `block_q8_0` | 32 | 블록당 f32 1개 | 없음 |
| A256 | `block_q8_K` | 256 | 블록당 f32 1개 | 16개 단위 int16 합 |

**SPM 배치** (토큰 하나, 길이 K): 세 영역이 이 순서로 각각 line 경계에서 시작한다.

1. `QS`: int8 × K, ⌈K/64⌉ line
2. `D`: f32 × (K / 블록크기), ⌈4·K/블록크기/64⌉ line
3. `BS` (A256만): int16 × (K/16), ⌈2·(K/16)/64⌉ line

세 영역 합을 `QA_LINES(kind, K)`라 한다. T개 토큰은 QA 벡터 T개를 stride `QA_LINES`로 연속 배치한다.

**양자화 규칙** (`QACT` 명령어, ISA 시뮬레이터, RTL이 비트 단위로 일치해야 하며 ggml 레퍼런스와도 일치하도록 정의)

- **A32** (`quantize_row_q8_0_ref`와 동일): amax = 블록의 max|x|, d = amax / 127, id = (d ≠ 0) ? 1/d : 0 (모두 f32). q = roundf(x · id), 반올림은 0에서 먼 쪽(half away from zero). 저장하는 `D`는 d를 fp16으로 반올림한 뒤 f32로 되돌린 값이다. q 계산에는 반올림 전 f32 id를 쓴다.
- **A256** (`quantize_row_q8_K_ref`와 동일): |x|가 가장 큰 원소(동률이면 첫 번째)의 부호 있는 값을 max로 둔다. amax = 0이면 d = 0, q = 0. 아니면 iscale = −127 / max, q = min(127, nearest\_int(iscale · x)), d = 1 / iscale. nearest\_int는 짝수 쪽 반올림(round-half-even)으로, ggml의 magic-number 구현과 같은 결과를 낸다. `BS[j]` = q\[16j .. 16j+15\]의 합.
  - v0.2.1 정정: v0.2 본문은 iscale = −128 / max로 적었으나, 고정 커밋(e3797b18)의 `quantize_row_q8_K_ref`는 `-127.f/max`를 쓴다 (`-128.f/max`는 IQ2\_XXS 때문에 주석 처리됨). x86 CPU 백엔드의 `quantize_row_q8_K`도 이 함수를 그대로 부른다. "ggml과 동일"이 우선이므로 −127로 고친다.

A32와 A256 모두 q는 정상 입력에서 \[−127, 127\] 범위다. 다만 아래 "구현 명확화"의 QACT 특수값 규칙(포화 변환)에서는 −128이 나올 수 있고, Q8\_0 가중치 q는 −128일 수 있으므로, 클러스터의 곱셈기는 부호 있는 8비트 전체 범위를 받아야 한다.

## 수치 규칙

정수 연산은 슈퍼블록 끝까지 정확하게 하고, fp32 연산은 블록(또는 sb)당 몇 번만 한다. ISA 시뮬레이터와 RTL은 비트 단위로 같아야 하므로 fp 연산의 순서와 반올림을 여기서 고정한다. ggml CPU와는 누적 순서가 달라(x86 Q8\_0 경로는 활성값 양자화 구현도 다르다. "양자화 활성값(QA) 레이아웃" 절 참고) 비트 일치는 하지 않으며, `test-backend-ops`의 NMSE 허용치로 검증한다.

**블록 항(term)** — 행 하나, 토큰 하나 기준. d, dmin은 가중치 메타(fp16 → f32), Dx는 활성값 QA의 블록 스케일, BS는 bsums.

- Q8\_0 (블록 b, 32개): isum = Σ q\_w · q\_x
- Q4\_K (sb s, 서브블록 j = 0..7, 각 32개): sc\_j, m\_j는 `get_scale_min_k4` 해석. S\_j = Σ q\_w · q\_x (q\_w ∈ \[0, 15\]), isum = Σ\_j sc\_j · S\_j, msum = Σ\_j m\_j · (BS\[2j\] + BS\[2j+1\])

```latex
\begin{aligned}
t_{Q8\_0} &= (d \cdot D_x) \cdot \mathrm{f32}(\mathit{isum}) \\
t_{Q4\_K} &= (d \cdot D_x) \cdot \mathrm{f32}(\mathit{isum}) - (d_{min} \cdot D_x) \cdot \mathrm{f32}(\mathit{msum})
\end{aligned}
```

괄호 순서대로 계산하고, 곱과 덧셈은 각각 따로 반올림한다 (FMA 융합 금지).

**정수 폭** (모두 int32 누산기로 충분, 정수 연산은 정확하므로 순서 자유)

| 값 | 최대 크기 | 필요 폭 |
| --- | --- | --- |
| Q8\_0 isum | 32 × 127 × 128 ≈ 0.52M | 부호 21비트 |
| Q4\_K S\_j | 32 × 15 × 128 = 61,440 | 부호 17비트 |
| Q4\_K isum | 8 × 63 × 61,440 ≈ 31M | 부호 26비트 |
| Q4\_K msum | 8 × 63 × 4,096 ≈ 2.1M | 부호 23비트 |
| Q4\_K msum (임의 BS) | 8 × 63 × 65,534 ≈ 33M | 부호 26비트 |

MVM은 SPM의 어떤 바이트든 QA로 받는다 (SPM에는 타입 태그가 없어 `LD`로 올린 QA도 합법). 그래서 QS는 int8 전 범위, Dx는 임의의 f32(subnormal은 DAZ, NaN·Inf는 fp 규칙대로), BS는 임의의 int16일 수 있고, 하드웨어는 이 경우도 시뮬레이터와 비트 단위로 같아야 한다. QACT가 만든 BS(|BS| ≤ 2,048)에서는 msum이 23비트지만 임의 BS에서는 BS\[2j\] + BS\[2j+1\]이 17비트, msum이 부호 26비트다 (M3, `docs/rtl-m3.md` SP1).

**행 누적**: 항 번호 i = 행 안에서의 블록(Q8\_0) 또는 sb(K-quant) 번호. 부분합 16개 p\_0..p\_15를 +0.0으로 시작해 항 i를 p\_(i mod 16)에 순서대로 더한다. 최종값 y는 16개 부분합의 완전 이진 트리 합이다: ((p0+p1)+(p2+p3)) + ((p4+p5)+(p6+p7)) 꼴로, 왼쪽부터 짝지어 4단계. MVM의 ACC 플래그가 켜져 있으면 y\_out = y\_old + y.

부분합 16개는 아키텍처 상수다. Q8\_0 최악의 경우에도 같은 부분합에 들어오는 두 항 사이가 8사이클 이상 벌어지므로, 지연 8사이클 이하의 파이프라인 fp 덧셈기로 II=1을 유지할 수 있다.

**fp 공통 규칙**

- f32는 IEEE binary32, round-to-nearest-even.
- fp16 → f32 변환은 fp16 subnormal을 포함해 정확하게 한다.
- f32 연산의 입력과 결과에서 subnormal은 0으로 처리한다 (DAZ + FTZ). 부호는 유지한다.
- int → f32 변환은 round-to-nearest-even (Q4\_K isum은 최대 26비트라 24비트를 넘을 수 있음).
- NaN 결과는 canonical NaN `0x7FC00000`으로 낸다. Inf는 IEEE대로 전파한다.
- 벡터 유닛의 초월함수(exp, rsqrt 등)는 공용 C 레퍼런스 구현 하나로 정의하고, RTL은 그 구현과 비트 단위로 일치해야 한다.

## MVM 클러스터 마이크로아키텍처

클러스터는 자기 채널의 가중치 구간을 사이클당 1 beat씩 소비하는 직선 파이프라인이다. 300 MHz × 64B = 19.2 GB/s로 DDR4-2400 채널 피크와 맞으므로, 목표는 연산이 아니라 beat 공급이 끊기지 않게 하는 것이다.

**파이프라인 단계** (전 단계 valid/ready 흐름 제어, DRAM 응답이 멈추면 전체가 멈춤)

1. **스트림 요청기**: 한 채널의 행 구간은 DRAM에서 연속이므로 rows\_c × ROW\_BEATS beat를 64 beat 버스트로 연달아 요청한다 (rows\_c = 채널 c가 가진 행 수 = max(0, min((c+1)·R, N) − c·R), 0일 수 있으며 그때는 요청하지 않는다. R × ROW\_BEATS는 채널 0 기준 상한·할당 크기일 뿐 읽는 양이 아니다). 미처리 요청은 최대 `RD_OUTSTANDING`.
2. **청크 파서**: beat마다 {META, QUANT, 행 번호, 블록 번호, 행 끝} 태그를 붙인다. 부분 청크와 행 경계를 카운터로 처리한다.
3. **메타 레지스터**: 현재 청크의 메타 beat를 잡아 둔다. Q4\_K 6비트 스케일 언패킹은 여기서 비트 슬라이싱으로 한다.
4. **가중치 언팩**: 두 포맷을 부호 8비트 레인 값으로 통일한다. Q4\_K는 q (0..15), Q8\_0은 q 그대로다. Q4\_K는 beat당 128레인, Q8\_0은 64레인을 쓴다.
5. **활성값 읽기**: 클러스터 로컬 활성값 버퍼에서 현재 beat의 K 위치에 해당하는 int8을 토큰당 128개 읽는다. Dx와 BS도 같은 버퍼에서 읽는다.
6. **MAC 레인 × 128**: s8 × s8 곱을 32개 단위로 합해 beat당 서브블록(블록) 합을 Q4\_K는 4개, Q8\_0은 2개 낸다.
7. **정수 스케일 단계**: sc\_j × S\_j를 누적해 sb(또는 블록)별 isum을 만든다. Q4\_K msum은 메타와 BS만으로 계산하므로 가중치 경로와 병렬로 처리한다.
8. **fp 항 단계**: 수치 규칙의 항 t를 계산한다.
9. **누산기**: (행, 토큰)마다 부분합 16개, 행 경계에서 두 세트를 번갈아 쓰고 이전 행의 트리 합을 뒤에서 마무리한다.
10. **출력**: 행 결과 y를 수집 네트워크로 SPM에 보낸다. 채널 c의 행 r은 SPM `dst + t × stride + (c·R + r)`에 쓴다.

**자원 추정** (U200, `LANES` = 128)

| 항목 | T = 1 | T = 8 |
| --- | --- | --- |
| 클러스터당 MAC/사이클 | 128 | 1024 |
| 클러스터당 DSP | 128 (패킹 없음) | 512 (토큰 2개 패킹) |
| 4 클러스터 DSP 합 (VU9P 6840개 중) | 512 | 2048 |
| 클러스터당 활성값 버퍼 (토큰당 `QA_LINES(A256, K_MAX)` = 584 line) | 1024비트 × 292 (37,376 B) | 8192비트 × 292 |

활성값 버퍼의 1024비트 폭은 beat 하나에 대응하는 QS 128개(토큰 하나)를 한 번에 읽는 폭이다. 깊이 292는 K\_MAX의 A256 QA 기준 QS 256 + D 4 + BS 32 워드다 (A32는 QS 256 + D 32 = 288). RTL 함의(v0.2.1 정정): 파이프라인 5단계의 Dx·BS 읽기를 위해 이 용량이 필요하며, Dx·BS를 QS와 다른 메모리로 나눠도 되지만(그때 Dx·BS 쪽은 토큰당 36워드 × 1024비트 이상) 합은 이 값보다 작으면 안 된다. 브로드캐스트는 QA 벡터 전체(`QA_LINES` line)를 복사한다.

DSP 패킹은 두 곱이 한 피연산자를 공유해야 한다. beat 하나는 한 행의 서로 다른 K 위치라 가중치도 x도 공유되지 않으므로, T = 1에서는 패킹하지 않는다 (4비트 곱 일부를 LUT 곱셈기로 돌리는 것은 선택). T ≥ 2에서는 같은 가중치에 토큰 두 개를 곱하는 조합으로 패킹한다. 이 선택은 `tsr_mac_lane` 기술 래퍼 안에 가두고 코어 구조에는 드러내지 않는다.

**활성값 브로드캐스트**: MVM 시작 시 SPM의 QA 벡터를 모든 클러스터의 로컬 버퍼에 동시에 복사한다. K = 27648 기준 약 500 line으로, `ffn_down` MVM 한 번(클러스터당 약 31만 사이클) 대비 무시할 수준이다.

**배치**: 각 클러스터는 자기 채널의 DDR 컨트롤러가 있는 SLR에 둔다. SLR을 넘는 것은 브로드캐스트 버스, 결과 수집 버스, 제어 신호뿐이며, 깊이를 파라미터로 둔 파이프라인 단을 넣는다.

**연속 MVM 겹치기**: 다음 MVM이 앞 MVM의 dst를 읽지 않으면, 클러스터는 앞 MVM의 꼬리와 다음 MVM의 가중치 요청을 겹쳐도 된다. 결과가 순차 실행과 같으므로 허용되는 as-if 최적화다. MoE의 짧은 전문가 행렬(Qwen3-30B-A3B 기준 채널당 약 3,456 beat, 토큰당 1,152개)에서 MVM마다 붙는 1\~2 µs의 채우기 비용을 숨기는 것이 목적이며, M5에서 구현한다.

## 벡터 유닛과 스칼라 레지스터

벡터 유닛(VU)은 SPM의 f32 벡터를 사이클당 `VL`(16) 원소씩 스트리밍 처리하고, 스칼라 유닛은 리덕션 결과와 상수를 다룬다. RMSNorm, SwiGLU, 잔차 덧셈, RoPE를 FPGA로 옮겨 split을 줄이는 것이 목적이며, 마일스톤 M6부터 쓴다. 그 전에는 MVM, QACT, LD/ST만으로 동작한다.

**VU 구조**

- 소스 벡터 최대 2개, 목적지 1개, 길이 n. 소스와 목적지가 같은 주소여도 된다 (원소 단위 in-place).
- 원소 연산: ADD, SUB, MUL, MAX, 스칼라 곱·덧셈(MULS, ADDS), SILU, SWIGLU, EXP, COPY. 함수 코드는 ISA 절.
- 리덕션: SUM, SUMSQ, MAX → f32 스칼라 레지스터. 원소 i를 부분합 (i mod 128)에 순서대로 넣고, 128개 부분합을 완전 이진 트리로 합친다. 부분합 수 128은 아키텍처 상수다 (VL = 16일 때 같은 부분합 갱신 간격 8사이클). SUMSQ는 x·x를 따로 반올림한 뒤 더한다.
- TOPK (M8): 길이 n 벡터에서 가장 큰 k개(k ≤ 16)의 (인덱스, 값)을 SPM에 쓴다. 값 내림차순, 같으면 인덱스 오름차순인 전순서로 정의하므로 구현 방식과 무관하게 결과가 같다. MoE 라우터의 top-k와 greedy 샘플링(k = 1, 어휘 15만 개 이상)에 쓴다.
- 비교(MAX, TOPK)는 부호-크기 비트 전순서를 쓴다: −inf < … < −0 < +0 < … < +inf < NaN.
- softmax는 전용 연산 없이 RED.MAX → SOP.MULI(−1) → VOP.ADDS → VOP.EXP → RED.SUM → SOP.RECIP → VOP.MULS로 조합한다.
- RoPE: 위치 텐서 기반 (아래).

**스칼라 유닛**

- 레지스터 `s0`..`s15`, 모두 f32, 모두 쓰기 가능. MVM이나 VU 명령어가 스칼라 피연산자로 읽는다.
- 연산: 즉시값 적재, 덧셈, 곱셈, SQRT, RECIP. SQRT와 RECIP는 IEEE 정확 반올림으로 정의한다 (스칼라라 비용이 작고, 시뮬레이터에서 `sqrtf`, `1.0f/x`로 바로 일치시킬 수 있음). rsqrt는 SQRT 후 RECIP, 반올림 두 번.

**근사 함수**: EXP, RECIP\_APPROX, SIN, COS는 원소마다 정확 반올림하기엔 비싸므로 근사로 정의한다. 알고리즘(범위 축소, 다항식 계수, 연산 순서)은 공용 C 파일 `sw/sim/tsr_ref_math.c` 하나가 정의하고, RTL은 이와 비트 단위로 일치해야 한다. SiLU(x) = x × RECIP\_APPROX(1 + EXP(−x)), SwiGLU(a, b) = SiLU(a) × b. SIN·COS는 위치 × 각속도로 수만 라디안까지 들어오므로, π/2를 여러 조각의 f32 상수로 나눈 범위 축소를 쓴다.

**RoPE (위치 기반, M6)**: 모델 로드 시 소프트웨어가 회전 쌍 i마다 각속도 A\_i와 모델 상수 M을 계산해 AUX에 둔다. ggml `rope_yarn`의 외삽·보간 혼합과 freq factor는 모두 위치에 선형이라 A\_i 하나로 접히고, attn\_factor와 YaRN mscale은 M으로 접힌다. 토큰마다 디바이스는 SPM의 I32 위치 p를 읽어 θ = f32(p) × A\_i, c = M × COS(θ), s = M × SIN(θ)로 회전한다. 호스트가 토큰마다 테이블을 만들지 않으므로 단계 B·C와 맞는다. 모드는 NORM(인접 쌍)과 NEOX(앞뒤 절반 쌍).

**RMSNorm lowering 예** (길이 5120, eps는 즉시값)

```
RED.SUMSQ  s1, x, 5120
SOP.MULI   s1, s1, 1/5120
SOP.ADDI   s1, s1, eps
SOP.SQRT   s1, s1
SOP.RECIP  s1, s1
VOP.MULS   y, x, s1, 5120
VOP.MUL    y, y, w_norm, 5120
```

## 실행 모델과 제어 인터페이스

실행은 엄격한 순차 의미론을 따른다. 다음 명령어는 앞 명령어의 모든 효과(SPM 쓰기, DRAM 쓰기 응답까지)가 끝난 상태를 보며, 결과가 같다고 보장될 때만 하드웨어가 실행을 겹칠 수 있다 (예: 연속 MVM 겹치기). MVM이 실행 시간 대부분을 차지하므로 해저드 처리 없이도 손실이 작다.

**정수 레지스터와 인자**

- R0\~R15는 int32이고 R0은 항상 0이다. 모든 주소 필드는 `R[x] + imm`으로 계산한다 (ISA 절).
- 실행 시작 시 R1\~R7은 0, R8\~R15는 CSR `ARG0`\~`ARG7` 값으로 초기화된다. 호스트는 토큰 id나 위치 같은 값을 인자로 넘긴다.
- v1(M2\~M4)은 LI, ADDI, 레지스터 기준 주소까지 구현하고 v1 lowering은 대부분 R0을 쓴다. RLD, MULI, LOOP는 M8, WAITM과 JUMP는 M9에서 구현한다.
- SPM의 데이터가 레지스터로 들어오는 통로는 RLD 하나뿐이다. 디바이스가 계산한 값(전문가 번호, 샘플링한 토큰, KV 쓰기 위치)은 모두 RLD를 거쳐 주소가 된다.

**실행 흐름**

1. 호스트가 명령 버퍼(32B 명령어 배열)를 ch0의 CMD 영역에 쓴다.
2. 호스트가 `CMD_ADDR`, `CMD_COUNT`를 쓰고 `CTRL.START`를 켠다.
3. 커맨드 프로세서가 명령어를 순서대로 프리페치·실행한다. v1 lowering은 루프 없이 펼쳐서 내고, LOOP는 M8, JUMP는 M9부터 쓴다.
4. `END`를 만나면 `STATUS.DONE`과 `done` 출력을 켠다. 오류가 나면 그 자리에서 멈추고 `STATUS.ERR`, `ERR_PC`, `ERR_CODE`를 남긴다.

호스트는 BUSY 동안 실행 중인 명령 버퍼와 IO 영역을 건드리지 않는다. 다음 토큰의 명령 버퍼는 CMD 영역의 다른 절반에 미리 써 둘 수 있다.

**CSR 맵** (32비트, 바이트 오프셋)

| 오프셋 | 이름 | 접근 | 내용 |
| --- | --- | --- | --- |
| 0x000 | `ID` | R | 매직 값 `0x54535231` ("TSR1") |
| 0x004 | `VERSION` | R | ISA major\[15:8\], minor\[7:0\] |
| 0x008 | `CFG0` | R | NUM\_CH\[7:0\], LANES\[19:8\], T\_MAX\[23:20\] |
| 0x00C | `CFG1` | R | SPM 크기 (line 단위) |
| 0x010 | `CFG2` | R | VL\[7:0\], NSREG\[15:8\], K\_MAX/256\[31:16\] |
| 0x014 | `CAPS` | R | 구현된 기능 비트: bit0 VU, bit1 예약, bit2 RLD·MULI·LOOP, bit3 TOPK, bit4 WAITM·JUMP |
| 0x020 | `CTRL` | W | bit0 START, bit1 SOFT\_RESET, bit2 ABORT |
| 0x024 | `STATUS` | R/W1C | bit0 BUSY, bit1 DONE, bit2 ERR, bit3 WAITING (WAITM 대기 중) |
| 0x028 | `CMD_ADDR_LO` | RW | 명령 버퍼 시작 (ch0, beat 주소) 하위 32비트 |
| 0x02C | `CMD_ADDR_HI` | RW | 상위 비트 |
| 0x030 | `CMD_COUNT` | RW | 명령어 개수 (JUMP가 있는 프로그램에서는 첫 구간 길이) |
| 0x040 | `ERR_PC` | R | 오류 난 명령어 인덱스 |
| 0x044 | `ERR_CODE` | R | 오류 코드 |
| 0x048 | `ERR_INFO` | R | 오류 부가 정보 (주소 등) |
| 0x050 | `PC` | R | 현재 명령어 주소 (디버그) |
| 0x060\~0x07C | `ARG0`\~`ARG7` | RW | 실행 시작 시 R8\~R15로 복사 |
| 0x100\~ | `PERF_*` | R | 64비트 카운터: 전체 사이클, 은퇴 명령어 수, 채널별 읽은 beat 수, 채널별 DRAM 대기 사이클, MVM·VU 사이클, WAITM 대기 사이클 |

`CAPS`를 두는 이유는 같은 소프트웨어가 기능이 다른 비트스트림을 다루기 위해서다. 백엔드는 `CAPS`를 보고 lowering 경로(호스트 보조 MoE 대 디바이스 MoE 등)를 고른다.

**오류 코드**

| 코드 | 이름 | 조건 |
| --- | --- | --- |
| 1 | `ILLEGAL_OP` | 정의되지 않았거나 이 비트스트림에 없는 opcode·함수 코드, 0이 아닌 예약 필드 |
| 2 | `BAD_FMT` | 지원하지 않는 가중치 포맷 |
| 3 | `QA_MISMATCH` | fmt와 QA 종류 불일치 |
| 4 | `SPM_RANGE` | SPM 범위 초과 |
| 5 | `DRAM_RANGE` | 채널 범위 초과 |
| 6 | `K_RANGE` | K > K\_MAX 또는 블록 크기의 배수가 아님 |
| 7 | `T_RANGE` | T = 0 또는 T > T\_MAX |
| 8 | `ABORTED` | `CTRL.ABORT`로 중단 |
| 9 | `LOOP_NEST` | LOOP 중첩이 `LOOP_DEPTH` 초과, 또는 본문 경계가 바깥 본문과 같은 명령어에서 끝남 |
| 10 | `WAIT_TIMEOUT` | WAITM이 시간 제한 안에 조건을 만족하지 못함 |

**XRT 어댑터**: ap\_ctrl\_hs 제어 레지스터(0x00)와 커널 인자(`cmd_addr` 64비트, `cmd_count`)를 코어 CSR 쓰기로 변환하고, `ap_start`를 `CTRL.START`에, `done`을 `ap_done`에 연결한다. 코어 CSR 나머지는 인자 뒤 오프셋에 그대로 노출해 성능 카운터와 오류 정보를 호스트가 읽게 한다.

## ISA

명령어는 고정 256비트(32B)이고 32비트 워드 w0\~w7로 나뉜다 (little-endian). 모든 주소는 처음부터 "레지스터 + 즉시값"으로 정의해, 호스트 독립과 디바이스 MoE를 인코딩 변경 없이 수용한다. v1 lowering은 기준 레지스터로 R0을 써서 절대 주소처럼 쓴다.

**w0 형식**

| 비트 | 필드 | 뜻 |
| --- | --- | --- |
| \[7:0\] | opcode | 명령어 |
| \[11:8\] | sub | 포맷 코드, 모드, 채널 등 명령어별 부코드 |
| \[15:12\] | flags | 명령어별 플래그 |
| \[19:16\] | ra | 첫째 주소 필드의 기준 레지스터 |
| \[23:20\] | rb | 둘째 주소 필드의 기준 레지스터 |
| \[27:24\] | rc | 셋째 주소 필드의 기준 레지스터 |
| \[31:28\] | rd | 결과 레지스터 (정수 레지스터를 쓰는 명령어) |

**공통 규칙**

- 아래 표의 A1, A2, A3는 첫째·둘째·셋째 주소 필드이고, 값은 각각 R\[ra\] + w, R\[rb\] + w, R\[rc\] + w다 (32비트 wrap-around). 그 밖의 필드는 즉시값이다.
- SPM 주소는 line 단위(RLD·SOP의 SPM 워드 접근만 4B 워드 단위), DRAM 주소는 beat 단위, 길이는 원소 단위.
- 예약 필드·비트와 쓰지 않는 레지스터 필드는 0이어야 하며, 아니면 `ILLEGAL_OP`.
- f32 스칼라 레지스터 번호는 즉시값 워드 안의 4비트 필드다.

**명령어 목록** (단계 = 처음 구현하는 마일스톤)

| opcode | 이름 | sub | 피연산자 | 동작 | 단계 |
| --- | --- | --- | --- | --- | --- |
| 0x00 | `NOP` | – | – | 없음 | M2 |
| 0x01 | `END` | – | – | 완료 통지, 정지 | M2 |
| 0x02 | `LI` | – | w1 imm | R\[rd\] = w1 | M2 |
| 0x03 | `ADDI` | – | w1 imm | R\[rd\] = R\[ra\] + w1 | M2 |
| 0x04 | `MULI` | – | w1 imm | R\[rd\] = R\[ra\] × w1 (하위 32비트) | M8 |
| 0x05 | `RLD` | 0 I32, 1 F32 정수변환 | A1 = SPM 워드(w1); flags bit0 MUL: w2 | R\[rd\] = 그 워드의 정수값, MUL이면 × w2. F32 변환은 0 방향 절삭 | M8 |
| 0x06 | `LOOP` | – | w1 본문 길이, w2 | 다음 w1개 명령어를 R\[ra\] + w2번 반복 (0이면 건너뜀), 중첩 `LOOP_DEPTH` | M8 |
| 0x07 | `WAITM` | 0 EQ, 1 NE, 2 GE | A1 = ch0 DRAM 워드 주소(w1, 4B 단위), w2 마스크, w3 비교값(R\[rb\] + w3), w4 폴링 간격, w5 시간 제한 | 조건을 만족할 때까지 다시 읽고, 그 워드를 R\[rd\]에 쓴다 | M9 |
| 0x08 | `JUMP` | – | A1 = ch0 명령 주소(w1, beat 단위), w2 명령어 수 | 이후 A1에서 w2개를 가져와 실행 (프로그램 체인, 상주 루프) | M9 |
| 0x10 | `LD` | ch | A1 = SPM dst(w1), A2 = DRAM src(w2), w3 n\_lines | DRAM → SPM | M2 |
| 0x11 | `ST` | ch | A1 = SPM src(w1), A2 = DRAM dst(w2), w3 n\_lines | SPM → DRAM | M2 |
| 0x12 | `DQROW` | fmt | A1 = SPM dst(w1), A2 = 행 번호(w2), A3 = W\_OFF(w3), w4 N, w5 K | 대칭 할당 텐서의 행 하나를 f32로 역양자화 (`GET_ROWS`, 토큰 임베딩) | M7 |
| 0x18 | `QACT` | 0 A32, 1 A256 | A1 = QA dst(w1), A2 = f32 src(w2), w3 K, w4 T, w5 src\_stride | f32 벡터 T개 → QA 벡터 T개 | M2 |
| 0x20 | `MVM` | fmt | A1 = dst(w1), A2 = x(w2), A3 = W\_OFF(w3), w4 N, w5 K\[15:0\]·T\[19:16\], w6 dst\_stride | 행렬 × 활성값. flags bit0 ACC | M3 |
| 0x30 | `VOP` | – | A1 = dst(w1), A2 = a(w2), A3 = b(w3), w4 n, w5 func\[7:0\]·s\[11:8\] | 원소 연산 | M6 |
| 0x31 | `RED` | 0 SUM, 1 SUMSQ, 2 MAX | A1 = src(w1), w2 n, w3 sd\[3:0\] | 리덕션 → f32 스칼라 | M6 |
| 0x32 | `ROPE` | 0 NORM, 1 NEOX | A1 = dst(w1), A2 = src(w2), A3 = 위치 I32(w3), w4 각속도 테이블(SPM line), w5 head\_dim, w6 n\_dims, w7 n\_heads\[15:0\]·s\[19:16\] | 토큰 1개 회전, s = 상수 M | M6 |
| 0x33 | `TOPK` | – | A1 = dst(w1), A2 = src(w2), w3 n, w4 k | (인덱스 I32, 값 f32) 쌍 k개를 dst에 연속 기록 | M8 |
| 0x40 | `SOP` | 아래 표 | w1 sd\[3:0\]·sa\[7:4\]·sb\[11:8\], w2 imm(f32 비트), A1 = SPM 워드(w3) | f32 스칼라 연산 | M6 |
| 0x7F | `FENCE` | – | – | 실행 겹치기 금지 경계 (겹치기가 없는 구현에서는 NOP) | M2 |

**MVM fmt 코드**: 0 = Q8\_0, 1 = Q4\_K, 2\~15 예약. 후보는 4비트 계열(Q4\_0, MXFP4)뿐이며 5·6비트 경로는 두지 않는다.

**VOP 함수 코드**: 0 ADD, 1 SUB, 2 MUL, 3 MAX, 4 MULS(b 대신 s), 5 ADDS(b 대신 s), 6 SILU(a만), 7 SWIGLU(SiLU(a) × b), 8 EXP(a만), 9 COPY(a만).

**SOP sub 코드**: 0 MOVI(sd = imm), 1 ADD(sd = sa + sb), 2 MUL, 3 ADDI(sd = sa + imm), 4 MULI(sd = sa × imm), 5 SQRT, 6 RECIP, 7 LDS(sd = SPM 워드 A1), 8 STS(SPM 워드 A1 = sa).

**MVM 의미**

```
kind = (fmt == Q8_0) ? A32 : A256            # Q4_K → A256
for t in 0 .. T-1:
    x = QA(kind, K) at  x + t × QA_LINES(kind, K)
    for n in 0 .. N-1:                       # 행 n은 채널 ⌊n/R⌋, R = ⌈N/C⌉
        y = DOT(fmt, W_OFF, n, K, x)          # 수치 규칙의 항 + 행 누적
        p = f32 at SPM  dst + t × dst_stride, 원소 n
        p = (flags.ACC) ? p + y : y
```

- `ACC`는 잔차 덧셈과 MoE 전문가 출력 합산을 MVM에 융합할 때 쓴다.
- `dst_stride` = 0이면 ⌈4N/64⌉ line으로 해석한다. dst와 x 영역이 겹치면 결과는 정의되지 않는다.
- 전문가 MVM은 A3 = R\[전문가 번호 × EXP\_STRIDE\] + 텐서 W\_OFF로 표현한다. 별도 opcode가 없다.

**기타 의미 요점**

- `QACT`의 dst는 토큰마다 `QA_LINES(kind, K)` 간격, src는 `src_stride` line 간격으로 읽는다.
- `ROPE`의 각속도 테이블은 f32 A\_i를 n\_dims/2개 담는다. 각 헤드는 head\_dim 원소씩 연속이고, n\_dims < head\_dim이면 나머지는 그대로 복사한다.
- `DQROW`는 행 번호로 채널과 채널 내 위치를 계산해 행 하나를 읽고, 수치 규칙의 비트 해석대로 f32로 푼다 (d × q, Q4\_K는 d × sc × q − dmin × m). 곱과 뺄셈은 각각 반올림한다.
- `LOOP` 본문이 바깥 본문과 같은 명령어에서 끝나면 `LOOP_NEST` 오류다.
- `WAITM`은 매번 DRAM을 새로 읽는다. 호스트는 데이터를 먼저 쓰고 플래그 워드를 마지막에 쓰며, WAITM 이후의 명령어는 그 데이터를 본다.

## ggml 백엔드 통합

`ggml-tesseract`는 카드 하나를 디바이스 하나로 등록하고, 받은 split의 노드를 ISA 명령 버퍼로 낮춰(lowering) 한 번에 실행한다. 특정 llama.cpp 커밋에 고정해 개발하고, 백엔드 인터페이스 이름과 시그니처는 그 커밋의 `ggml-backend-impl.h` 기준으로 확인한다.

**구성 요소**

| 요소 | 역할 |
| --- | --- |
| reg / device | 디바이스 "Tesseract0" 하나 등록. 레이어 오프로드(`-ngl`) 대상이 되도록 GPU 계열 디바이스 타입으로 등록 (현재 enum 확인 필요) |
| buffer type | 가상 주소 공간 하나. 실제 위치는 버퍼마다 첫 데이터 접근(set·get·memset·clear) 때 버퍼 usage로 정한다 (v0.2.1 정정: 이전 판은 `init_tensor`에서 텐서 타입으로 정한다고 적었으나, 고정 커밋의 llama.cpp는 usage가 아직 ANY일 때 `init_tensor`를 부르고 WEIGHTS는 그 뒤에 설정한다. 그래서 `init_tensor`는 텐서 메타데이터만 기록하고, 타입만으로는 F32 가중치와 F32 활성값을 구별할 수 없다. `docs/llama-pin.md`). **WEIGHTS 버퍼**: 텐서마다 배치. MUL\_MAT 가중치(2D)는 하드웨어 포맷(Q8\_0·Q4\_K)으로 WEIGHT 힙에 대칭 할당, 그 밖(1D 텐서 등)은 원본 그대로 AUX. **ANY·COMPUTE 버퍼**: 원본 바이트 그대로 둔다. IO 영역에 빈 창이 있으면 버퍼 전체가 창 하나(활성값, `test-backend-ops` 텐서)이고, 없으면 호스트 메모리다. 잎(leaf) 텐서만 든 ANY 버퍼(LoRA 어댑터 등 데이터, 프로그램이 직접 읽고 쓰지 않음)는 처음부터 호스트 메모리에 둔다. 이런 버퍼의 텐서가 MUL\_MAT src0이면 graph\_compute에서 지연 변환(Q8\_0·Q4\_K)해 WEIGHT 힙에 캐시한다 |
| set\_tensor | WEIGHTS 버퍼의 가중치: 호스트 스테이징에 모은 뒤 (필요하면 Q8\_0으로 변환해) 리패킹 + 행 분할 + 채널별 쓰기. 그 밖: 원본 바이트를 IO 창(또는 호스트 메모리)에 그대로 쓰기 |
| get\_tensor | IO 창·호스트 메모리·AUX: 원본 바이트 그대로 읽기 (`test-backend-ops`는 src0를 이렇게 읽어 CPU에 넘긴다). WEIGHTS 버퍼의 가중치: 역리패킹 (디버그용; 변환된 타입은 디바이스 사본을 다시 인코딩하므로 원본 바이트가 아니다) |
| supports\_op | 아래 조건 |
| graph\_compute | split의 노드를 명령 버퍼로 lowering, 실행, 완료 대기 |
| 전송 계층 | `tsr_transport` 인터페이스: 영역 할당, 채널별 읽기·쓰기, 실행, 대기. 구현은 `xrt`, `sim`(ISA 시뮬레이터 내장), 이후 `zynq` |

**supports\_op (v1)**: 아래를 모두 만족하는 `MUL_MAT`만 true. 그 외 연산은 스케줄러가 CPU로 보낸다.

- src0가 Tesseract 버퍼(COMPUTE 버퍼 제외)에 있는 2D 가중치. GGUF 타입과 무관하게 디바이스에는 Q8\_0 또는 Q4\_K로 저장되어 있다. 텐서 메타데이터의 타입은 원래 GGUF 타입 그대로이므로, 백엔드는 텐서별 실제 디바이스 포맷 표를 따로 가진다. 아직 할당되지 않은 src0는 그래프 입력이 아닌 잎(leaf) 텐서일 때만 받는다 (`test-backend-ops`가 할당 전에 묻는 경우). llama.cpp 스케줄링에서는 가중치가 모두 할당되어 있으므로, 할당되지 않은 src0는 그래프 입력이나 다른 연산의 결과(활성값)이고 거부한다 (v0.2.1 명확화. 예: 양자화 KV 캐시(`-ctk q8_0`)에서 Q에 거는 Hadamard 회전 행렬 `attn_inp_k_rot`).
- src1 타입 F32, src0·src1 연속(contiguous), src0의 3·4번째 차원 = 1
- K ≤ `K_MAX`
- 토큰 수 T에는 조건이 없다 (v0.2.1 명확화). llama.cpp는 가중치를 어느 버퍼에 둘지 T = 512로 고정한 probe(`weight_buft_supported`)로 정하므로, T에 따라 커지는 활성값 크기로 거부하면 어휘가 큰 모델(N + K > 131072, Qwen2.5·Llama-3)의 output.weight가 조용히 CPU에 남는다. 상주 가중치를 실행 때 큰 T로 거부해도 스케줄러가 그 가중치를 매번 디바이스에서 읽어 CPU로 복사한다 (변환된 타입은 손실, 일부 I-quant는 중단). 활성값이 IO에 다 들어가지 않는 경우는 lowering이 처리한다 ("v1 lowering 제약").

attention 안의 `MUL_MAT`(src0이 F16/F32 KV 캐시, 또는 양자화 KV의 회전 행렬)은 이 조건에서 자동으로 걸러져 CPU에 남는다. `token_embd`는 `GET_ROWS`로 쓰이므로 역시 CPU에 남는다.

v1 이후의 지원 op 확장은 아래 "MoE 지원"과 "호스트 독립 로드맵" 절의 단계를 따른다. 백엔드는 CSR `CAPS`를 읽어 비트스트림이 가진 기능에 맞는 경로를 고른다.

**lowering (v1)**: split 안의 `MUL_MAT` 노드마다 다음 명령어를 내고, split 전체를 명령 버퍼 하나로 묶어 한 번만 실행한다. src1과 dst는 ggml-alloc이 Tesseract compute 버퍼에 배치하므로, 호스트와의 복사는 스케줄러가 처리한다. 그 버퍼가 IO 창이면 프로그램이 그 자리를 읽고 쓰고, 호스트 메모리에 둔 버퍼면 graph\_compute가 IO 스테이징 영역을 거쳐 옮긴다 ("v1 lowering 제약").

```
LD    spm_x,  io(src1), lines
QACT  spm_xq, spm_x, K, T
MVM   spm_y,  spm_xq, W_OFF(src0), N, K, T
ST    io(dst), spm_y, lines
END
```

토큰 수가 `T_MAX`보다 크면 T\_MAX 단위 타일로 나눈다. v1 하드웨어는 T\_MAX = 1이라 프리필이 느리며, 이는 알려진 제약이다. 그래프 모양과 텐서 주소가 같으면 생성한 명령 버퍼를 캐시해 재사용한다. 캐시는 바이트(64 MiB)와 항목 수(4096)로 제한하고 가장 오래 쓰지 않은 것부터 내보내며, 4 MiB가 넘는 프로그램(T\_MAX = 1에서 토큰 수에 비례해 커지는 긴 프리필)은 캐시하지 않는다 (v0.2.1 명확화). 토큰마다 바뀌는 입력 텐서(토큰 id, 위치, KQ 마스크, KV 쓰기 인덱스)는 IO 영역에 두고 프로그램이 매번 읽으므로, 캐시한 명령 버퍼를 패치할 필요가 없다.

**주의할 점**

- **분할 업로드**: llama.cpp는 큰 텐서를 여러 번의 `set_tensor`(offset, size)로 나눠 올릴 수 있다. 리패킹은 텐서 전체(또는 행 단위로 완결된 구간)가 모인 뒤에 한다.
- **split 개수**: v1은 `MUL_MAT`만 디바이스로 가서 토큰당 split이 수백 개 생긴다. 스케줄러의 split·입력 상한 상수에 걸리지 않는지 고정한 커밋에서 확인한다. `GGML_SCHED_DEBUG`로 split 구성을 볼 수 있다.
- **output.weight**: Qwen2.5-32B는 N = 152064인 Q6\_K 행렬이라 로드 시 Q8\_0(약 0.83GB)으로 변환되고, MVM 한 번이 크다. llama.cpp는 `-ngl` ≥ 1이면 출력 레이어를 가장 먼저 디바이스에 배정하고, 위 T 무관 `supports_op` 덕분에 512토큰 probe를 통과해 실제로 Tesseract에 놓인다. 출력이 묶인(tied) 모델은 `token_embd`의 사본(`TENSOR_DUPLICATED`)이 출력 레이어로 놓인다. 디코드·프리필은 출력할 행(n\_outputs)만 계산하지만, compute 버퍼 예약은 n\_ubatch개 출력의 logits(-ub 512에서 152064 × 512 × 4 B ≈ 297 MiB)를 잡으므로 그 청크는 호스트 메모리에 놓이고 스테이징을 거친다.
- **검증 연결**: `test-backend-ops`가 Tesseract 백엔드를 CPU와 비교하도록 등록하고, `TSR_DEVICE=sim|xrt`, `TSR_TRACE=<디렉터리>` 환경변수로 장치 선택과 트레이스 덤프를 제어한다.
- **v1 lowering 제약** (M1 구현, 상세는 `sw/ggml-tesseract/README.md`, `docs/llama-pin.md`):
  - 프로그램이 직접 읽고 쓰는 활성값(src1, dst)은 ch0 IO 영역(256 MiB)에 있다. v1 소프트웨어는 IO 영역을 compute 버퍼 창(아래쪽)과 스테이징 영역(맨 위, max(16 MiB, 2 × `SPM_BYTES`), U200에서 16 MiB)으로 나눈다. 백엔드의 `get_max_size`는 창 용량(240 MiB)이라 ggml-alloc은 compute 버퍼를 그 이하의 청크로 자르고, 한 텐서가 그보다 크면 그 텐서만 든 청크를 만든다. 청크는 첫 데이터 접근 때 빈 창에 들어가면 IO 창, 아니면 호스트 메모리다 (v0.2.1 명확화: 이전 판은 IO를 넘는 compute 버퍼를 `-ub`로 줄여 맞추고 src1 + dst가 IO보다 크면 `supports_op`가 거부한다고 적었다. 그 규칙은 위 `supports_op` 절의 이유로 없앤다).
  - 호스트 메모리 청크에 src1이나 dst가 있는 `MUL_MAT`은 앞에 쌓인 명령을 먼저 실행한 뒤 혼자 실행된다. 스테이징 영역에 들어가는 토큰 수 단위(T 타일)로 src1 행을 호스트 → 스테이징으로 쓰고(src1이 IO 창이면 그 자리에서 LD), LD/QACT/MVM/ST 후 dst 행을 스테이징 → 호스트로 읽는다. 그래서 IO에 한꺼번에 있어야 하는 활성값은 타일 하나뿐이고 `-ub`에는 IO 상한이 없다 (호스트 메모리만 쓴다). 토큰 하나의 x + QA + y가 SPM에 들어가는 모양이면 스테이징 영역(SPM의 2배 이상)에도 토큰 하나가 들어간다. 전송량은 IO 창일 때와 같다 (src1 쓰기와 dst 읽기를 스케줄러 대신 graph\_compute가 한다).
  - llama.cpp는 KV 캐시를 디바이스 버퍼에 두려 하므로 v1은 `-nkvo`로 실행한다 (KV 연산은 v1에 없다).
  - src1의 f32 행이 line 경계에서 시작하도록 K는 32의 배수여야 한다 (Q8\_0 변환 조건과 같음). 토큰 하나의 x + QA + y가 SPM에 들어가지 않는 모양도 거부한다.
  - N·4가 64의 배수가 아니고 T > 1이면 dst의 토큰 행 시작이 beat 경계가 아니다. 이때도 위 스테이징 경로를 탄다: ST는 line 단위로 패딩된 스테이징 영역에 쓰고, 호스트가 타일마다 dst로 압축 복사한다. 스테이징 경로와 디바이스가 만든 src0(지연 Q8\_0 변환)를 읽는 경우에는 split 하나가 명령 버퍼 여러 개로 나뉘어 실행된다. LLM 가중치 행렬은 N이 16의 배수라 이 경우(행 정렬 때문의 스테이징)는 없다.
  - T = 1이고 N·4가 64의 배수가 아니면 ST가 마지막 line의 꼬리(0)를 dst 다음 바이트에 쓴다. 백엔드의 `get_alloc_size`가 텐서 크기를 64바이트 배수로 올려 이 꼬리가 텐서 자신의 패딩에 떨어진다.

## MoE 지원

MoE는 U200에 가장 잘 맞는 워크로드다. 전체 파라미터는 64GB에 들어가고 토큰당 읽는 양은 작아서, 같은 카드에서 Dense 32B(토큰당 약 20GB)보다 Qwen3-30B-A3B(토큰당 약 2GB, 파일 약 18.6GB)가 10배 가까이 빠를 수 있다.

**ggml이 내리는 모양**: 라우터 `MUL_MAT`(GGUF에서는 F32, 로드 시 Q8\_0으로 변환) → softmax → top-k → `MUL_MAT_ID` × 3(gate, up, down) → 라우터 확률 곱 → 합산. `MUL_MAT_ID`는 3D 전문가 텐서와 토큰별 전문가 번호 I32 텐서(ids)를 받는다. src1은 gate·up에서는 토큰당 하나(브로드캐스트), down에서는 전문가별로 다르다.

| 단계 | 방식 | 필요한 것 | 마일스톤 |
| --- | --- | --- | --- |
| 호스트 보조 | 라우터·top-k는 CPU, ids가 넘어오면 백엔드가 `graph_compute`에서 읽어 전문가별 MVM을 낸다 | 라우터 Q8\_0 변환, 연속 MVM 겹치기, 전문가 텐서 대칭 할당 | M5 |
| 디바이스 단독 | 라우터, softmax, TOPK, 전문가 MVM을 모두 디바이스가 수행 | RLD(MUL), LOOP, TOPK, 레지스터 기준 W\_OFF | M8 |
| 프리필 | 토큰을 전문가별로 묶어 T > 1 MVM | 호스트(M7) 또는 디바이스 정렬 | M7 이후 |

**디바이스 단독 lowering 예** (토큰 1개, 전문가 8개, 전문가 행렬 2048 ↔ 768, 라우터 확률은 h에 미리 곱해 down MVM의 ACC로 합산)

```
TOPK        sel, probs, 128, 8              # sel = (idx, val) × 8
LI          R1, 0
LOOP        8회, 본문 9개
  RLD.I32   R2, [R1 + sel_w], MUL EXP_STRIDE  # R2 = 전문가 번호 × EXP_STRIDE
  SOP.LDS   s1, [R1 + sel_w + 1]              # 라우터 확률
  MVM.Q4_K  g,  xq, [R2 + W_gate], N=768,  K=2048
  MVM.Q4_K  u,  xq, [R2 + W_up],   N=768,  K=2048
  VOP.SWIGLU h, g, u, 768
  VOP.MULS  h, h, s1, 768
  QACT.A256 hq, h, K=768, T=1
  MVM.Q4_K  y,  hq, [R2 + W_down], N=2048, K=768, ACC
  ADDI      R1, R1, 2
```

y는 루프 전에 잔차(또는 0)로 초기화한다. 확률을 down 전에 곱하므로 ggml(down 후 곱)과 누적 순서가 달라지며, 모델 단위 KL 기준으로 검증한다.

**제약**: gpt-oss-120b, GLM-4.5-Air급은 64GB에 들어가지 않는다. gpt-oss-20b는 들어가지만 MXFP4 포맷이 필요하다 (예약 fmt 코드).

## 호스트 독립 로드맵

llama.cpp는 토큰마다 그래프를 만들지만, KV 길이 버킷이 같으면 그래프 모양이 같고 토큰마다 바뀌는 것은 입력 텐서(토큰 id, 위치, KQ 마스크, KV 쓰기 인덱스)뿐이다. 이 성질을 이용해 세 단계로 호스트 개입을 줄인다. KV 패딩 단위와 llama.cpp의 그래프 재사용 경로는 고정 커밋에서 확인한다.

| 단계 | 호스트가 토큰마다 하는 일 | 디바이스에 필요한 것 | llama.cpp와의 관계 | 마일스톤 |
| --- | --- | --- | --- | --- |
| A. 전체 그래프 디바이스화 | 입력 텐서 쓰기, 실행 1회, logits 읽기, 샘플링 | attention·KV 캐시 디바이스화, 나머지 op 전부, 그래프 모양별 프로그램 캐시 | 정상 API 그대로 (`llama_decode`) | M7 |
| B. 위치 독립 프로그램 | 인자 2개(토큰 id, 위치) 쓰기, 실행 1회, 결과 읽기 | RLD, MULI, LOOP, DQROW(임베딩 조회), KV 버킷별 프로그램 | 백엔드 내부 최적화, 샘플링 위치는 확인 필요 | M8 |
| C. 디바이스 자체 루프 | 출력 토큰 링버퍼 폴링, 중단·새 프롬프트 신호 | TOPK 샘플링, WAITM 메일박스, JUMP 체인 | 디코드 루프만 전용 러너 `sw/runner` | M9 |

**단계 A에 필요한 ggml op**: `RMS_NORM`, `MUL`, `ADD`, `GLU`(SwiGLU), `ROPE`(위치 텐서), `GET_ROWS`(DQROW), `SET_ROWS`(KV 쓰기), `CPY`·`CONT`, `SOFT_MAX` 또는 `FLASH_ATTN_EXT`, 8비트(q8\_0) KV를 쓰는 attention `MUL_MAT`, 메타데이터 op(`VIEW`, `RESHAPE`, `PERMUTE`, `TRANSPOSE`). 이 단계가 끝나면 토큰당 split이 1개가 된다.

**단계 B**: KV 길이 버킷마다 프로그램 하나를 컴파일해 두고, 토큰 id와 위치를 인자 레지스터(R8\~R15)로 받는다. 임베딩 행, RoPE 위치, KV 쓰기 주소는 레지스터에서 계산하고, attention은 위치에서 계산한 횟수만큼 LOOP로 KV 블록을 돈다. 단일 시퀀스 디코드에서는 인과 마스크가 "위치까지 전부 유효"라 마스크 텐서가 필요 없다.

**단계 C**: llama.cpp는 모델 로딩, 토크나이저, 채팅 템플릿, 그래프 생성에만 쓰고, 디코드 루프는 디바이스에 상주한다. 상주 프로그램의 골격은 다음과 같다.

```
loop:
  WAITM   R9, [mailbox], NE 0          # 호스트의 시작·중단 신호
  (한 토큰 디코드: 임베딩 → 레이어 × L → 출력 헤드)
  TOPK    tok, logits, n_vocab, 1      # greedy
  RLD.I32 R8, [tok]                    # 다음 토큰 id
  ST      [ring + R10], tok, 1         # 출력 링버퍼에 기록
  ADDI    R10, R10, 1 ; ADDI R11, R11, 1  # 링 위치, 시퀀스 위치
  JUMP    loop                         # 버킷 경계에서는 다음 버킷 프로그램으로
```

[openTPU](https://github.com/FeSens/openTPU)는 같은 구조를 모델별로 손으로 짠 프로그램으로 구현했다 (위치를 레지스터로 받는 상주 디코드 프로그램, 버킷 간 프로그램 체인, 호스트 플래그 대기). Tesseract는 이 프로그램을 llama.cpp 그래프에서 자동으로 만든다는 점이 다르다.

## 검증 전략

검증은 오라클 사슬이다: ggml CPU가 ISA 시뮬레이터를 검증하고, ISA 시뮬레이터가 RTL과 하드웨어를 비트 단위로 검증한다. 시뮬레이터를 RTL보다 먼저 만들고, 백엔드는 처음부터 `TSR_DEVICE=sim`으로 시뮬레이터를 가짜 디바이스로 쓴다.

| 단계 | 기준(오라클) | 합격 조건 | 도구 |
| --- | --- | --- | --- |
| 연산 단위 (sim) | ggml CPU 백엔드 | NMSE 허용치 이내 | `test-backend-ops` |
| 모델 전체 (sim) | ggml CPU 백엔드, 그리고 입력을 1 ulp 흔든 CPU 자신(바닥값) | 평균 KL ≤ 바닥값 KL × 3, top-1 일치 ≥ 80% (초기값, M1에서 조정) | `tools/klcheck` (자체), `llama-perplexity` |
| 모델 전체 수치 (sim) | 레퍼런스 모델(`libtsr_ref`), 그와 코드를 공유하지 않는 numpy 스펙 모델, fp64 정확값 | 모든 가중치 MUL_MAT의 모든 출력 원소가 두 모델과 비트 일치, fp32 오차 상한 위반 0 | `tools/numcheck` (자체) |
| RTL 블록 | 공용 C 레퍼런스 함수 | 비트 일치 | cocotb + Verilator |
| RTL 코어 전체 | ISA 시뮬레이터 | 트레이스 재생 비트 일치 | cocotb + Verilator |
| 하드웨어 | ISA 시뮬레이터 | 트레이스 재생 비트 일치, 같은 토큰 열 | XRT 호스트 러너 |

**모델 단위 KL 기준**: 디바이스는 누적 순서가 CPU와 달라 비트 일치가 불가능하고, 한 번 다르게 반올림된 int8 값은 뒤 레이어로 퍼진다. 그래서 "CPU를 1 ulp 흔들었을 때 CPU 자신이 얼마나 벗어나는가"를 바닥값으로 두고, 디바이스가 그 바닥값 근처에 있는지 본다. 같은 프롬프트에서 디바이스가 낸 토큰을 CPU에 다시 먹여 매 위치를 같은 문맥으로 비교한다. 방법과 초기 합격선은 [openTPU의 검증 방식](https://github.com/FeSens/openTPU)을 따랐다.

**모델 단위 수치 검증 (numcheck, 2026-10-22)**: KL은 CPU와의 거리만 보므로, 짧은 프롬프트 하나로 디바이스가 스펙 산술을 정확히 수행하는지를 따로 확인한다. 방법·오차 상한 유도·결과는 `docs/numcheck.md`.
- 시뮬레이터의 MVM과 `libtsr_ref`는 같은 `tsr_dot_row`를 호출하므로 둘의 비트 일치는 공통 결함을 잡지 못한다. 그래서 이 문서만 보고 따로 만든 numpy 모델(B1b)을 함께 둔다. 순차 합산으로 바꾼 변이 모델은 실제 텐서 원소의 67~87%에서 틀려 누산 순서에 민감함을 확인했다.
- fp32 오차 상한(B2)은 어떤 합산 순서든 성립하므로 순서 검사는 못 한다. 16 부분합과 고정 트리 순서는 계속 비트 일치로 검증한다.
- 관찰된 사실 두 가지:
  - CPU가 항상 더 정확한 쪽은 아니다. Q6\_K에서는 디바이스 쪽 오차가 더 작다("로드 시 변환" 참고). 따라서 CPU 대비 KL·NMSE에는 CPU 자신의 오차가 섞여 있다.
  - 양자화 모델에서는 1 ulp 바닥값이 일부 위치에서 사라진다. 잔차 경로는 흔들림을 유지하지만, 모든 MUL_MAT의 8비트 활성값 양자화가 반올림 경계 근처가 아닌 원소의 흔들림을 흡수한다. 그래서 앞쪽 위치의 바닥값 KL이 정확히 0이 되고(0.5B Q8\_0 32토큰 중 0~6번, 32B 16토큰 중 0번) 평균 바닥값이 낮게 잡힌다. 합격선 조정(미결정 사항)에서 함께 다룬다.

**연산 단위 NMSE의 예외 (결정, 2026-10-22)**: 출력이 1개인 F32 가중치 MUL_MAT(`test-backend-ops`의 `type_a=f32,type_b=f32,m=1,n=1,k=2048`)은 가중치·활성값을 모두 8비트로 양자화하는 이 문서의 경로에서 NMSE 5e-4를 무작위 입력의 약 15%에서 넘는다. 같은 8비트 값의 정확한 내적도 같은 입력에서 넘으므로 구현 결함이 아니다. 허용치와 하드웨어 경로는 바꾸지 않고, ctest는 이 모양 하나만 `test-backend-ops` 비교에서 빼고 `ggml_tesseract_scalar_nmse`(레퍼런스 모델과 비트 일치)로 대신 검증한다. 수치 근거는 `docs/llama-pin.md`.

**트레이스 형식**: SPM은 시작 시 정의되지 않으므로 모든 입력은 DRAM에서 온다. 트레이스 하나는 디렉터리 하나다.

- `meta.json`: 구성 파라미터, 사용한 DRAM 영역 목록, 명령어 수
- `cmd.bin`: 명령어 배열
- `init/ch<c>_<offset>.bin`: 실행 전 DRAM 영역 내용
- `expect/ch<c>_<offset>.bin`: 실행 후 기대 DRAM 내용 (`ST`로 쓰인 영역)

백엔드가 `TSR_TRACE`로 실제 llama.cpp 실행의 split마다 트레이스를 덤프하므로, 실제 모델의 실제 텐서 모양이 그대로 RTL 회귀 테스트가 된다.

회귀 트레이스 세트 (모두 시뮬레이터 재생 통과, 재생성 스크립트는 각 디렉터리의 `regen.sh`):

| 세트 | 내용 | 규모 |
| --- | --- | --- |
| `traces/m2` | M2 명령어만(`tsr-trace-gen --ops m2`), 6개 구성, 오류 종료 포함 | 172개 |
| `traces/gen` | M1 생성기(LD·QACT·MVM·ST) | 32개 |
| `traces/m3` | `--ops m3`: MVM 경계(부분 청크, K\_MAX, N = 0, N < C, 채널 둘이 나눠 쓰는 y line, ACC 특수값, 임의 QA 이미지, 스케일 특수값)와 MVM 오류 23종, 7개 구성(U200 v1, T\_MAX 8, 1채널, 3채널, 16채널, 오류 두 세트) | 284개, 150 MiB |
| `traces/model` | Qwen2.5-0.5B Q8\_0과 Q4\_K\_M(재양자화) 각각 프롬프트 "Hi", 1토큰 실행의 split 전부 | 290개, 989 MiB |
| `traces/tmp/tbo`, `tbo_tmax8` | `test-backend-ops` 덤프 (재생성 스크립트 없음) | 1,402개씩 |

**블록 단위 테스트 범위**

- 언팩과 dequant: 포맷별로 ggml 역양자화 결과와 비교
- 청크 파서: K를 블록 크기 배수 중 무작위로, 부분 청크와 N < C, 홀수 N 포함
- QACT: amax = 0, 동률 최대값, 반올림 경계값, 정상수 amax와 섞인 subnormal 원소와 subnormal뿐인 블록(DAZ, `test_ref_ggml.c`의 directed 벡터)
- fp 항·누산기: subnormal(FTZ, "구현 명확화"의 \[2^−126 − 2^−150, 2^−126) 경계 벡터 포함), Inf, NaN, 부호 있는 0
- 메모리 포트: 응답 지연과 backpressure를 무작위로 주는 cocotb 메모리 모델, 미처리 요청 상한
- SLR 파이프라인 깊이 파라미터를 바꿔도 결과가 같음

**성능 검증**: 시뮬레이터도 채널별 beat 수를 세므로 이론 사이클을 낸다. 하드웨어 `PERF_*` 카운터와 비교해 DRAM 대기 사이클 비율로 실효 대역폭을 확인한다.

## 저장소 구조와 마일스톤

저장소는 계층 분리를 그대로 디렉터리로 옮긴다. llama.cpp는 특정 커밋에 고정한 포크를 서브모듈로 두고, `ggml/src/ggml-tesseract`에 백엔드를 연결해 CMake 옵션 `GGML_TESSERACT`로 켠다.

```
tesseract/
├── docs/spec.md             # 이 문서 (Markdown 내보내기)
├── sw/
│   ├── ggml-tesseract/            # 백엔드: reg/device/buffer, lowering, 리패커, 프로그램 캐시
│   ├── sim/                 # ISA 시뮬레이터, tsr_ref_math.c, 트레이스 입출력
│   ├── transport/           # tsr_transport 구현: sim, xrt
│   ├── runner/              # 단계 C 전용 디코드 러너 (M9)
│   └── tools/               # 트레이스 러너, 테스트 벡터 생성기, klcheck (ggml 링크)
├── rtl/
│   ├── core/                # 벤더 독립 SV: cmdproc(레지스터·LOOP), cluster, vu, spm, dma
│   ├── tech/generic/        # tsr_sram_*, tsr_mac_lane 일반 구현
│   ├── tech/xilinx/         # URAM/BRAM, DSP48E2 패킹
│   └── platform/alveo_xrt/  # 제어 브리지, AXI4 메모리 어댑터, kernel.xml
├── tb/                      # cocotb 테스트, 무작위 지연 메모리 모델
├── traces/                  # 회귀용 트레이스 (작은 모델)
├── third_party/llama.cpp/   # 고정 커밋 포크 (서브모듈)
└── build/u200/              # v++ 설정, connectivity, pblock 제약
```

마일스톤은 M0\~M9의 10단계이고, 각 단계는 아래 그림의 통과 조건을 만족해야 다음으로 넘어간다. M1까지는 하드웨어 없이 소프트웨어만으로 32B 모델이 끝까지 돌고, M4가 v1 목표, M5가 MoE, M7\~M9가 호스트 독립 단계 A\~C다.

**마일스톤: M4에서 v1 목표, M9에서 호스트 독립 디코드에 도달한다** (그림 `docs/fig/milestones.jpg`)

| 단계 | 작업 | 통과 조건 |
| --- | --- | --- |
| M0 · 셸 브링업 | DMA 테스트 커널로 채널별·4채널 동시 대역폭 측정 | 실측 대역폭 기록 · 성능 예측의 기준값 |
| M1 · ISA 시뮬레이터 + 백엔드 | `TSR_DEVICE=sim`, MUL\_MAT만 오프로드, 하드웨어 없음 | test-backend-ops · 32B sim KL 기준 통과 |
| M2 · RTL 코어 골격 | 커맨드 프로세서(R0\~R15, LI·ADDI), DMA, LD/ST, QACT | Verilator 트레이스 재생 비트 일치 |
| M3 · MVM 클러스터, Q8\_0 | 1채널로 시작해 4채널로 확장 | Qwen2.5-0.5B Q8\_0 실기 동작 |
| M4 · Q4\_K 언팩, 4채널 | v1 목표 · 그 밖의 타입은 로드 시 Q8\_0 변환 | 32B Q4\_K\_M 실기 동작 · 1 tok/s 이상 |
| M5 · MoE, 호스트 보조 | 라우터 Q8\_0 변환, 연속 MVM 겹치기, 전문가 대칭 할당 | Qwen3-30B-A3B 실기 동작 |
| M6 · 벡터 유닛, op 확장 | RMSNorm, 위치 기반 RoPE, SwiGLU, 잔차, softmax | 레이어 안 호스트 왕복 제거 |
| M7 · attention·KV 디바이스화, T > 1 | 단계 A: DQROW, 프리필 타일링, 추측 디코딩 | 토큰당 split 1개 · 프리필 향상 실측 |
| M8 · 레지스터 확장, 디바이스 MoE | 단계 B: RLD·MULI·LOOP, TOPK, 버킷별 프로그램 | 호스트는 토큰당 인자 2개만 쓰기 |
| M9 · 디바이스 자체 디코드 루프 | 단계 C: WAITM, JUMP, 토큰 링버퍼, `sw/runner` | 호스트는 링버퍼 폴링만 |

진행 기록: M1 통과(2026-10-22, 32B klcheck 비율 1.36, top-1 98.83%), M2 통과(2026-10-23, `traces/m2` 172개 Verilator 재생 비트 일치). M0는 카드 장착과 XRT·플랫폼 설치 뒤에 한다. M3의 RTL은 Q4\_K 경로까지 함께 설계·검증한다(클러스터 데이터패스를 공유하므로). M3 통과 조건은 실기 동작이라 플랫폼 어댑터와 XRT 전송 계층이 M3에 들어간다.

M1에서 소프트웨어 경로가 먼저 완성되고, M2 이후 하드웨어 단계는 모두 같은 트레이스로 검증한다. ISA 기능은 각 단계에 필요한 만큼만 구현하되, 인코딩은 처음부터 이 문서의 ISA 절을 따른다.

## 미결정 사항

- [x] 프로젝트 이름: Tesseract로 확정
- [ ] U200 셸과 Vitis/XRT 버전 확정, `platforminfo`로 채널별 SLR 위치 확인 (M0)
- [ ] exp·역수·sin·cos 근사 알고리즘과 계수 확정 (`tsr_ref_math.c`, M6 전)
- [ ] 비순차 실행(유닛 간 겹치기) 도입 여부: M4 성능 카운터의 명령어 간 공백 비율로 판단
- [ ] T\_MAX 합성값과 토큰 패킹 방식 (M7에서 앞당김, 2026-10-23). 합성값 8을 목표로 한다. 시뮬레이터에서 T > 1 경로를 확인했다: numcheck PASS(0.5B 두 개, 32B), T\_MAX 1과 출력 바이트 동일, 사이클 모델 프리필 0.5B 5.9배·32B 7.2배. RTL 자원·타이밍과 DSP 패킹은 M3 C단계에서 정한다. 남은 것: 출력 헤드(N ≈ 15만)는 토큰당 x + QA + y가 약 9,600 line이라 4 MiB SPM에 토큰 6개만 들어가고, T = 7~8이면 헤드 가중치를 두 번 읽는다(32B에서 패스당 +0.83 GB, +4%). 백엔드가 헤드를 N 방향으로 나누면 12~13개가 들어간다. 일반 프리필은 마지막 토큰의 logits만 계산하므로 영향이 없고, 모든 위치의 logits가 필요한 추측 디코딩(M7)에서만 문제가 된다
- [ ] (M7로 연기, 2026-10-23: 기본 경로 우선) 추측 디코딩의 출력 동일성: 디바이스 MUL\_MAT은 토큰별로 T와 무관하게 비트 동일하지만, llama.cpp CPU attention이 T = 1(vec\_dot)과 T ≥ 2(tinyBLAS)에서 다른 커널을 써서 greedy 출력이 근소한 동률에서 갈릴 수 있다(0.5B, 37번째 토큰에서 관찰). 디바이스가 있으면 auto FA가 꺼진다. `-fa on`이면 n\_kv < 512, T ≤ 63에서 동일하다. FA on은 M1 검증 설정(FA off)과 다르므로 채택하면 klcheck·numcheck를 다시 돌린다
- [ ] KV 캐시 레이아웃: n\_kv가 늘어나면 행 범위 분할이 맞지 않으므로 행 interleave(행 r → 채널 r mod C) 같은 별도 배치와 DRAM 위치, 8비트 KV 우선: llama.cpp를 -ctk q8\_0 -ctv q8\_0로 실행하는 것을 전제로 할지 (M7)
- [ ] 보류 포맷: 4비트 계열 Q4\_0, MXFP4(gpt-oss-20b)만 후보. 5·6비트와 I-quant의 하드웨어 경로는 두지 않는다
- [ ] 라이선스: RTL은 Solderpad Hardware 또는 CERN-OHL-S, 소프트웨어는 MIT
- [ ] llama.cpp 업스트림 기여 시점
- [ ] 모델 단위 KL 합격선 조정 (M1). numcheck에서 양자화 모델의 1 ulp 바닥값이 앞쪽 위치에서 0이 됨을 확인했다. 제안: 바닥값이 움직인 위치만으로 비율 계산
- [ ] `PERF_CH_RD_BEATS[0]`에 명령어 fetch beat가 섞여 채널 0의 LD beat를 하드웨어와 대조할 수 없다. fetch를 빼거나 `PERF_FETCH_BEATS`(인덱스 5 + 2 × `NUM_CH`)를 추가할지 (M4 성능 검증 전)
- [ ] 단계 B의 샘플링 위치: llama.cpp 샘플링을 디바이스로 옮기는 경로가 고정 커밋에 있는지 확인, 없으면 `sw/runner`가 담당 (M8)
- [ ] greedy 외 샘플링(temperature, top-p)과 디바이스 난수 생성기 (M9)
- [ ] `sw/runner`와 llama-server 연동 방식 (M9)

## 구현 명확화 (v0.2.1)

M1 착수 시 시뮬레이터와 RTL이 비트 단위로 같아야 하는데 본문에서 열려 있던 부분을 아래처럼 고정한다. C 미러는 `sw/sim/include/tsr_isa.h`, `tsr_ref.h`다.

- **FTZ 판정 시점**: f32 연산은 입력의 subnormal을 부호 있는 0으로 바꾼 뒤(DAZ) binary32 지수 범위 그대로(점진적 언더플로 포함) IEEE RNE로 계산하고, 그렇게 반올림된 결과가 subnormal이면 부호 있는 0으로 바꾼다. 동치로, 정확한 결과의 크기가 2^−126 − 2^−150 미만이면 부호 있는 0, 그 이상이면 보통의 RNE 결과다. 이 규칙은 IEEE 754 §7.5의 "반올림 후 tininess"(지수 범위 제한 없이 24비트로 반올림한 값이 2^−126 미만인지로 판정, x86 MXCSR.FTZ가 이 방식)와도, 반올림 전 tininess와도 다르다 (v0.2.1 정정: 이전 판은 이 규칙을 "반올림 후 tininess 판정"이라 잘못 불렀다). 정확한 결과가 \[2^−126 − 2^−150, 2^−126) 구간이면 이 규칙은 ±2^−126을 내지만, 반올림 전 판정은 이 구간 전체에서, 지수 무제한 반올림 후 판정은 \[2^−126 − 2^−150, 2^−126 − 2^−151)에서 ±0을 낸다. 덧셈·뺄셈은 이 구간의 결과가 나오지 않고(subnormal 범위의 합은 정확하다) 곱셈·나눗셈에서만 생긴다. 경계 벡터(0x3F7FFFFF × 0x00800000 = 0x00800000, 0x3F1FA0BA × 0x00CD4708 = 0x00800000, 0x00FFFFFF / 2 = 0x00800000, 0x3F7FFFFE × 0x00800000 = 0)는 `sw/sim/tests/test_ref_fp.c`에 있으며 RTL fp 블록 테스트에도 넣는다.
- **QACT 입력**: amax와 max 탐색도 DAZ를 거친 값으로 한다. A32의 d = amax / 127, id = 1 / d, x · id는 위 f32 규칙을 따르는 연산이며, `D`는 d를 fp16 RNE로 바꾼 뒤 f32로 되돌린 값이다. A256의 iscale = −127 / max, iscale · x, d = 1 / iscale도 같은 f32 규칙을 따른다.
- **QACT 특수값** (v0.2.1, `tsr_ref.c`가 정의): NaN 입력은 amax·max 탐색에서 제외한다(비교에 참여하지 않음). f32 → int8 변환은 NaN → 0, 반올림한 정수를 \[−128, 127\]로 포화한다 (A32 half-away, A256 half-even; `tsr_q8_round_away`/`tsr_q8_round_even`). 결과적으로 A32에서 amax = Inf이면 d = Inf, id = 0, `D` = Inf, q = 0이고, A256에서 max = ±Inf이면 iscale = ∓0, d = ∓Inf, q = 0이며, max = ±m인데 m이 너무 작아 −127 / max가 넘치면 iscale = ∓Inf로 q가 포화되고(x = 0인 원소는 NaN → 0) d = ∓0이다. 모든 원소가 0(또는 NaN)인 A256 블록의 `BS`는 0이다 (ggml은 이때 bsums를 쓰지 않는다).
- **ggml과의 일치 범위**: QACT는 입력에 subnormal·NaN·Inf가 없고 블록 amax ≥ 2^−119(d가 정상수로 남는 범위)이거나 amax = 0이면 `quantize_row_q8_0_ref`/`quantize_row_q8_K_ref`와 비트 단위로 같다 (`sw/sim/tests/test_ref_ggml.c`로 확인). 그 밖에서는 위 DAZ/FTZ 규칙 때문에 다를 수 있다 (예: A32에서 d = amax / 127이 subnormal이면 FTZ로 d = 0, q = 0). DQROW(`tsr_dequant_row`)는 d·dmin이 유한하면 `dequantize_row_q8_0`/`dequantize_row_q4_K`와 비트 단위로 같고, 비유한 스케일에서는 NaN 인코딩만 다르다 (ggml은 x86 기본 NaN 0xFFC00000이나 payload NaN, Tesseract는 canonical NaN).
- **fp16 변환의 NaN**: fp16 → f32는 NaN의 부호와 payload를 유지하고 quiet 비트를 세운다. f32 → fp16은 NaN을 부호 | 0x7E00으로 바꾼다. 두 변환 모두 ggml의 `GGML_FP16_TO_FP32`·`GGML_FP32_TO_FP16`과 모든 입력에서 비트 단위로 같다. 변환은 f32 연산이 아니므로 canonical NaN 규칙을 적용하지 않는다 (NaN은 뒤따르는 f32 연산에서 canonical이 된다).
- **꼬리 바이트**: QACT·MVM 등 SPM에 쓰는 명령어는 피연산자 마지막 line의 유효 원소 뒤 바이트를 0으로 쓴다 (본문의 "정의되지 않음"을 0으로 고정. 트레이스가 결정적이 되도록). MVM ACC에서도 꼬리는 0이다.
- **stride 0**: QACT의 `src_stride` = 0은 ⌈4K/64⌉ line으로 해석한다 (`dst_stride`와 같은 규칙).
- **명령 버퍼 끝**: `END` 없이 `CMD_COUNT`개를 다 실행하면 `ILLEGAL_OP`, `ERR_PC` = `CMD_COUNT`.
- **`QA_MISMATCH`**: SPM에는 타입 태그가 없으므로 v1 시뮬레이터와 하드웨어는 이 오류를 내지 않는다 (코드만 예약). 종류 일치는 lowering의 책임이다.
- **오류 검사 순서** (한 명령어에서 여러 조건이 동시에 걸리면 앞의 것): `ILLEGAL_OP`(opcode, sub, flags, 예약 필드) → `BAD_FMT` → `K_RANGE` → `T_RANGE` → `SPM_RANGE` → `DRAM_RANGE`. `ERR_INFO`는 범위 오류일 때 넘친 주소(SPM line 또는 DRAM beat; 정확한 정의는 아래 "범위 검사"), 그 밖에는 0이다. 오류가 난 명령어는 아무 상태도 바꾸지 않는다.
- **SPM 초기값**: 시뮬레이터는 시작 시 SPM을 0xA5로 채운다. 프로그램은 쓰지 않은 SPM을 읽으면 안 된다.
- **트레이스 `meta.json`**: 필드는 `sw/sim/include/tsr_trace.h` 주석을 따른다.

아래는 ISA 시뮬레이터(`sw/sim/tsr_sim.c`, M1 3·4단계)를 만들며 고정한 항목이다. RTL과 하드웨어에서 보이는 동작이므로 트레이스 재생으로 함께 검증한다.

- **명령어별 사용 필드** (나머지 비트·워드·레지스터 필드는 모두 예약, 0이 아니면 `ILLEGAL_OP`):

  | 명령어 | w0에서 쓰는 필드 | 쓰는 워드 |
  | --- | --- | --- |
  | `NOP`, `END`, `FENCE` | opcode | 없음 |
  | `LI` | opcode, rd | w1 |
  | `ADDI` | opcode, ra, rd | w1 |
  | `LD`, `ST` | opcode, sub(채널), ra, rb | w1\~w3 |
  | `QACT` | opcode, sub(QA 종류), ra, rb | w1\~w5 |
  | `MVM` | opcode, sub(fmt), flags bit0(ACC), ra, rb, rc | w1\~w4, w5\[19:0\], w6 |

- **sub 값**: `LD`·`ST`의 채널이 `NUM_CH` 이상이면 `ILLEGAL_OP`. `QACT`의 sub(QA 종류)가 2 이상이면 `ILLEGAL_OP` (가중치 포맷이 아니므로 `BAD_FMT`가 아니다). `MVM`의 fmt가 2 이상이면 `BAD_FMT`. `CAPS` 비트가 없는 기능의 opcode(M1 시뮬레이터는 `CAPS` = 0이라 VU·SOP·RLD·MULI·LOOP·TOPK·WAITM·JUMP 전부)와 `DQROW`(M7 전), 정의되지 않은 opcode는 `ILLEGAL_OP`.
- **QACT의 K·T**: K는 0 < K ≤ `K_MAX`이고 QA 블록 크기(A32 32, A256 256)의 배수여야 하며 아니면 `K_RANGE`. T는 1 ≤ T ≤ `T_MAX`, 아니면 `T_RANGE` (MVM과 같은 규칙).
- **범위 검사**: 피연산자의 시작 주소 a(`R[x] + imm`, 32비트 wrap 후)와 범위 e(line 또는 beat, 64비트로 계산)에 대해 a + e ≤ 한계(wrap 없이)이면 통과한다. e는 `LD`·`ST`에서 n\_lines, T개 토큰을 stride로 놓는 피연산자에서 (T − 1) × stride + 토큰당 길이(stride 0은 위 규칙으로 바꾼 값), `MVM` 가중치에서 R × `ROW_BEATS`(행이 가장 많은 채널 0 기준, N = 0이면 0)다. e = 0이어도 같은 식을 쓴다 (a ≤ 한계). SPM 한계 = `SPM_BYTES`/64 line, DRAM 한계 = min(`CH_BYTES`/64, 2^`ADDR_BEAT_W`) beat. 같은 종류의 범위 오류가 여러 피연산자에 걸리면 주소 필드 순서(A1, A2, A3)대로 앞의 것을 보고하고, `ERR_INFO`("넘친 주소")는 그 피연산자의 시작 주소 a다.
- **길이 0**: `MVM`의 N = 0과 `LD`·`ST`의 n\_lines = 0은 합법이며 아무것도 쓰지 않는다 (범위 검사는 위 식대로 한다). N = 0인 `MVM`은 가중치도 SPM의 x도 읽지 않는다 (관찰할 수 있는 차이는 없다).
- **명령어 fetch**: 명령어 pc는 ch0 beat `CMD_ADDR` + ⌊pc/2⌋의 바이트 32 × (pc mod 2)부터다. 그 beat가 DRAM 한계 밖이면 `DRAM_RANGE`, `ERR_PC` = pc, `ERR_INFO` = beat 주소의 하위 32비트. 프리페치는 `CMD_COUNT` 안에서만 읽고, 범위 오류는 그 명령어를 실행할 차례에만 보고한다. 실행 중에 자기 명령 버퍼를 `ST`로 고치는 프로그램은 결과가 정의되지 않는다.
- **오류 시 STATUS**: 오류로 멈추면 `STATUS` = DONE | ERR이고 BUSY는 내려가며 `done` 출력도 켜진다 (어댑터가 `ap_done`으로 오류 종료도 알 수 있도록). 호스트는 DONE 후 ERR 비트로 성공 여부를 판단한다.
- **ABORT**: 명령어 사이에서만 확인한다. `ABORTED`의 `ERR_PC`는 실행하지 않은 첫 명령어, `ERR_INFO` = 0. BUSY가 아닐 때 쓴 ABORT는 다음 START가 지우므로 효과가 없다.
- **CSR 부가 규칙**: BUSY 동안의 START는 무시한다. START는 `ERR_PC`·`ERR_CODE`·`ERR_INFO`·`PC`를 0으로, R1\~R7을 0, R8\~R15를 `ARG0`\~`ARG7`로 만든다 (시뮬레이터는 s0\~s15도 +0.0으로 만든다; s 레지스터 초기값은 M6에서 확정). `STATUS`의 W1C 대상은 DONE과 ERR뿐이고 BUSY·WAITING 쓰기는 무시한다. SOFT\_RESET은 `STATUS`, `ERR_*`, `PC`, 레지스터, `PERF_*`를 0으로 하고 `CMD_ADDR`, `CMD_COUNT`, `ARG*`, SPM, DRAM은 그대로 둔다. `PC`는 실행 중(또는 마지막으로 실행한) 명령어의 인덱스다. 정의되지 않은 CSR 오프셋은 읽으면 0, 쓰기는 무시한다.
- **겹치는 피연산자**: `MVM`의 토큰별 dst 구간끼리 겹치는 경우(T > 1이고 dst\_stride < ⌈4N/64⌉)와 `QACT`의 dst와 src가 겹치는 경우는 dst와 x가 겹치는 `MVM`처럼 결과가 정의되지 않는다. lowering은 이런 프로그램을 내지 않는다. (시뮬레이터는 `MVM`에서 모든 y를 먼저 계산한 뒤 토큰 순서대로 쓰고, `QACT`는 토큰마다 읽고 쓴다.)
- **ACC 덧셈**: y\_out = y\_old + y는 fp 공통 규칙(DAZ, FTZ, canonical NaN)을 따르는 f32 덧셈 한 번이다.
- **성능 카운터**: 시뮬레이터의 `PERF_*` 사이클 값은 `tsr_sim.c` 주석에 적은 결정적 모델이며 비트 일치 대상이 아니다. 채널별 읽은 beat 수 중 `MVM` 가중치(채널 c: rows\_c × `ROW_BEATS`)와 `LD` beat 수는 하드웨어와 같아야 하고, 명령어 fetch beat 수는 프리페치 방식에 따라 다를 수 있다.
- **트레이스 재생 규칙**: init 영역 밖의 DRAM은 0이다. 재생기는 `cmd.bin`을 명령 주소에 올린 뒤 init 영역을 올리고(겹치면 init이 우선; 기록기는 `cmd.bin`을 init과 일치시키고 init이 덮지 않은 곳은 0으로 쓴다), 실행 후 `STATUS`·`ERR_CODE`·`ERR_PC`·`ERR_INFO`와 expect 영역을 비트 단위로 비교한다. 실행이 쓴 DRAM beat 집합은 expect 영역의 합집합과 같아야 하고, 실행이 읽은 DRAM은 init 영역(또는 `cmd.bin`)이나 같은 실행에서 먼저 쓴 곳이어야 한다. `perf`는 참고값이다.

아래는 M2 RTL(`docs/rtl-m2.md` §11 P3~P10)을 만들며 고정한 항목이다. 시뮬레이터로는 관찰할 수 없는 경우(실행 중 CSR 접근, 하드웨어 리셋)가 많아 `tb/core`의 directed 테스트로 검증한다.

- **실행 중 CSR 읽기**: 모든 CSR은 BUSY 동안에도 언제든 읽을 수 있고, 읽기에는 부수 효과가 없다. 예외는 아래 `PERF_*` 스냅샷뿐이다. XRT 호스트는 실행 중 `STATUS`와 `PERF_*`를 폴링한다.
- **`PERF_*` 하드웨어 의미**: `PERF_CYCLES`는 BUSY인 사이클 수, `PERF_CH_WAIT[c]`는 채널 c에 끝나지 않은 읽기가 하나 이상 있는데 읽기 beat가 오지 않은 사이클 수다. `PERF_MVM_CYCLES`는 MVM 유닛이 명령어를 들고 있는 사이클 수(수락 다음 사이클부터 done 사이클까지, M3)다. N = 0인 MVM은 1, 검사에서 오류로 끝난 MVM은 0을 더한다. 64비트 카운터의 LO를 읽으면 그 순간의 64비트 전체를 표본으로 잡고, 같은 카운터의 다음 HI 읽기는 그 표본의 HI를 한 번 돌려준다. 그 밖의 HI 읽기는 현재 값이며, SOFT\_RESET은 표본을 버린다.
- **BUSY 중 SOFT\_RESET**: 다음 명령어 경계에서 ABORT처럼 멈춘 뒤 실행이 끝나면 soft reset을 적용한다. `STATUS`는 BUSY에서 0이 되고 DONE은 켜지지 않는다. 같은 CTRL 쓰기의 START는 무시한다.
- **ABORT와 명령 버퍼 끝**: ABORT는 명령 버퍼 끝 검사보다 먼저 확인한다. 마지막 명령어 뒤에 본 ABORT는 `ILLEGAL_OP`가 아니라 `ABORTED`이고 `ERR_PC` = `CMD_COUNT`다. ABORT는 `PC`를 바꾸지 않는다 (마지막으로 실행한 명령어).
- **DONE ⇒ 메모리 정지**: `STATUS.DONE`과 `done`은 그 실행의 모든 메모리 트랜잭션이 끝난 뒤에만 켜진다. 버린 명령어 프리페치도 포함한다.
- **하드웨어 리셋**: `ap_rst_n`은 동기식 active-low다. 리셋 후 모든 CSR은 전원 투입 직후 값이다. SOFT\_RESET과 달리 `CMD_ADDR`, `CMD_COUNT`, `ARG*`도 0이고 SPM 내용은 정의되지 않는다. 메모리 쪽은 코어와 함께 리셋되어야 한다. 끝나지 않은 트랜잭션은 완료되지 않고 버려지며, 리셋 전 요청의 응답이 리셋 후에 오면 안 된다. 이 조건은 M4 플랫폼 어댑터가 보장한다. `tsr_fetch`는 남은 응답을 견디지만 `tsr_mem_arb`의 미완료 카운터는 이를 전제로 한다.

## 작업 규칙과 시작 순서 (Claude Code용)

이 문서를 계약으로 삼아 M1부터 착수한다. 문서와 코드가 어긋나면 문서를 먼저 고치고 코드를 맞춘다.

**규칙**

1. 이 문서(`docs/spec.md`)가 단일 진실 원천이다. ISA, 메모리 레이아웃, 수치 규칙을 바꾸는 변경은 문서 수정과 같은 커밋에 넣는다.
2. ISA 시뮬레이터가 오라클이다. RTL 변경은 관련 트레이스 재생이 비트 단위로 일치해야 머지한다.
3. 리패킹, QA 양자화, 내적, 근사 함수의 레퍼런스 구현은 `sw/sim` 안에 한 벌만 두고, 백엔드·시뮬레이터·cocotb 테스트가 공유한다.
4. llama.cpp는 `third_party/llama.cpp`의 고정 커밋만 쓴다. 커밋 해시와 확인한 백엔드 API는 `docs/llama-pin.md`에 기록하고, 이 문서에서 "고정 커밋에서 확인"이라고 적은 항목은 착수 전에 확인해 거기에 적는다.
5. 예약 필드에는 0을 쓰고, 시뮬레이터와 하드웨어 모두 0이 아니면 `ILLEGAL_OP`로 멈춘다.
6. RTL은 SystemVerilog로 쓰고 Verilator lint 경고가 없어야 하며, 모듈마다 cocotb 테스트를 둔다. 벤더 프리미티브는 `rtl/tech/` 밖에서 쓰지 않는다.
7. 성능 수치에는 측정 조건(모델, 프롬프트 길이, 생성 토큰 수, 빌드)을 함께 적는다.

**M1 착수 순서**

1. 저장소 골격, CMake, `third_party/llama.cpp` 서브모듈 고정, `docs/llama-pin.md` 작성.
2. 레퍼런스 라이브러리: Q8\_0·Q4\_K 리패킹과 역리패킹, 그 밖의 타입 → Q8\_0 로드 변환, `ROW_BEATS`, QA 양자화(A32, A256), 블록 항과 행 누적. ggml의 역양자화·양자화 함수와 대조하는 단위 테스트 포함.
3. ISA 시뮬레이터: 디코더, 정수·f32 레지스터, SPM, 채널별 DRAM, CSR, 오류 코드, beat 카운터. 명령어는 M1에 필요한 NOP, END, LI, ADDI, LD, ST, QACT, MVM, FENCE부터.
4. `tsr_transport` 인터페이스와 `sim` 구현, 트레이스 덤프와 재생 도구.
5. `ggml-tesseract` 백엔드 골격: reg/device/buffer type, 대칭 할당, 분할 업로드, `supports_op`(MUL\_MAT), lowering, 명령 버퍼 캐시.
6. `test-backend-ops`의 MUL\_MAT 비교 통과.
7. Qwen2.5-0.5B Q8\_0, 이어서 Qwen2.5-32B Q4\_K\_M을 sim으로 끝까지 실행하고 `tools/klcheck`로 KL 기준 확인. 이것이 M1 통과 조건이다.
