# dct 패키지 보강 설계 (v2.0.0)

날짜: 2026-06-27
대상: `dct/dct.py`, `dct/__init__.py`, `pyproject.toml`, `README.md`, 신규 `tests/`

## 배경

`dct` 패키지는 블록 DCT를 stride=kernel_size 합성곱으로 구현한다. 검증 결과
핵심 설계와 성능은 견실하나(라운드트립 오차 1.6e-6, conv 방식이 grouped conv
대비 MPS에서 ~12배, einsum 대비 ~4.5배 빠름) 다음 단점이 확인되었다.

1. **순방향 계수가 정규직교가 아님** — scipy 기본 `norm="backward"`를 써서 모든
   계수가 16배(에너지 256배) 균일 스케일. Gram 대각이 모두 256이며 off-diagonal은
   ~4e-5(직교는 맞으나 단위노름 아님). Parseval 미성립. 라운드트립은 IDCT가 스케일을
   정확히 되돌리므로 무해.
2. **fp16 직접 입력 불가** — 커널 buffer가 float32 하드코딩이라 `x.half()`는
   `MPSHalfType vs MPSFloatType` 에러. autocast 내부에서는 동작.
3. **죽은 코드 / 거친 부분** — `reparameterize()` 스텁, `"kernel"` 버퍼 이중 등록,
   주석 없는 `.T` 비대칭, IDCT2d 입력 검증 부재, `True if cond else False` 군더더기,
   DCT2d/IDCT2d 간 device 처리 비대칭(`device` vs `k.device`).
4. **테스트·docstring 부재.**

## 결정 사항

- normalization: `norm` 인자 추가, 기본 **`"ortho"`** (정규직교).
- 하위호환: 자유롭게 변경 가능 → 버전 **2.0.0**으로 상향(기본 동작이 바뀌는
  breaking change 허용).
- 테스트/문서: pytest 스위트 + 공개 클래스 docstring 추가.

## 변경 상세

### 1. normalization — `norm` 인자

- `DCT2d`, `IDCT2d`(및 공통 베이스)에 `norm: str = "ortho"` 추가.
- 커널 초기화 시 `scipy.fft.dct(np.eye(k), dct, norm=norm, orthogonalize=True)`,
  IDCT는 `scipy.fft.idct(..., norm=norm, orthogonalize=True)`로 전달.
- 허용값 `{"ortho", "backward", "forward"}`. 그 외엔 명확한 `ValueError`.
- **불변식:** 동일한 `norm`에서 `IDCT2d(DCT2d(x)) == x`가 정확히 성립
  (모든 scipy norm에서 idct(type=d)는 dct(type=d)의 정확한 역변환).
- `norm`을 `extra_repr`에 포함.

### 2. dtype / fp16 — forward 캐스팅

- 각 `forward`에서 커널을 입력 dtype에 맞춘다:
  `weight = self.kernel if self.kernel.dtype == x.dtype else self.kernel.to(x.dtype)`.
- 효과: fp16/bf16 입력 직접 동작, autocast 일관 동작, fp64는 `module.double()` 후
  입력 dtype 일치로 캐스팅 없이 동작. 커널이 작아(64×1×8×8) 캐스팅 비용은 무시 가능.
- 생성자 dtype 인자는 추가하지 않는다(YAGNI). 버퍼 기본 dtype은 float32 유지.

### 3. 구조 정리 / 죽은 코드 제거

- 공통 베이스 클래스 하나로 통합해 `"kernel"` 버퍼를 **한 번만 등록**.
  - 베이스가 1D 행렬(분석/합성)을 만들고, 서브클래스가 2D zigzag 커널을 구성·등록
    하던 이중 등록 구조를 제거.
  - zigzag/device 처리를 베이스로 일원화(현 `device` vs `k.device` 비대칭 해소).
- **수학적 구성은 보존** — DCT는 1D 행렬 `.T`, IDCT는 비전치라는 정/역 필터 방향
  비대칭은 라운드트립을 만드는 의도된 구성이므로 유지하되 **이유를 주석으로 명시**.
- `DCT2d.reparameterize()` 제거.
- `_is_a_batched_tensor`, `_is_resolution_divisible`를 `return x.ndim == 4`,
  `return x.size(2) % d == 0 and x.size(3) % d == 0` 형태로 단순화.

### 4. IDCT2d 입력 검증

- DCT2d와 대칭으로:
  - 4D 아니면 `RuntimeError`(메시지: batched image tensor 기대).
  - 채널 수 `n`이 `self.selections`로 나눠떨어지지 않으면 `RuntimeError`
    (현재는 조용히 잘못된 reshape).

### 5. 테스트 / docstring / 메타

- 신규 `tests/test_dct.py` (pytest):
  - 라운드트립 정확성: `norm` ∈ {ortho, backward}, kernel_size ∈ {4, 8, 16},
    여러 batch/channel/해상도.
  - scipy 대조 정확성: 동일 norm에서 `scipy.fft.dctn`과 일치(zigzag 역정렬 후 비교).
  - ortho일 때 기저 정규직교성: Gram ≈ I.
  - selections: 출력 채널 수 = c·selections, 절단량 증가 시 RMSE 단조 증가,
    IDCT 복원 shape.
  - 입력 검증: 비4D, 해상도 비가분, IDCT 채널 비가분 → 각각 에러.
  - autograd: 입력으로 gradient 전파.
  - dtype: fp16/bf16 forward 동작(지원 디바이스), fp64(`.double()`).
  - device: cpu 항상, mps/cuda는 있으면 parametrize/skip.
- 공개 클래스 `DCT2d`, `IDCT2d`에 docstring(인자: dct type, kernel_size,
  selections, norm; 입출력 shape; normalization 의미).
- `pyproject.toml`: dev 의존성에 `pytest` 추가(`[project.optional-dependencies]`),
  `version = "2.0.0"`.
- `dct/__init__.py`: `__version__ = "2.0.0"`.
- `README.md`: normalization 기본값(ortho)과 `norm` 인자 한 줄 추가.

## 검증 기준 (완료 정의)

- `pytest`가 cpu에서 전부 통과(mps 가능 시 mps도).
- ortho 라운드트립 오차 < 1e-4, ortho Gram-I 오차 < 1e-4.
- 리팩터 후에도 라운드트립 정확성 유지(회귀 없음).
- 기존 단점 1~4가 모두 해소되거나(코드) 명시(문서).

## 범위 밖 (YAGNI)

- `reparameterize` 실제 구현(고정 변환이라 융합할 학습 파라미터 없음).
- 생성자 dtype 인자, 겹치는 블록/패딩 모드, 1D/3D DCT, `dct` 타입 외 변환.
