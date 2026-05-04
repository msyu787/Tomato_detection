## Tomato Observation System

YOLO 기반 토마토 관측 시스템입니다.  
실시간 영상에서 토마토를 검출하고(ripe/unripe), 객체 추적을 통해 개수를 집계하며, CLI 또는 Flask 웹 UI로 실행할 수 있습니다.

## 폴더 구조

```
tomato-observation-system/
├── configs/                    # 설정 파일용 디렉터리(선택)
├── models/                     # YOLO 가중치(.pt)
├── scripts/
│   ├── main_tomato_observer.py # CLI 진입점
│   └── trackers/
│       ├── basic_bytetracker.py
│       └── basic_sort.py
├── src/
│   └── tracking/
│       ├── pipelines/          # ByteTrack/SORT·토마토 파이프라인
│       └── utils/              # ROI, 모션 등 유틸
├── templates/
│   └── index.html              # Flask 웹 UI 템플릿
├── tomato_observer_app.py      # 웹 UI 서버
├── pyproject.toml
├── uv.lock
├── README.md
├── .python-version
├── .gitignore
├── .project_root
└── .cursorrules
```

`uv sync` 후에는 프로젝트 루트에 `.venv`가 생기며, 실행 시 `__pycache__`가 생성될 수 있습니다.

## 주요 기능

- 실시간 입력 처리: 웹캠/비디오(OpenCV), ZED(옵션)
- 토마토 상태 분류: `ripe`, `unripe`
- 객체 추적: `ByteTrack` 또는 `SORT` 선택 가능
- ROI 기반 추론 영역 제한, 모션 보정/안정화 옵션 제공
- 웹 UI에서 시작/중지 및 임계값 조절
- 세션 통계 및 CSV 저장 지원

## 환경 요구사항

- Python 3.11 이상
- 주요 패키지:
  - `torch`, `torchvision`
  - `ultralytics`
  - `opencv-python`
  - `flask`
  - `supervision`, `trackers`, `vidstab`, `pyrootutils`

## 설치

이 프로젝트는 `uv` 기준으로 의존성을 관리합니다.

```bash
uv sync
```

필요 시 실행:

```bash
uv run python --version
```

## 실행 방법

### 1) CLI 실행

```bash
python scripts/main_tomato_observer.py
```

기본 실행 설정은 `scripts/main_tomato_observer.py`의 `CONFIG`에서 조정합니다.

### 2) 웹 UI 실행

```bash
python tomato_observer_app.py
```

실행 후 브라우저에서 Flask 서버 주소(기본 localhost)로 접속하여 카메라 시작/중지 및 임계값을 제어합니다.

## 설정 가이드

주요 설정 위치:

- `scripts/main_tomato_observer.py`의 `CONFIG`
- `src/tracking/pipelines/tomato_observer_pipeline.py`의 `DEFAULT_CONFIG`

자주 조절하는 파라미터:

- `tracker_type`: `bytetrack` / `sort`
- `camera_backend`: `opencv` / `zed`
- `source`: 카메라 인덱스 또는 영상 파일 경로
- `model_path`: YOLO 모델 경로
- `device`: `cpu` 또는 CUDA 디바이스 인덱스
- `conf`: confidence threshold
- `nms_iou`: NMS IoU threshold
- `min_box_area`: 최소 박스 면적 필터

웹 UI 관련:

- `tomato_observer_app.py`에서 `camera_mode`, `conf_thres`, `iou_thres` 기본값을 관리합니다.

