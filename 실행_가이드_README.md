# 제주도 버스 승차인원 예측 — 실행 가이드

> `jeju_bus_teamE1i4.ipynb` 실행을 위한 환경 설정 및 데이터 준비 가이드

---

## 1. 폴더 구조

노트북과 같은 위치에 아래 구조로 데이터를 배치해주세요.

```
프로젝트 폴더/
├── jeju_bus_teamE1i4.ipynb    ← 노트북 파일
└── data/
    └── bus/
        ├── train.csv                     ← 학습 데이터 (필수)
        ├── test.csv                      ← 예측 대상 (필수)
        ├── bus_bts.csv                   ← BTS 교통카드 데이터 (필수)
        ├── submission_sample.csv         ← 제출 양식 (필수)
        ├── OBS_ASOS_TIM_20260309132252.csv       ← 9월 기상 데이터 (필수)
        └── OBS_ASOS_TIM_20260309155017_10월.csv   ← 10월 기상 데이터 (필수)
```

> 노트북 내부에서 `DATA_DIR = 'data/bus/'`로 경로를 참조합니다.
> 폴더명이 다르면 노트북 셀 2의 `DATA_DIR`을 수정하세요.

---

## 2. 환경 설정 (Linux 기준)

### Python 버전
- Python **3.10** 이상 권장

### 필수 라이브러리 설치

```bash
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm xgboost optuna koreanize-matplotlib
```

| 라이브러리 | 용도 |
|-----------|------|
| `pandas`, `numpy` | 데이터 처리 |
| `matplotlib`, `seaborn` | 시각화 |
| `koreanize-matplotlib` | 한글 폰트 자동 적용 |
| `scikit-learn` | RandomForest, KFold, 평가지표 |
| `lightgbm` | LightGBM 모델 (GPU 사용 시 GPU 빌드 필요) |
| `xgboost` | XGBoost 모델 (CPU로 실행) |
| `optuna` | 하이퍼파라미터 자동 튜닝 |

### GPU 관련 (선택사항)
- **LightGBM GPU**: 노트북에서 `device='gpu'`로 설정되어 있습니다.
  - GPU가 없으면 해당 부분을 `device='cpu'`로 변경하세요.
  - GPU 사용 시 CUDA + LightGBM GPU 빌드가 필요합니다.
- **XGBoost**: CPU(`tree_method='hist'`)로 설정되어 있어 별도 GPU 설정 불필요합니다.

### GPU 없이 실행하려면
노트북에서 아래 부분을 찾아 수정하세요:

```python
# 변경 전 (GPU)
'device': 'gpu',

# 변경 후 (CPU)
'device': 'cpu',
```

해당 부분은 **셀 "Step 1"**과 **셀 "Step 2"**에 각각 있습니다.
`Ctrl+F`로 `device': 'gpu'`를 검색하면 찾을 수 있습니다.

---

## 3. 실행 순서

**위에서 아래로 순서대로 실행하면 됩니다.**

| 순서 | 셀 내용 | 예상 시간 | 비고 |
|------|---------|----------|------|
| 1 | 라이브러리 & 한글 폰트 설정 | 5초 | 한글 테스트 그래프 확인 |
| 2 | 데이터 로드 | 10~30초 | bus_bts.csv가 240만행이라 시간 소요 |
| 3 | EDA 시각화 | 1~2분 | 그래프 여러 장 출력 |
| 4 | 전처리 + 피처 엔지니어링 | 2~5분 | BTS 피처 계산에 시간 소요 |
| 5 | 모델링 Step 1 (55개) | **10~30분** | Optuna 15회 + 3모델 × 3시드 × 5폴드 |
| 6 | 피처 중요도 시각화 | 5초 | Step 1 완료 후 확인 |
| 7 | 모델링 Step 2 (59개) | **10~30분** | 동일 구조 |
| 8 | 블렌딩 + 제출파일 생성 | 1분 | CSV 파일 생성 |

> 전체 실행 시간: GPU 있으면 약 **30~40분**, CPU만 있으면 약 **1~2시간**

---

## 4. 한글 폰트 문제 해결

### 그래프에 한글이 깨지는 경우

1. `koreanize-matplotlib`이 설치되어 있는지 확인
```bash
pip install koreanize-matplotlib
```

2. Jupyter 커널을 **재시작** (Kernel → Restart)한 뒤 처음부터 다시 실행

3. 그래도 안 되면 matplotlib 캐시를 수동 삭제
```bash
rm ~/.matplotlib/fontlist-*.json
```
그 후 커널 재시작 → 처음부터 실행

---

## 5. 자주 발생하는 오류

### `FileNotFoundError: train.csv`
→ `data/bus/` 폴더에 데이터 파일이 있는지 확인하세요.

### `ModuleNotFoundError: No module named 'lightgbm'`
→ `pip install lightgbm` 실행

### `ModuleNotFoundError: No module named 'koreanize_matplotlib'`
→ `pip install koreanize-matplotlib` 실행

### `LightGBMError: GPU Tree Learner was not enabled`
→ LightGBM GPU 빌드가 안 된 환경입니다. `device='gpu'`를 `device='cpu'`로 변경하세요.

### `XGBoostError: gpu_hist`
→ 이미 CPU(`tree_method='hist'`)로 설정되어 있으므로 발생하지 않습니다. 발생 시 노트북이 구버전일 수 있으니 최신 파일을 사용하세요.

---

## 6. 출력 파일

실행 완료 후 노트북과 같은 폴더에 아래 CSV 파일들이 생성됩니다.

| 파일 | 설명 |
|------|------|
| `submission_team1_55.csv` | 55개 피처 3모델 평균 |
| `submission_team1_59.csv` | 59개 피처 3모델 평균 |
| `submission_team1_blend_*.csv` | 55+59 블렌딩 (비율별) |
| `submission_team1_3way.csv` | 55+59+95 3-way 블렌딩 (최고 점수) |

---

## 7. 파이프라인 요약

```
data/bus/*.csv
    ↓
[1] 라이브러리 로드 + 한글 폰트
    ↓
[2] 데이터 로드 (train, test, bus_bts, 기상)
    ↓
[3] EDA 시각화 (요일/승차관계/정류장/이용자/날씨)
    ↓
[4] 전처리 (기상 집계, 날짜 피처, BTS merge, log1p)
    ↓
[5] 피처 엔지니어링 (55개 공통 + 4개 팀원 고유 = 59개)
    ↓
[6] 모델링 (LGB GPU + RF CPU + XGB CPU × 3시드 × 5폴드)
    ↓
[7] 블렌딩 (55개×0.5 + 59개×0.2 + 95개×0.3)
    ↓
submission_team1_3way.csv → Private 13위
```
