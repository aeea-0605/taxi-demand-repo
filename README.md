# NYC 뉴욕 택시 수요 예측 프로젝트
---

## 1. 개요
<br/>

### **1-1. 프로젝트 목적**
다양한 모델링과 하이퍼 파라미터 튜닝을 통해 최적의 택시 수요 예측 모델을 생성하고, Airflow를 사용해 모델 학습 및 예측에 대한 자동화 파이프라인을 구축하는 프로젝트입니다.

<br/>

### **1-2. 프로젝트 목표**
- 비선형적인 모델에서 Feature가 Target에 끼치는 영향력을 Shaply-value를 통해 분석한다.
- 모델링한 history에 대해 log를 기록하고, 예측성능이 좋은 모델 및 파라미터를 도출한다.
- BashOperator를 사용한 airflow task를 생성하고, scheduler를 통해 주기적으로 모델을 학습 & 예측한다.

<br/>

### **1-3. 기술적 Summary**
- BigQuery를 사용한 데이터 ETL
- Simple Regresor를 사용한 Baseline 설정
- Ensemble Model, RandomizedGridSearchCV를 사용한 Hyper Parameter Tuning
- Shap을 사용한 Feature에 대한 영향력 분석
- sacred를 사용한 모델링 결과 log 기록
- GCP-Composer를 사용한 Airflow 환경구축
- yaml 파일을 사용한 scheduler로 실행되는 main 파일의 환경 설정
- Docker Container(Postgres, Ubuntu)를 사용한 Apache-Airflow 환경구축 (진행 예정)

<br/>

### **1-4. 프로젝트 구성도**

<br/>

### **1-5. 데이터셋 및 설명**
BigQuery에서 제공하는 데이터셋 사용
- `bigquery-public-data.new_york_taxi_trips.tlc_yellow_trips_2015`
    - 2015년 1월의 데이터를 추출하여 분석에 사용
- `bigquery-public-data.geo_us_boundaries.zip_codes`
    - trip 데이터셋에 zip_code 값을 추가하기 위해 사용

<br/>

---
---
## 2. 분석 과정 및 결과
- 각 과정에서의 자세한 Visualization 및 상세한 분석과정은 Jupyter Notebook을 참고해주세요.

<br/>

### **1. EDA - Time, Distance, Region**
> [01_EDA.ipynb](https://github.com/aeea-0605/taxi-demand-repo/blob/main/notebooks/01_EDA.ipynb)

#### **Time**
- datatime : 특정 Event 발생으로 1월 26~27일에 수요가 급격히 감소
- hour : 오전 5시가 Off-Peak Time이며, Peak Time은 오후 6~7시 이다.
- weekday : 월~토 까지 수요가 증가하고 일요일에 감소한다.
- 요일별 hour
    - 대체적으로 오후 6~7시에 수요가 몰린다.
    - 주말이 평일보다 새벽의 수요가 많다.
    - 평일은 아침, 주말은 점심에 상대적으로 수요가 많다.

#### **Distance**
- 단거리 Peak Time : 오후 6~10시에 수요가 많다.
- 장거리 Peak Time : 오후 1~3시에 수요가 많다.

#### **Region**
- 수요가 가장 많은 zip-code는 10002이며, 지역마다 수요의 편차가 크다.

---

### 2. **Preprocessing**
> [02_preprocessing.ipynb](https://github.com/aeea-0605/taxi-demand-repo/blob/main/notebooks/02_preprocessing.ipynb)
- BigQuery : datatime과 좌표에 대한 전처리를 진행한 후 Data Extract
- Python
    - 범주형 변수에 대한 전처리 (One-Hot Encoding)
    - Train, Test 데이터셋 세분화 및 Features와 Target 분리

#### **Baseline 진행방향**
- Target의 분포를 확인해 log scaling 여부 판단

---

### **3. Baseline Modeling**
> [03_baseline_model.ipynb](https://github.com/aeea-0605/taxi-demand-repo/blob/main/notebooks/03_baseline_model.ipynb)

#### **<Target에 대한 분포>**
![image](https://user-images.githubusercontent.com/80459520/138265731-c28e4812-f03d-478f-bedb-e61b1d95709e.png)
- 비대칭도가 심하게 왼쪽으로 기울어져 있기 떄문에 Log scaling을 적용하는 것이 모델의 예측성능이 더 좋을 것이라고 판단.

#### **Baseline Simple Regression**
- One-Hot Encoding 적용 X
    - MSE : 103723, 시간과 요일에 대한 Feature Inportance가 높음.
- One-Hot Encoding 적용
    - MSE : 27045, 결과에 대한 Feature 해석 모호

#### **모델링 진행방향**
One-Hot Encoding을 진행했을 때 성능은 좋아지지만 해석이 모호함. 따라서 한 차원에서 범주형 전처리를 하는 방향으로 결정

1. One-Hot Encoder에서 Label Encoder로 변경하여 범주형 변수의 전처리 진행 >> 차원을 줄이고, 결과해석을 높이기 위해
2. 다양한 Ensemble 모델링을 통해 가장 높은 성능을 보유한 모델 선정
3. 추가적인 Feature Engineering을 통해 모델의 성능 향상

---

### **4. Ensemble Modeling**
> [04_ensemble_modeling.ipynb](https://github.com/aeea-0605/taxi-demand-repo/blob/main/notebooks/04_ensemble_modeling.ipynb)
- sacred를 사용해 모델링 결과를 log로 기록 : [experiments](https://github.com/aeea-0605/taxi-demand-repo/tree/main/airflow/data/taxi-demand-prediction/experiments)
    - 1. Raw Target, XGBoost
    - 2. Log scaled Target, XGBoost
    - 3. Raw Target, LightGBM
    - 4. Log scaled LightGBM
    - 5. Raw Target, Random Forest
- RandomizedGridSearchCV를 사용해 200번의 task, 3번의 교차검증을 진행

#### **Modeling history**
<img width="738" alt="스크린샷 2021-10-21 오후 11 41 19" src="https://user-images.githubusercontent.com/80459520/138301215-da3a9157-94ed-4140-8735-03ae570812a7.png">

- Best Model : MSE가 11397.45로, Experiment_2인 Log scaled Target, XGBoost가 선정

#### **Best model의 실제값과 예측값 비교를 통한 성능 평가**
<img width="807" alt="스크린샷 2021-10-21 오후 11 36 31" src="https://user-images.githubusercontent.com/80459520/138300290-0c74515b-1642-4049-ac1e-224ffe7f1729.png">

- datetime : 26~28일에 실제값이 예측값에 현저히 못미치며 큰 차이 발생
- hour : 새벽엔 거의 유사하지만, 7~20시까지 실제값에 비해 예측값이 높게 측정
- weekday : 월, 화에 실제값보다 예측값이 높게 측정
- 평일/주말 : 주말보다 평일의 예측성능이 낮음

#### **Feature Engineering 진행방향**
수요가 갑자기 떨어진 기간에 대해서는 예측이 안되는 것을 알 수 있다. (특정 Event에 대해 예측력이 떨어짐)

- 과거 시점의 수요 데이터를 반영해 학습시킴으로써 Event에 대응할 수 있도록 feature engineering 진행

---

### **5. Feature Engineering & Final Modeling**
> [05_feature_engineering.ipynb](https://github.com/aeea-0605/taxi-demand-repo/blob/main/notebooks/05_feature_engineering.ipynb)
- 과거 수요에 대한 count 및 평균 & 표준편차에 데이터 추가
- feature engineering이 완료된 데이터셋으로 final modeling 진행
    - final model 또한 best model의 모델, 파라미터와 동일
- 다양한 관점에서 모델의 결과 분석 진행

#### **Final Model의 metric 성능 평가**
<img width="235" alt="스크린샷 2021-10-22 오전 12 06 49" src="https://user-images.githubusercontent.com/80459520/138305845-e7dbd2e7-a0cb-486b-a271-939bbb37a367.png">

- MSE 기준 모델의 성능 변화 : 103723 > 27045 > 11397 > 919

#### **Shap-value를 통한 Feature Impact 분석**

< SHAP Feature Importance & Summary Plot >
<img width="882" alt="스크린샷 2021-10-22 오전 12 41 28" src="https://user-images.githubusercontent.com/80459520/138311572-c30ae5f5-ad46-4f2a-89b8-c2bdd22914a6.png">

- Target에 대한 영향력 TOP3 특성은 `lag_1h_cnt` > `avg_7d_cnt` > `lag_1d_cnt` 이며, 모두 Target에 대해 양의 영향력을 준다.
- 범주형 변수인 `zip_code_le`는 Target에 대해 음의 영향력을 준다.
- 두 그래프를 비교해보았을 때, `std_7d_cnt`, `day` 특성은 feature value에 따른 shap value의 상관성이 모호해 해석의 모호성이 존재한다.

< SHAP Dependence Plot >
<img width="794" alt="스크린샷 2021-10-22 오전 1 02 04" src="https://user-images.githubusercontent.com/80459520/138314971-caf65acf-8ede-4ab6-9f45-e2ddb353848c.png">

- `lag_1h_cnt` : 상관성이 강한 변수는 `lag_1d_cnt`이며, [0, 500] 구간에서 lag_1d_cnt값이 높을 때 Target에 대해 영향력이 크다.
- `avg_7d_cnt` : 상관성이 강한 변수는 `hour`이며, [0, 200] 구간에서 hour값이 높을 때 Target에 대해 영향력이 낮다.
- `lag_1d_cnt` : 상관성이 강한 변수는 `lag_1h_cnt`이며, lag_1h_cnt값이 높을 때 Target에 대해 영향력이 낮다.
- `zip_code_le` : 상관성이 강한 변수는 `hour`이며, hour값이 높을 때 50근처에 있는 지역들은 Target에 대해 영향력이 크다.

#### **Final model의 실제값과 예측값 비교를 통한 성능 평가**
<img width="802" alt="스크린샷 2021-10-22 오전 1 08 59" src="https://user-images.githubusercontent.com/80459520/138316042-6e0a6357-a9e6-43cf-9526-1753a2440db3.png">

- Best model의 성능 평가 결과와 비교했을 떄, 전반적으로 실제값과 비슷하게 예측이 된다는 것을 알 수 있다.
- 특정 Event에 대응할 수 있는 예측성능을 보유

#### **Google Composer - Airflow 진행방향**
- Train, Predict에 대한 2개의 dags 구성
- Bask Operator를 사용해 task를 구성하고, bash command로 실행시킬 main python file은 변수를 입력받아 특정 환경에서 실행될 수 있게 구성
- BigQuery를 사용한 데이터 Extract 및 Load를 위해 gcp connection에 사용할 jwt 생성

<br/>

---
---
## 3. Airflow 결과
<br/>
