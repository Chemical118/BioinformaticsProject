# Bioinformatics project v1.2
2021 생명정보학 Project  
RubisCO 단백질 서열 CNN & 유전알고리즘을 이용한 최대 찾기

### v1.1
- keras 모델로 학습하는 것을 기본으로 함
- 유전알고리즘을 이용해 최대 찾는 부분 추가
- 코드에 오류 수정 (70%에서 88%수준으로 정확도 향상)

### v1.2
- 회귀분석을 이용한 부분 추가 (rmain.py)
- 회귀분석 실행 시에 R<sup>2</sup> 0.6 ~ 0.75 정도 나옴
* * *
## 파일 설명
#### process.py : EMBL code를 이용해서 Uniport에서 data download  
#### phocbp.py : find.py 결과를 분석하기 위한 .py  
#### datas.py : 필요한 모든 함수들을 담아놓은 .py  
- data_list : 모든 정보를 단순히 포함하는 .py
- blosum62 : BLOSUM62 matrix
- nums : 0 ~ 1을 10개의 등급으로 바꿔주는 함수  
#### compro.py : motif의 개수를 알기 위해서 전처리를 실행하고, 이를 그래프로 그려 분석하는 .py  
- process : 3개의 list를 반환하는 함수
#### kmain.py : 같은 data를 CNN을 학습하기 위해 keras에 맞추어서 변형한 code (thanks to greendaygh)
#### ktmain.py : 같은 data를 논문에서 제시한 위치만 학습하도록 변형한 code
#### keras_rubisco : 저장된 CNN Model  
#### find.py : 저장된 Model을 이용헤, 지시적 진화를 통해 최대값과 해당 단백질 서열 찾기  
#### ans 폴더
- ans.txt : find.py 실행 결과
- data.xml : Blastp 실행 결과
- kmain.txt : kmain.py 실행 결과
- ktmain.txt : ktmain.py 실행 결과
#### rmain.py : 회귀 분석을 이용한 분석
* * *
## 필요한 module
numpy  
matplotlib  
selenium  
pandas  
Biopython  
keras  
sklearn  
