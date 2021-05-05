# Bioinformatics_project
2021 생명정보학 Project
RubisCO 단백질 서열 CNN!!

* * *
## .py 별 설명
process.py : EMBL code를 이용해서 Uniport에서 data download  
datas.py : 필요한 모든 함수들을 담아놓은 .py  
- data_list : 모든 정보를 단순히 포함하는 .py
- blosum62 : BLOSUM62 matrix
- nums : 0 ~ 1을 10개의 등급으로 바꿔주는 함수  
compro.py : motif의 개수를 알기 위해서 전처리를 실행하고, 이를 그래프로 그려 분석하는 .py  
- process : 3개의 list를 반환하는 함수 (반환 형태는 주석 참고)
main.py : CNN을 하기 위한 전처리 과정 + CNN 학습
network.py : CNN을 구연한 .py [[출처]](https://github.com/MichalDanielDobrzanski/DeepLearningPython)  

* * *
## 필요한 module
numpy  
matplotlib  
selenium  
pandas  
Biopython  
