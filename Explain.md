# KDE (Kernel Density Estimation) 이란? 
  커널함수 (kernel function)을 이용한 밀도추정(density estimation) 방법의 하나이다. 
# 밀도추정(density estimation) 이란?
  pdf를 구하는 것
# 밀도추정 방법의 종류
  * parametric : pdf에 대한 모델을 정해 놓고 데이터들로 부터 모델의 파라미터만 추정
  * non-parametric density estimation : 순수히 데이터만으로 pdf 를 추정
    * 가장 간단한 방법이 히스토그램
    but 히스토그램은 사용하기 힘든 문제가 있기 때문에
    Kernel Density Estimation(커널 밀도 추정) 방법을 사용한다. 
# Kernel function 이란 ? 
  원점을 중심으로 대칭이면서 적분값이 1인 non-negative 함수
  Gaussain, Epanechnioov, uniform 함수 등이 대표적이다.

# Gaussain kernel ?
  주어진 데이터를 고차원 특징 공간으로 사상해주는 것
  가우시안 커널을 사용함으로써 무한 공간의 차원으로 맵핑시킬 수 있다. 즉, 비교가 가능해 진다. 
  지금 여기서는 하나의 레이어를 기준으로 모든 레이어를 비교 하는 것.
  가우시안 커널로 고차원으로 맵핑시킨다고 생각을 하면 된다. 
  
  
출저 : https://darkpgmr.tistory.com/147
  
  
                        
