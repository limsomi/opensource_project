# 머신러닝을 이용한 뇌체조 인식
치매 예방에 좋은 뇌체조를 실생활에서도 쉽게 접할 수 있게 뇌체조 손동작을 학습시킨 모델을 사용하여 사용자 손의 좌표를 검출하여 뇌제조 동작을 인식하는 프로그램

## 파일설명
- object_detection.ipynb : object detection 모델을 생성
- skeleton_extraction.ipynb : object detetion 모델을 사용하여 거기에 맞춰 사진을 자른다. mediapipe를 사용하여 자른 사진 부분에서 skeletion extration을 실행하여 각각의 좌표값을 파일로 저장한다.
- machine_learning.ipyng : XGB,RandomForest, Gradient Boosting, Ada Gradient Boosting 등 모델 생성
- main.py : 뇌체조 인식 프로그램 실행

## 실행과정
### git clone https://github.com/ultralytics/yolov5.git 
위의 코드를 실행한 후 안에 있는 detect.py 파일을 지우고 이 프로젝트에 존재하는 detect.py 파일로 바꿔준다.


