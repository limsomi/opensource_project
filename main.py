import csv
import os
import cv2
 

op_type=""
cap=cv2.VideoCapture(0)
if cap.isOpened():
    while True:
        if op_type=="":
            img=cv2.imread('./background_image.jpg', cv2.IMREAD_COLOR)
            img=cv2.resize(img, (1161,960), interpolation=cv2.INTER_LINEAR)
            cv2.imshow('output_keypoints_with_lines', img)
            while (1):
                if (cv2.waitKey(0)==ord('1')):
                    op_type="pp"
                    break
                elif (cv2.waitKey(0)==ord('2')):
                    op_type="cnv"
                    break
        else:
            break
cap.release()
cv2.destroyAllWindows()

input_path = './recorded_video/'
filename='video'
file_ext='.mp4'
 
output_path='./recorded_video/%s%s' %(filename,file_ext)
uniq=1
while os.path.exists(output_path):
  output_path='./recorded_video/%s(%d)%s' % (filename,uniq,file_ext)
  uniq+=1

cap = cv2.VideoCapture(0)
if cap.isOpened:
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = 25.40
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'DIVX'),
                         fps,(frame_width, frame_height))
    while True:
        ret,frame = cap.read()
        if ret == True:
            out.write(frame)
            cv2.imshow('Press ESC key when the recording ends',frame)
            if cv2.waitKey(1) & 0xFF== 27:
                break
        else:
            print('no file')
            break
            
cap.release()
out.release()
cv2.destroyAllWindows()

#영상 프레임 추출 함수

#프레임 저장할 디렉토리 생성
output_directory='./images/frame'
num=1
try:
    while os.path.exists(output_directory):
        output_directory='./images/frame(%d)' % (num)
        num+=1
    os.makedirs(output_directory)
except OSError:
    print ('Error: Creating directory. ' +  output_directory)
 


v_cap = cv2.VideoCapture(output_path)

cnt = 0
while v_cap.isOpened():
    try:
        ret, image = v_cap.read()
        image = cv2.resize(image, (1920, 1080))
        if int(v_cap.get(1)) % 10 == 0:
            image=cv2.flip(image,1)
            cv2.imwrite(output_directory+'/capture%d.jpg' % (v_cap.get(1)), image)
            print("Frame Captured: %d" % v_cap.get(1))
            cnt += 1
    except:
        break

v_cap.release()

#label 저장할 디렉토리 생성
label_directory='./object_detection/label'
num=1
try:
    while os.path.exists(label_directory):
        label_directory='./object_detection/label(%d)' % (num)
        num+=1
    os.makedirs(label_directory)
except OSError:
    print ('Error: Creating directory. ' +  label_directory)
    

#object detection 실행-->좌표값 txt 파일 얻기
from IPython import get_ipython
import subprocess
subprocess.run('python ./yolov5/detect.py --weights ./object_detection/%s_best.pt --img 416 --conf 0.5 --source "%s"'%(op_type,output_directory))



#skeleton extraction
import numpy as np
import mediapipe as mp
import pandas as pd


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3)
mp_drawing = mp.solutions.drawing_utils


image_list=os.listdir(output_directory)
label_list=os.listdir(label_directory)

keypoint_data_right=list()
keypoint_data_left=list()

for i in range(len(label_list)):
    image_file_name=label_list[i].split('.')[0]

    image=cv2.imread(output_directory+'/'+image_file_name+'.jpg')
    label=open(label_directory+'/'+label_list[i])
    exist_1=label.readline()

    if(exist_1!=""):
        line=exist_1.split()

        x1=int(line[1])
        x2=int(line[2])
        y1=int(line[3])
        y2=int(line[4])

        croppedImage1=image[y1:y2,x1:x2]
        class1=int(line[0])

    exist_2=label.readline()
    if(exist_2!=""):
        line=exist_2.split()
 
        x1=int(line[1])
        x2=int(line[2])
        y1=int(line[3])
        y2=int(line[4])


        croppedImage2=image[y1:y2,x1:x2]
        class2=int(line[0])

    results1 = hands.process(cv2.cvtColor(croppedImage1, cv2.COLOR_BGR2RGB))
    if(exist_2!=""):
        results2 = hands.process(cv2.cvtColor(croppedImage2, cv2.COLOR_BGR2RGB))


    if results1.multi_hand_landmarks:

        for hand_no, hand_landmarks in enumerate(results1.multi_hand_landmarks):
            sample=list()
            sample.append(class1)
            for j in range(21):
                sample.append(hand_landmarks.landmark[mp_hands.HandLandmark(j).value].x)
                sample.append(hand_landmarks.landmark[mp_hands.HandLandmark(j).value].y)
                sample.append(hand_landmarks.landmark[mp_hands.HandLandmark(j).value].z)
            if(exist_2!="" and ((op_type=="pp" and class1<=2) or (op_type=="cnv" and class1<=5))):
                keypoint_data_left.append(sample)
            else:
                keypoint_data_right.append(sample)
    else:
        sample=['' for i in range(64)]
        keypoint_data_right.append(sample)


    if exist_2!="" and results2.multi_hand_landmarks:
    
        for hand_no, hand_landmarks in enumerate(results2.multi_hand_landmarks):
            sample=list()
            sample.append(class2)
            for l in range(21):
                sample.append(hand_landmarks.landmark[mp_hands.HandLandmark(l).value].x)
                sample.append(hand_landmarks.landmark[mp_hands.HandLandmark(l).value].y)
                sample.append(hand_landmarks.landmark[mp_hands.HandLandmark(l).value].z)
            keypoint_data_left.append(sample)
    else:
        sample=['' for i in range(64)]
        keypoint_data_left.append(sample)


df_right=pd.DataFrame(keypoint_data_right,columns=['label','WRIST_X','WRIST_Y','WRIST_Z',
                                       'THUMB_CMC_X','THUMB_CMC_Y','THUMB_CMC_Z',
                                       'THUMB_MCP_X','THUMB_MCP_Y','THUMB_MCP_Z',
                                       'THUMB_IP_X','THUMB_IP_Y','THUMB_IP_Z',
                                       'THUMB_TIP_X','THUMB_TIP_Y','THUMB_TIP_Z',
                                       'INDEX_FINGER_MCP_X','INDEX_FINGER_MCP_Y','INDEX_FINGER_MCP_Z',
                                       'INDEX_FIGNER_PIP_X','INDEX_FIGNER_PIP_Y','INDEX_FIGNER_PIP_Z',
                                       'INDEX_FINGER_DIP_X','INDEX_FINGER_DIP_Y','INDEX_FINGER_DIP_Z',
                           'INDEX_FINGER_TIP_X','INDEX_FINGER_TIP_Y','INDEX_FINGER_TIP_Z',
                                       'MIDDLE_FINGER_MCP_X','MIDDLE_FINGER_MCP_Y','MIDDLE_FINGER_MCP_Z',
                                       'MIDDLE_FINGER_PIP_X','MIDDLE_FINGER_PIP_Y','MIDDLE_FINGER_PIP_Z',
                                       'MIDDLE_FINGER_DIP_X','MIDDLE_FINGER_DIP_Y','MIDDLE_FINGER_DIP_Z',
                                       'MIDDLE_FINGER_TIP_X','MIDDLE_FINGER_TIP_Y','MIDDLE_FINGER_TIP_Z',
                                       'RING_FINGER_MCP_X','RING_FINGER_MCP_Y','RING_FINGER_MCP_Z',
                                    'RING_FINGER_PIP_X','RING_FINGER_PIP_Y','RING_FINGER_PIP_Z',
                                       'RING_FINGER_DIP_X','RING_FINGER_DIP_Y','RING_FINGER_DIP_Z',
                                       'RING_FIGNER_TIP_X','RING_FINGER_TIP_Y','RING_FINGER_TIP_Z',
                                       'PINKY_MCP_X','PINKY_MCP_Y','PINKY_MCP_Z',
                                       'PINKY_PIP_X','PINKY_PIP_Y','PINKY_PIP_Z',
                                       'PINKY_DIP_X','PINKY_DIP_Y','PINKY_DIP_Z',
                           'PINKY_TIP_X','PINKY_TIP_Y','PINKY_TIP_Z'])

df_left=pd.DataFrame(keypoint_data_left,columns=['label','WRIST_X','WRIST_Y','WRIST_Z',
                                       'THUMB_CMC_X','THUMB_CMC_Y','THUMB_CMC_Z',
                                       'THUMB_MCP_X','THUMB_MCP_Y','THUMB_MCP_Z',
                                       'THUMB_IP_X','THUMB_IP_Y','THUMB_IP_Z',
                                       'THUMB_TIP_X','THUMB_TIP_Y','THUMB_TIP_Z',
                                       'INDEX_FINGER_MCP_X','INDEX_FINGER_MCP_Y','INDEX_FINGER_MCP_Z',
                                       'INDEX_FIGNER_PIP_X','INDEX_FIGNER_PIP_Y','INDEX_FIGNER_PIP_Z',
                                       'INDEX_FINGER_DIP_X','INDEX_FINGER_DIP_Y','INDEX_FINGER_DIP_Z',
                           'INDEX_FINGER_TIP_X','INDEX_FINGER_TIP_Y','INDEX_FINGER_TIP_Z',
                                       'MIDDLE_FINGER_MCP_X','MIDDLE_FINGER_MCP_Y','MIDDLE_FINGER_MCP_Z',
                                       'MIDDLE_FINGER_PIP_X','MIDDLE_FINGER_PIP_Y','MIDDLE_FINGER_PIP_Z',
                                       'MIDDLE_FINGER_DIP_X','MIDDLE_FINGER_DIP_Y','MIDDLE_FINGER_DIP_Z',
                                       'MIDDLE_FINGER_TIP_X','MIDDLE_FINGER_TIP_Y','MIDDLE_FINGER_TIP_Z',
                                       'RING_FINGER_MCP_X','RING_FINGER_MCP_Y','RING_FINGER_MCP_Z',
                                    'RING_FINGER_PIP_X','RING_FINGER_PIP_Y','RING_FINGER_PIP_Z',
                                       'RING_FINGER_DIP_X','RING_FINGER_DIP_Y','RING_FINGER_DIP_Z',
                                       'RING_FIGNER_TIP_X','RING_FINGER_TIP_Y','RING_FINGER_TIP_Z',
                                       'PINKY_MCP_X','PINKY_MCP_Y','PINKY_MCP_Z',
                                       'PINKY_PIP_X','PINKY_PIP_Y','PINKY_PIP_Z',
                                       'PINKY_DIP_X','PINKY_DIP_Y','PINKY_DIP_Z',
                           'PINKY_TIP_X','PINKY_TIP_Y','PINKY_TIP_Z'])

#skeleton data 저장
keypoint_path='./skeleton_extraction/'

filename='keypoint'
file_ext='.csv'
 
keypoint_file='./skeleton_extraction/%s_%s_%s%s' %(op_type,filename,'right',file_ext)
keypoint_file_left='./skeleton_extraction/%s_%s_%s%s' %(op_type,filename,'left',file_ext)
uniq=1
while os.path.exists(keypoint_file):
  keypoint_file='./skeleton_extraction/%s_%s_%s(%d)%s' %(op_type,filename,'right',uniq,file_ext)
  keypoint_file_left='./skeleton_extraction/%s_%s_%s(%d)%s' %(op_type,filename,'left',uniq,file_ext)
  uniq+=1

df_right.to_csv(keypoint_file,index=None)
df_left.to_csv(keypoint_file_left,index=None)

#model 불러오기
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder


def predict_op(keypoint_file,model,label):
    
    keypoint=pd.read_csv(keypoint_file)
    x=keypoint.iloc[:,1:]
    clf=xgb.XGBClassifier()

    clf.load_model(model)
    clf._le=LabelEncoder().fit(label)
    y_pred=clf.predict(x)
    return y_pred

pp_label=['l_fg','l_fist','l_tb','r_fg','r_fist','r_tb']
cnv_label=['l_1','l_2','l_3','l_4','l_5','l_fist',
    'r_1','r_2','r_3','r_4','r_5','r_fist']
    
pp_model='./model/xgboost_total_pp-1.model'
cnv_model='./model/xgboost_total_cnv-2.model'

    
if(op_type=="pp"):
    pred_r=predict_op(keypoint_file,pp_model,pp_label)
    pred_l=predict_op(keypoint_file_left,pp_model,pp_label)
elif(op_type=="cnv"):
    pred_r=predict_op(keypoint_file,cnv_model,cnv_label)
    pred_l=predict_op(keypoint_file_left,cnv_model,cnv_label)


pp_l_label=['l_fg','l_fist','l_tb']
pp_r_label=['r_tb','r_fist','r_fg']

cnv_l_label=['l_5','l_4','l_3','l_2','l_1','l_fist']
cnv_r_label=['r_fist','r_1','r_2','r_3','r_4','r_5']

def count_op(right_label,left_label,pred_r,pred_l):
    cnt=0
    for i in range(len(pred_r)-1):
        prev_r=pred_r[i]
        prev_l=pred_l[i]
        cur_r=pred_r[i+1]
        cur_l=pred_l[i+1]
        try:
            if(left_label.index(prev_l)==left_label.index(cur_l)-1 and left_label.index(prev_l)==right_label.index(prev_r)) :
                index=left_label.index(prev_l)
                if(right_label[index]==prev_r and cur_r==right_label[index+1]):
                    cnt+=1
            elif(prev_l==left_label[-1] and cur_l==left_label[0] and left_label.index(prev_l)==right_label.index(prev_r)):
                if(prev_r==right_label[-1] and cur_r==right_label[0]):
                    cnt+=1
        except:
            continue
    return cnt

if(op_type=="pp"):
    cnt=count_op(pp_r_label,pp_l_label,pred_r,pred_l)
elif(op_type=="cnv"):
    cnt=count_op(cnv_r_label,cnv_l_label,pred_r,pred_l)

print("count: "+str(cnt))
