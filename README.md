# 1. 본 파일의 구조
> 1. echocardiography 폴더  
> + 가장 중요한 폴더이다. 여기에 데이터를 입력한다. 폴더 구조에 맞게 데이터들을 분류하여 넣어야 한다.  
-Directory 구조-  
ㄴechocardiography/train  
ㄴㄴechocardiography/train/A2C  
ㄴㄴechocardiography/train/A4C  
ㄴechocardiography/validation  
ㄴㄴechocariography/validation/A2C  
ㄴㄴechocardiography/validation/A4C  
ㄴechocardiography/test  
ㄴㄴechocardiography/test/A2C  
ㄴㄴechocardiography/test/A4C  

> 2. SAUNet에 관한 폴더  
> + ckpt\SAUNet  
SAUNet의 모델이 이곳에 저장된다.
> + Loss_we_use  
SAUNet의 Loss function이 이곳에서 정의된다.
> + Models_we_use  
SAUNet 모델이 이곳에서 정의된다.
> + shape_attentive_unet  
SAUNet 모델이 이곳에서 정의된다.

> 3. PSPnet에 관한 폴더  
> + utils  
PSPnet, dataloader, HeartDisease 클라스가 이곳에서 정의된다.  
(HeartDisease 클라스는 이미지 preproccessing 클라스이다.)
> + weights
PSPnet의 모델이 이곳에 저장된다.

> 4. ckpt_for_test  
ensemble 모델의 전반적인 test가 여기서 진행된다.




# 2. SEVSNUK-net 사용법
## 유의사항: 꼭!! echocardiography 파일 내에 아래 설명대로  directory를 구성하고 Data를 넣은 뒤 실행시켜주세요!!
SEVSNUK-net을 사용할 때 가장 핵심 파일은 train.py와 test.py이다.

train.py는 SEVSNUK-net을 학습시키고자 할 때 실행하는 파일이고

test.py는 이미 학습되어진 SEVSNUK-net를 test하고 싶을 때 실행하는 파일이다.

기본적으로 이 모델은 A2C와 A4C 두 데이터 모두를 사용한다.  
만약 어느 한쪽의 경우에만 평가하고 싶다면 echocardiography/test/A2C 또는 A4C에만 데이터를 넣어야 한다. (Directory도 제거 바람)  
test, validation의 경우도 마찬가지이다. 

> 1. test만 하고 싶을때  
echocardiography/test 내에 /A2C 및 /A4C 디렉토리 모두 존재하고 안에 데이터 있을 경우:  
-> python3 test.py 실행시 A2C/A4C 전체에 대한 DSC/JI 값 평균을 출력함  
echocardiography/test 내에 /A2C 디렉토리만 존재하고 안에 데이터 있을 경우:  
-> python3 test.py 실행시 A2C에 대한 DSC/JI 값 평균을 출력함  
echocardiography/test 내에 /A4C 디렉토리만 존재하고 안에 데이터 있을 경우:  
-> python3 test.py 실행시 A4C에 대한 DSC/JI 값 평균을 출력함  

> 2. train만 하고 싶을때  
python3 train.py만 실행  
-> 단, 이경우 끝나면 ckpt_for_test 내에 저장된 model들이 달라지기 때문에 test.py 실행시 결과 달라질 수 있음

> 3. train하고 test하고 싶을때  
python3 train.py  실행 후 python3 test.py 실행하면 됨

## 1) train.py
train.py는 다음과 같이 구성된다.

    #1. Make HDF5 Files
    convert_data_2_hdf5(test_directory='validation', rootdir = "echocardiography/")

    #2. Train SAUNet
    train_saunet_final(max_epoch=60)

    #3. Train PSPNet
    training()

    #4. Move Best Models to Safe Place
    mv_best_models()

    #5. Ensemble Models
    ensemble()

1. SAUNet을 위한 data-preprocessing (data directory 내 hdf5 file 생성)

2. SAUNet training (60 epoch) 진행 (모델은 ckpt/SAUNet에 저장됨)

3. Default 파라미터로 PSPNet training 진행 (모델은 weights에 저장됨)

4. training이 모두 끝나면, ckpt에 저장된 모델중 최선의 모델 (best validation model)을 ckpt_for_test directory로 copy
(각각 ckpt_for_test/SAUNet, ckpt_for_test/PSPNet에 저장)

5. ensemble 과정을 통해 훈련된 모델들을 ensemble 모델에 탑재, 그 후 관련 state dict를 ckpt_for_test/ensemble/model.pt에 저장

## 2) test.py
test.py는 다음과 같이 구성된다.

    if __name__ == '__main__':
        # 1. ensemble model 구성
        ensemble()
        # 2. model test
        test_FINAL_Net(ensemble_type="2")

1. ensemble 과정을 통해 훈련된 모델들을 ensemble 모델에 탑재, 그 후 관련 state dict를 ckpt_for_test/ensemble/model.pt에 저장  
-> 이걸 여기서도 하는 이유는 본래 training을 하면서 ckpt_for_test에 가장 좋은 모델을 저장해서 보낼껀데, training을 진행 안하고 바로 test를 할 것이기 때문에 ensemble을 미리 해주는 과정이다.

2. ensemble 모델로 echocardiography/test directory에 들어간 data에 대해 inference를 진행해 DICE/Jaccard index를 구함   
-> 단순히 echocardiography/test에 final evaluation data만 넣으면 된다.



# 2. (추가) SEVSNUK-net을 이용해 Mask Image 생성하는 법
## 유의사항: 꼭!! echocardiography 파일 내에 아래 설명대로  directory를 구성하고 Data를 넣은 뒤 실행시켜주세요!!
SEVSNUK-net을 사용하여 Mask Image (원본 Ground Truth와 크기 동일)을 생성하기 위해선 create_mask.py를 실행시키면 된다.

기본적으로 이 모델은 A2C와 A4C 두 데이터 모두를 사용한다.  
만약 어느 한쪽의 경우에만 Mask를 생성하고 싶다면 echocardiography/test/A2C 또는 A4C에만 데이터를 넣어야 한다. (Directory도 제거 바람)  
test, validation의 경우도 마찬가지이다. 

> 1. mask를 생성하고 싶은 경우  
echocardiography/test 내에 /A2C 및 /A4C 디렉토리 모두 존재하고 안에 데이터 있을 경우:  
-> python3 create_mask.py 실행시 A2C/A4C 전체에 대한 output mask가 output_test/A2C 혹은 output_test/A4C 내부에 저장됨  
echocardiography/test 내에 /A2C 디렉토리만 존재하고 안에 데이터 있을 경우:  
-> python3 create_mask.py 실행시 A2C에 대한 output mask가 output_test/A2C 내부에 저장됨  
echocardiography/test 내에 /A4C 디렉토리만 존재하고 안에 데이터 있을 경우:  
-> python3 create_mask.py 실행시 A4C에 대한 output mask가 output_test/A2C 내부에 저장됨  


## 1) create_mask.py
create_mask.py는 다음과 같이 구성된다.

    if __name__ == '__main__':
        # 1. ensemble model 구성
        ensemble()
        # 2. Create mask
        test_and_create_output()

1. ensemble 과정을 통해 훈련된 모델들을 ensemble 모델에 탑재, 그 후 관련 state dict를 ckpt_for_test/ensemble/model.pt에 저장  
-> 이걸 여기서도 하는 이유는 본래 training을 하면서 ckpt_for_test에 가장 좋은 모델을 저장해서 보낼껀데, training을 진행 안하고 바로 create_mask를 할 것이기 때문에 ensemble을 미리 해주는 과정이다.

2. ensemble 모델로 echocardiography/test directory에 들어간 data에 대해 inference를 진행해 얻은 최종 output mask를 저장  
-> 단순히 echocardiography/test에 final evaluation data만 넣으면 된다.  
-> output mask는 A2C 데이터의 경우 output_test/A2C에, A4C 데이터의 경우 output_test/A4C에 저장된다.