# -*- coding: utf-8 -*-
"""CV_CNN.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1H9Zy_tT1-_3N9nxlLdO4M0SDFZYxUnoY
"""

!pwd

# import os //python
# os.getcwd()
!ls -al

!mkdir image

!cp /content/drive/MyDrive/CV/101_ObjectCategories.tar.gz ./image/

# Commented out IPython magic to ensure Python compatibility.
# %cd image
!ls -al

!tar xvfz ./101_ObjectCategories.tar.gz > test.log

!pwd # test.log : 압축해제 내용

# Commented out IPython magic to ensure Python compatibility.
# %cd ../

import numpy as np
import os, glob #path 제어?
import cv2
from sklearn.model_selection import train_test_split

caltech_dir = './image/101_ObjectCategories'
categories = ['chair', 'camera', 'butterfly','elephant', 'flamingo']
num_classes = len(categories)

w = 64
h = 64
pixels = w*h*3 # color채널

X = [] # index
Y = [] # label

for idx, cat in enumerate(categories): # return index, catrfory 
  label = [0 for i in range(num_classes)] # 5개의 0으로 되어있는 label (for One-hot encoding) // sklearn 함수로도 가능
  label[idx] = 1
  print(f'label of {cat}: {label}')

  image_dir = os.path.join(caltech_dir,cat) # directory 찾기
  print(image_dir)
  files = glob.glob(os.path.join(image_dir,'*.jpg')) # img dir에서 jpg list 얻어오기

  for i, f in enumerate(files):
    
    img = cv2.imread(f) #img = Image.open(f) //-> np로 반환
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #img = img.convert('RGB') //cv로 읽어서 BGR
    img = cv2.resize(img, (w,h)) #64x64
    
    X.append(img)
    Y.append(label)    

# convert to ndarray!
X = np.array(X)
Y = np.array(Y)

X.shape # 334, 3, 64, 64 : tf?나 torch?는 chennel먼저 반환 -> 이게 더 빠름

Y.shape

X_train, X_test, y_train, y_test = train_test_split(X,Y) # random하게 shuffle도 해줌

xy = (X_train,X_test,y_train,y_test)
np.save('./image/5cat.npy',xy) # np format으로 저장 -> 저장해 둔 걸로 작업
print('ok!', np.shape)
print('# of train samples:',len(X_train))
print('# of test samples:', len(X_test))

84/(250+84)

from google.colab.patches import cv2_imshow # 주피터 cv.imshow - 창띄움 -> 코렙에서는..별루 그니까 matplotlib

idx = 10
print(y_train[idx])
# cv2_imshow(X_train[idx],cv2.COLOR_BGR2RGB)
cv2_imshow(cv2.cvtColor(X_train[idx],cv2.COLOR_BGR2RGB))

import tensorflow_datasets as tfds
import tensorflow as tf # alias

train_dataset = tf.data.Dataset.from_tensor_slices((X_train,y_train)) # (np) train/label / tf dataset
test_dataset = tf.data.Dataset.from_tensor_slices((X_test,y_test))
train_dataset

batch_train_dataset = train_dataset.batch(batch_size=8,drop_remainder=True) # drop_~ true : 날림 / iterater? python range
batch_test_dataset = test_dataset.batch(batch_size=8,drop_remainder=True)
batch_train_dataset

def format_label(label):
  string_label = categories[tf.math.argmax(label)] # 가장 큰 idx(1) 반환 / one-hot -> 문자열
  return string_label

import matplotlib.pyplot as plt # convert 전에 해서 안해도 됨 (RGB)

for i, (image, label) in enumerate(train_dataset.take(9)): # 배치 안묶어있음. 64x64x3
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image.numpy().astype("uint8")) # img -> np
    plt.title(f"{format_label(label)}")
    plt.axis("off") # img로 보려고 눈금 끔

# 전처리 끝

from tensorflow.keras import layers, models

# 시퀀셜X -> functional

inputs = layers.Input(shape=(w,h,3)) # input선언 rank(축) 3
x = layers.Conv2D(32,3,3, # channel 32, mask 3x3 
                  #activation='relu',
                  padding='same')(inputs)
x = layers.Activation('relu')(x) # 비선형적인 요소를 넣어주는 거
x = layers.MaxPooling2D(pool_size=(2,2))(x) # pooling기법 : 공간 축소 # 2x2로 되어있는 걸 큰것만 남기고 하나로 줄음 -> 한 평면의 다양한 특징들이 많이 나오게 해줌
x = layers.Dropout(0.25)(x) # 데이터가 적어서 제너레이션? overfit 피하기 위함

x = layers.Conv2D(64,3,3,
                  #activation='relu',
                  padding='same')(x)
x = layers.Activation('relu')(x)
x = layers.Conv2D(64,3,3,                  
                  #activation='relu',
                  padding='same')(x)                  
x = layers.MaxPooling2D(pool_size=(2,2))(x)
x = layers.Dropout(0.25)(x) 
# 여기까지// CNN : conv-pooling-activation : feature map으로 바뀌는 거
# low level

# 일자로 핌
x = layers.Flatten()(x)
x = layers.Dense(512)(x) # fully connected
x = layers.Activation('relu')(x) # 비선형적 요소 가미
x = layers.Dropout(0.5)(x)
x = layers.Dense(num_classes)(x) # 5개 (원핫)
outputs = layers.Activation('softmax')(x) # mulfi class classification / 확률이 되도록 해주기 때문 (합 1)

model = models.Model(inputs = inputs, outputs= outputs)
model.compile( # 학습한다~
    optimizer='adam', # cost f 줄여나가는데 사용할 알고리즘
    loss='categorical_crossentropy', # loss f. 퍼포먼스 평가 binary classification: logistic regression등 목적에 따라 사용 많이 하는게 있음
    metrics=['accuracy'] # 참인것 중 precision ㄷ아아 그 혼동행렬에서 쓰는 그거 
)
model.summary() # 모델 구성 출력

epochs = 50 # @param {type:'slider', min:10, max:100}
hist = model.fit(batch_train_dataset, # fit을 통해 training. 학습 시작
                 epochs=epochs, # 반복
                 validation_data = batch_test_dataset, # v(결과를 보면서 하이퍼 파라미터 튜닝을 한다. epoch등)와 t를 따로 나누는게 맞는데 연습이니까 그냥 같이 쓴다.
                 verbose=2) # log의 정도. auto로 넣어도 상관 X

score = model.evaluate(batch_test_dataset)
print(f'loss={score[0]}')
print(f'accuracy={score[1]}')

import matplotlib.pyplot as plt


def plot_hist(hist):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()

plot_hist(hist)

# overfit. generalization이 안됐다? 
# data가 모자라서 생겼음
# Trainable params: 92,165 //64x64하는데 필요..
# 저만큼 필요한데 데이터가 너무 적다

pres = model.predict(X_test) # 실제 서비스 할 때 사용하는 거
print(pres.shape)

error_img_dir = os.path.join('.','error')
try: 
  os.mkdir('error') # 틀린거 확인 할 수 있게 저장
except FileExistsError as e:
  print(e)

cnt = 0
for idx,(pre_prob,ans_onehot) in enumerate(zip(pres, y_test)):  
  pre = pre_prob.argmax()
  ans = ans_onehot.argmax()
  if pre == ans:
    continue
  cnt += 1
  print(f'Error: predicted {format_label(pre_prob)} != label {format_label(ans_onehot)}')
  fstr = os.path.join(error_img_dir,f'{idx:02d}-{format_label(pre_prob)}-ne-{format_label(ans_onehot)}.png')
  cv2.imwrite(fstr, X_test[idx])

print(f'Error Rate = {cnt}/{len(X_test)}')

# 모델 저장
# H5는 간단한거 할 때 좋은듯. 하나로 만들어지기 때문에. 예전에 하던 방식

from tensorflow.keras import models
# Calling `save('my_model')` creates a SavedModel folder `my_model`.
model.save("my_model")

# It can be used to reconstruct the model identically.
reconstructed_model = models.load_model("my_model")

# Let's check:
np.testing.assert_allclose(
    model.predict(X_test), reconstructed_model.predict(X_test)
)

# ============================================================
# The reconstructed model is already compiled and has retained the optimizer
# state, so training can resume:
reconstructed_model.fit(batch_train_dataset) # batch를 잡고...
#reconstructed_model.fit(X_test, X_test)

# 데이터 적어서 overfit
# image net 쓰자 (이미 만들어져 있는 모델 pretrained) efficeintNetB0 효율적이고 파라메터 적고 괜춘..
# VGG, ResNet 공부로 써봐 (좀 단순한 편)
# keras.io/api/applications

from tensorflow.keras.applications import EfficientNetB0
model = EfficientNetB0(weights='imagenet')

from tensorflow.keras.applications.resnet50 import  ResNet50
inputs = layers.Input(shape=(w,h,3))
outputs = ResNet50(include_top=True, # 상단 분류기 모델도 추가할게
                         weights=None, # 처음부터 시작 하겠어
                         input_shape=(w,h,3), # 인풋에 맞춤
                         classes=num_classes)(inputs) # 5개 필요해

model = models.Model(inputs = inputs, outputs= outputs)
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

epochs = 40 # @param {type:'slider', min:10, max:100}

# hist = model.fit(X_train, y_train,
#                  epochs=epochs,
#                  #validation_data = (X_test, y_test),
#                  )
hist = model.fit(batch_train_dataset,
                 epochs=epochs,
                 validation_data = batch_test_dataset,
                 verbose=2)

# Trainable params: 23,544,837
# loss 완벽하게 외웠군 -> 오버핏

from tensorflow.keras.applications.efficientnet import  EfficientNetB0
inputs = layers.Input(shape=(w,h,3))
outputs = EfficientNetB0(include_top=True, 
                         weights=None,
                         input_shape=(w,h,3),
                         classes=num_classes)(inputs)

model = models.Model(inputs = inputs, outputs= outputs)
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

epochs = 40 # @param {type:'slider', min:10, max:100}

# hist = model.fit(X_train, y_train,
#                  epochs=epochs,
#                  #validation_data = (X_test, y_test),
#                  )
hist = model.fit(batch_train_dataset,
                 epochs=epochs,
                 validation_data = batch_test_dataset,
                 verbose=2)

plot_hist(hist)

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

img_augmentation = Sequential(
  [
    # representing lower and upper bound for rotating clockwise and counter-clockwise. 
    preprocessing.RandomRotation(factor=0.15), # a float represented as fraction of 2pi, ex :0.15 (= 54 degree!) 
    
    preprocessing.RandomTranslation(height_factor=0.1, # lower and upper bound for shifting vertically
                                    width_factor=0.1 #lower and upper bound for shifting horizontally.
                                    ),
    preprocessing.RandomFlip(), # Randomly flip each image horizontally and vertically.
    preprocessing.RandomContrast(factor=0.1),
  ],
  name="img_augmentation",
) # 각각의 layer가 순서대로? 실행된다고 생각

for image, label in train_dataset.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    aug_img = img_augmentation(tf.expand_dims(image, axis=0))
    plt.imshow(aug_img[0].numpy().astype("uint8")) # 이동에 대한 ~가 줄어들고, 일반화 잘됨. 데이터는 많아지겠지. 학습시간도
    
    plt.title(f"{format_label(label)}")
    plt.axis("off")

from tensorflow.keras.applications.efficientnet import  EfficientNetB0
from tensorflow.keras import layers, models
from tensorflow.keras import layers

inputs = layers.Input(shape=(w,h,3))
x = img_augmentation(inputs)
outputs = EfficientNetB0(include_top=True, 
                         weights=None,
                         input_shape=(w,h,3),
                         classes=num_classes)(x)

model = models.Model(inputs = inputs, outputs= outputs)
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

epochs = 60 # @param {type:'slider', min:10, max:200}
batch_size = 8

hist = model.fit(batch_train_dataset,
                 epochs=epochs,
                 validation_data = batch_test_dataset,
                 verbose=2)

plot_hist(hist)

# 위에다 붙여서 하는 방법으로 
# pretrained 된 data weight를 가져와서 내껄로 transform 
# 자기가 만든 모델에 저기 있는  pre trained된 모델의 weight값들만 사용하려면 어떻게 해야하나요?

inputs = layers.Input(shape=(w,h,3))
x = img_augmentation(inputs)
model = EfficientNetB0(include_top=False, # 내가 새로 만들어서 하겠다
                       input_tensor=x, 
                       weights="imagenet") # weight를 불러오겠다!

# Freeze the pretrained weights
model.trainable = False # 일단을 얼려서 뒤에 붙이는 layer(img net의 feaure map을 입력으로 내 문제에서도 잘 작동하겠지 가 전제)가 훈련되도록 함.

# Rebuild top
x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output) # 펴줘?
x = layers.BatchNormalization()(x)

top_dropout_rate = 0.2
x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
outputs = layers.Dense(num_classes, # 출력은 공통이 아니니 여기 웨이트는 새로 만드는게 낫겠다 (학습은 넷 피쳐맵으로 먼저 해서 비슷비슷한 값 나오게)
                       activation="softmax", 
                       name="pred")(x)

# Compile
model = tf.keras.Model(inputs, outputs, name="EfficientNet")
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
model.compile(optimizer=optimizer, 
              loss="categorical_crossentropy", 
              metrics=["accuracy"])

epochs = 25  # @param {type: "slider", min:8, max:80}
hist = model.fit(batch_train_dataset,
                 epochs=epochs,
                 validation_data = batch_test_dataset,
                 verbose=2)
plot_hist(hist)

# img net(영상인식 대회. 딥러닝이 너무 잘 맞춰서 종료)데이터로 처리 -> 쌓은 데이터 사용(1000개?)
# 좋은 feature map이 나옴 -> 사용
# 다른 모델의 weight를 사용 하려면 제가 만든 모델에 그 모델의 구조가 포함되어야만 사용 할 수 있는거죠?
# fine tuning
# learn linear?를 작게 조금씩 바꿔감

def unfreeze_model(model):
    # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
    for layer in model.layers[-20:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )


unfreeze_model(model)

epochs = 15  # @param {type: "slider", min:8, max:50}
hist = model.fit(batch_train_dataset,
                 epochs=epochs,
                 validation_data = batch_test_dataset,
                 verbose=2)
plot_hist(hist)

# !!data!! // small data -> agumentation(train) -> knowlege transfer??(img net 같은데 api 가져와서 architeture기반으로 pretrain으로 학습을 시킨후 필요없는거 떼내고 내가 붙인걸 얼리고 이미지넷 데이터 웨이트 내껄로 보강 후 얼린 데이터 풀어서 재학습 -> 고단한작업..)
# parameter 가 적으면서 최신 network
# vgg 우승 못했지만 사람이 이해하기 쉽고 좋기때문에. 이해를 통한 tuning. 성능보단 공부

# cam
# 왜 이런 판단을 내렸는지
# 어떤 클래스가 얼마만큼의 엑티베이션을 일으켰는디
# 판정을 하는데 있어서 어느 부분을 가지고 판정했는지 검증 가능
# 내 모델이 어떻게 판정을 내렸는지

# data 전처리
# tf 가 업무용으로는 더 많이 씀 / pytorch / opencv, np도 가능
# 기존의 모델에서 transfer learning을 권함. 최신에 대해 알아야 함. 데이터 셋이 보통은 적을테니..
# fine tuning
# cam을 통해 확인

## 남이 만들어 둔 것을 최대한 사용해라! 프로그래머는 바퀴를 두개 만들지 않는닷
# featur - 영상의 정보를 가장 잘 보존하고 있다고 볼 수 있다

