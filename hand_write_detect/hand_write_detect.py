# Import dataset
from tensorflow.keras.datasets import mnist
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image as image_utils
import PIL.ImageOps

#讀模型遇到問題
# import json
# from keras.models import model_from_json

(X_train, y_train),(X_valid,y_valid) = mnist.load_data()

#預覽
plt.imshow(X_train[0], cmap='gray')
y_train[0]

plt.imshow(X_train[55555], cmap='gray')
y_train[55555]

#處理圖片資料(2維👉1維👉0~1)
X_train_1D = X_train.reshape(60000,784) #陣列調整形狀 2d->1d 長(28)*寬(28)=784 有60000筆資料(圖片)
X_valid_1D = X_valid.reshape(10000,784) #陣列調整形狀 2d->1d 長(28)*寬(28)=784 有10000筆資料(圖片)

X_train_1D_normal = X_train_1D/255 #標準化
X_valid_1D_normal = X_valid_1D/255 #標準化
#確認標準化
X_valid_1D.min() # 0 ~ 255 
X_valid_1D_normal.max() # 0.0 ~ 1.0

num_categories = 10
y_train
y_train_category = keras.utils.to_categorical(y_train, num_categories) #自動分類標示成0~9
y_valid_category = keras.utils.to_categorical(y_valid, num_categories)
# 測試看看 
y_train[0]
print(y_train_category[0])

y_valid_category = keras.utils.to_categorical(y_valid, num_categories)

# 測試看看
y_valid[0]
print(y_valid_category[0])

'''
#模型建構
model = Sequential()
model.add(Dense(units=512, activation='relu', input_shape=(784, )))
model.add(Dense(units=512, activation='relu'))
model.add(Dense(units=10, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', metrics=['accuracy']) 

model.fit(X_train_1D_normal, y_train_category, epochs=5, verbose=1, validation_data=(X_valid_1D_normal, y_valid_category)) 
#epochs:循環驗證次數  verbose:顯示資料的方式  validation_data:驗證資料

#驗證
# model.predict(X_valid_1D_normal[0]) #錯誤
# model.predict(np.array(X_valid_1D_normal[0])) #模型只吃NUMPY ARRAY #轉換成NUMPY ARRAY

model.predict(np.array([X_valid_1D_normal[0]])) #列出所有預測值
np.argmax(model.predict(np.array([X_valid_1D_normal[0]]))) #抓取最高的預測值
plt.imshow(X_valid[0], cmap='gray')

#存 model
model.save('basic-nn-model.h5')
'''

#---------------------------------------------------------------------

#讀 model
# model_2 = keras.models.load_model('mnist-nn-model-20220512') #.代表上個資料夾
model_2 = keras.models.load_model('basic-nn-model.h5')

np.argmax(model_2.predict(np.array([X_valid_1D_normal[0]])))
plt.imshow(X_valid[0], cmap='gray')

np.argmax(model_2.predict(np.array([X_valid_1D_normal[666]])))
plt.imshow(X_valid[666], cmap='gray')

def predict_number(image_path):
    # 可以處理本地檔案，也可以處理網路圖片URL
    if "http" in image_path:
        image_path = keras.utils.get_file(origin=image_path)
    test_image = image_utils.load_img(image_path, color_mode='grayscale', target_size=(28,28))
    test_image_inverted  = PIL.ImageOps.invert(test_image)
    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(test_image, cmap='gray')
    axarr[1].imshow(test_image_inverted, cmap='gray')

    test_image_array = image_utils.img_to_array(test_image_inverted)
    test_image_array_1D = test_image_array.reshape(1,784)
    test_image_array_1D_normal = test_image_array_1D/255
    return  np.argmax(model_2.predict(test_image_array_1D_normal)) 



predict_number(r"number/0.jpg")
predict_number("https://image.covertness.cn/shenduxuexichutan_20161006_180814.jpg")
predict_number("https://previews.123rf.com/images/aroas/aroas1704/aroas170400061/79321951-handwritten-sketch-black-number-5-on-white-background.jpg")