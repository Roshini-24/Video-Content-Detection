from tkinter import *
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pickle
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import os
from sklearn.metrics import accuracy_score
from sklearn import svm
import imutils
import numpy as np
import pandas as pd

import cv2
from keras.layers import Bidirectional, LSTM
from keras.utils.np_utils import to_categorical
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split 
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
import pickle
from keras_efficientnets import EfficientNetB7
from keras.callbacks import ModelCheckpoint 

from keras.models import Sequential, Model, load_model
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, InputLayer, BatchNormalization, Dropout


main = tkinter.Tk()
main.title("A Deep Learning-Based Approach for Inappropriate Content Detection and Classification of YouTube Videos")
main.geometry("1200x1200")
content_model = load_model("model/content_model.h5")

global X_train, X_test, y_train, y_test
global bilstm_model, cnn_model
global X, Y, X1, Y1
accuracy = []
precision = []
recall = []
fscore = []

def uploadDataset():
    global filename, dataset, textdata, labels
    text.delete('1.0', END)
    filename = filedialog.askdirectory(initialdir=".")
    text.insert(END,str(filename)+" Dataset Loaded\n\n")
    pathlabel.config(text=str(filename)+" Dataset Loaded")
    img = cv2.imread("Dataset/InappropriateContent/316.jpg")
    img = cv2.resize(img, (500, 400))
    cv2.imshow("Loaded Sample Image", img)
    cv2.waitKey(0)

def preprocessDataset():
    text.delete('1.0', END)
    global X, Y, X1, Y1
    if os.path.exists("model/X.txt.npy"):
        X = np.load('model/X.txt.npy')
        Y = np.load('model/Y.txt.npy')
        X1 = np.load('model/X1.txt.npy')
        Y1 = np.load('model/Y1.txt.npy')
    else:
        X = []
        Y = []
        for root, dirs, directory in os.walk('Dataset'):
            for j in range(len(directory)):
                name = os.path.basename(root)
                if 'Thumbs.db' not in directory[j]:
                    img = cv2.imread(root+"/"+directory[j])
                    img = cv2.resize(img, (32, 32))
                    label = 0
                    if name == 'InappropriateContent':
                        label = 1
                    X.append(img)
                    Y.append(label)
                    print(name+" "+directory[j]+" "+str(label))        
        X = np.asarray(X)
        Y = np.asarray(Y)
        np.save('model/X1.txt',X)
        np.save('model/Y1.txt',Y)
    X = X.astype('float32')
    X = X/255
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    unique, count = np.unique(Y1, return_counts = True)
    Y = to_categorical(Y)
    text.insert(END,"Total images found in dataset : "+str(X1.shape[0])+"\n\n")
    text.insert(END,"Labels in dataset : Safe & Inappropriate Content")
    text.update_idletasks()
    height = count
    bars = ['Safe Content', 'Inappropriate Content']
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.title("Safe & Inappropriate Content found in dataset")
    plt.xlabel("Youtube Content Type")
    plt.ylabel("Count")
    plt.show()

def calculateMetrics(algorithm, predict, target):
    acc = accuracy_score(target,predict)*100
    p = precision_score(target,predict,average='macro') * 100
    r = recall_score(target,predict,average='macro') * 100
    f = f1_score(target,predict,average='macro') * 100
    text.insert(END,algorithm+" Precision  : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall     : "+str(r)+"\n")
    text.insert(END,algorithm+" F1-Score   : "+str(f)+"\n")
    text.insert(END,algorithm+" Accuracy   : "+str(acc)+"\n\n")
    text.update_idletasks()
    precision.append(p)
    accuracy.append(acc)
    recall.append(r)
    fscore.append(f)
    LABELS = ['Safe Content', 'Inappropriate Content']
    conf_matrix = confusion_matrix(target, predict) 
    plt.figure(figsize =(6, 6)) 
    ax = sns.heatmap(conf_matrix, xticklabels = LABELS, yticklabels = LABELS, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,2])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()


def runExistingSVM():
    global vectorizer, X, Y
    Y = np.argmax(Y, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    global accuracy, precision, recall, fscore
    svm_cls = svm.SVC(C=2.0, kernel="sigmoid")
    svm_cls.fit(X_train,y_train)
    predict = svm_cls.predict(X_test)
    calculateMetrics("EfficientNet-SVM", predict, y_test)
    
def runProposeAlgorithms():
    text.delete('1.0', END)
    global bilstm_model, cnn_model, X, Y
    global accuracy, precision, recall, fscore
    accuracy.clear()
    precision.clear()
    recall.clear()
    fscore.clear()

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #split dataset into train and tesrt
    eb = EfficientNetB7(input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]), include_top=False, weights=None)
    eb.trainable = False
    cnn_model = Sequential()
    cnn_model.add(eb)
    cnn_model.add(Convolution2D(32, (1, 1), input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
    cnn_model.add(MaxPooling2D(pool_size = (1, 1)))
    cnn_model.add(Convolution2D(32, (1, 1), activation = 'relu'))
    cnn_model.add(MaxPooling2D(pool_size = (1, 1)))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(units = 256, activation = 'relu'))
    cnn_model.add(Dense(units = y_train.shape[1], activation = 'softmax'))
    cnn_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])    
    if os.path.exists("model/model_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/model_weights.hdf5', verbose = 1, save_best_only = True)
        hist = cnn_model.fit(X_train, y_train, batch_size = 32, epochs = 50, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        cnn_model = load_model("model/model_weights.hdf5")
    cnn_model = Model(cnn_model.inputs, cnn_model.layers[-2].output)#creating cnn model
    cnn_features = cnn_model.predict(X)  #extracting cnn features from test data
    X = cnn_features
    print(X.shape)
    cnn_features = np.reshape(cnn_features, (cnn_features.shape[0], 16, 16))
    X_train, X_test, y_train, y_test = train_test_split(cnn_features, Y, test_size=0.2)
    bilstm_model = Sequential() #defining deep learning sequential object
    #adding LSTM bidirectional layer with 32 filters to filter given input X train data to select relevant features
    bilstm_model.add(Bidirectional(LSTM(32, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True)))
    #adding dropout layer to remove irrelevant features
    bilstm_model.add(Dropout(0.2))
    #adding another layer
    bilstm_model.add(Bidirectional(LSTM(32)))
    bilstm_model.add(Dropout(0.2))
    #defining output layer for prediction
    bilstm_model.add(Dense(y_train.shape[1], activation='softmax'))
    #compile GRU model
    bilstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #start training model on train data and perform validation on test data
    if os.path.exists("model/bilstm_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/bilstm_weights.hdf5', verbose = 1, save_best_only = True)
        hist = bilstm_model.fit(X_train, y_train, batch_size = 16, epochs = 20, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)    
    else:
        bilstm_model = load_model("model/bilstm_weights.hdf5")    
    predict = bilstm_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    target = np.argmax(y_test, axis=1)
    calculateMetrics("Propose EfficientNet-BiLSTM Algorithm", predict, target)
    
def meanLoss(image1, image2):
    difference = image1 - image2
    a,b,c,d,e = difference.shape
    n_samples = a*b*c*d*e
    sq_difference = difference**2
    Sum = sq_difference.sum()
    distance = np.sqrt(Sum)
    mean_distance = distance/n_samples
    return mean_distance

def predict():
    global content_model
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="testVideos")
    cap = cv2.VideoCapture(filename)
    print(cap.isOpened())
    while cap.isOpened():
        imagedump=[]
        ret,frame=cap.read()
        for i in range(10):
            ret,frame=cap.read()
            if frame is not None:
                image = imutils.resize(frame,width=700,height=600)
                frame=cv2.resize(frame, (227,227), interpolation = cv2.INTER_AREA)
                gray=0.2989*frame[:,:,0]+0.5870*frame[:,:,1]+0.1140*frame[:,:,2]
                gray=(gray-gray.mean())/gray.std()
                gray=np.clip(gray,0,1)
                imagedump.append(gray)
        imagedump=np.array(imagedump)
        imagedump.resize(227,227,10)
        imagedump=np.expand_dims(imagedump,axis=0)
        imagedump=np.expand_dims(imagedump,axis=4)
        output=content_model.predict(imagedump)
        loss=meanLoss(imagedump,output)
        if frame is not None:
            if frame.any()==None:
                print("none")
        else:
            break
        if cv2.waitKey(10) & 0xFF==ord('q'):
            break
        print(str(frame)+" "+str(loss))
        if loss>0.00068:
            print('Inappropriate Content')
            cv2.putText(image,"Inappropriate Content",(100,80),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,255),4)
        else:
            cv2.putText(image,"Safe Content",(100,80),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,255,255),4)
        cv2.imshow("video",image)
    cap.release()
    cv2.destroyAllWindows()
    


def graph():
    df = pd.DataFrame([['EfficientNet-BiLSTM','Precision',precision[0]],['EfficientNet-BiLSTM','Recall',recall[0]],['EfficientNet-BiLSTM','F1 Score',fscore[0]],['EfficientNet-BiLSTM','Accuracy',accuracy[0]],
                       ['EfficientNet-SVM','Precision',precision[1]],['EfficientNet-SVM','Recall',recall[1]],['EfficientNet-SVM','F1 Score',fscore[1]],['EfficientNet-SVM','Accuracy',accuracy[1]],
                       
                      ],columns=['Parameters','Algorithms','Value'])
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar')
    plt.show()

    
def close():
    main.destroy()

font = ('times', 14, 'bold')
title = Label(main, text='A Deep Learning-Based Approach for Inappropriate Content Detection and Classification of YouTube Videos')
title.config(bg='DarkGoldenrod1', fg='black')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=5,y=5)

font1 = ('times', 13, 'bold')

uploadButton = Button(main, text="Upload Youtube Normal & Inappropriate Content Dataset", command=uploadDataset)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=560,y=100)

preprocessButton = Button(main, text="Dataset Preprocessing", command=preprocessDataset)
preprocessButton.place(x=50,y=150)
preprocessButton.config(font=font1)

svmButton = Button(main, text="Generate & Load EfficientNet-SVM Model", command=runExistingSVM)
svmButton.place(x=50,y=250)
svmButton.config(font=font1)

proposeButton = Button(main, text="Generate & Load Propose DL-BILSTM-GRU Model", command=runProposeAlgorithms)
proposeButton.place(x=50,y=200)
proposeButton.config(font=font1)

graphButton = Button(main, text="Comparison Graph", command=graph)
graphButton.place(x=50,y=300)
graphButton.config(font=font1)

predictButton = Button(main, text="Inappropriate Content Prediction from Test Video", command=predict)
predictButton.place(x=50,y=350)
predictButton.config(font=font1)

exitButton = Button(main, text="Exit", command=close)
exitButton.place(x=50,y=400)
exitButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=25,width=90)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=500,y=150)
text.config(font=font1)


main.config(bg='LightSteelBlue1')
main.mainloop()
