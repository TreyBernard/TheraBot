import pandas as pd 
import numpy as np 
import tensorflow as tf
import os,time,librosa,warnings,glob
import sounddevice as sd
import regex as re
from sklearn.metrics import confusion_matrix,classification_report
import librosa.display
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.layers import Dense,Input,Add,Flatten,Dropout,Activation,AveragePooling1D,Conv1D
from keras.models import Model,Sequential,load_model
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler,EarlyStopping,ReduceLROnPlateau,ModelCheckpoint
#from google.colab.output import eval_js
from base64 import b64decode
#from IPython.display import Audio,HTML
from scipy.io.wavfile import read as wav_read
import io
#import ffmpeg
warnings.filterwarnings("ignore")

#decorator function for calculating the total time reqired to execute various function
def calc_time(func):
  def inner(*args, **kwargs):
    st = time.time()
    result = func(*args,**kwargs)
    end = time.time()-st
    print("Total time required: {:.3f} ms".format(end * 1000))
    return result
  return inner

#function for getting ravdess dataset details and labeling
def ravdess_data():
  #directory of the audio dataset
  ravdess = "./Audiofiles/RAVDESS/"
  #label ravdess data
  emotion_ravdess = {'01':'neutral','02':'calm','03':'happy','04':'sad','05':'angry','06':'fearful','07':'disgust','08':'surprised'}
  #list to store ravdess emotion
  ravdess_emotion = []
  #list to store ravdess audio path
  ravdess_path = []
  #get subfolders from the path
  ravdess_folder = os.listdir(ravdess)
  for i in ravdess_folder:
    inner_files = os.listdir(ravdess+i+'/')
    for j in inner_files:
      #get the split part which contains the emotion information then append it into lists
      #print(f"Processing filename: {j}")
      emotion = j.split('-')[2]
      ravdess_path.append(ravdess+i+'/'+j)
      ravdess_emotion.append(emotion_ravdess[emotion])

  #convert to dataframe
  df_ravdess = pd.DataFrame([ravdess_path,ravdess_emotion]).T
  df_ravdess.columns = ["AudioPath","Label"]
  print("length of ravdess dataset",len(df_ravdess))

  return df_ravdess

#function for getting crema dataset details and labeling
def crema_data():
  #directory of the audio dataset
  crema = "./Audiofiles/CREMA-D/"
  #label ravdess data
  emotion_crema = {'SAD':'sad','ANG':'angry','DIS':'disgust','FEA':'fear','HAP':'happy','NEU':'neutral'}
  #list to store crema emotion
  crema_emotion = []
  #list to store crema audio path
  crema_path = []
  #get crema files in directory
  crema_files = os.listdir(crema)
  for i in crema_files:
    emotion = i.split('_')[2]
    crema_emotion.append(emotion_crema[emotion])
    crema_path.append(crema+i)

  #convert to dataframe
  df_crema = pd.DataFrame([crema_path,crema_emotion]).T
  df_crema.columns = ["AudioPath","Label"]
  print("length of crema dataset",len(df_crema))

  return df_crema

#function for getting tess dataset and labeling
def tess_data():
  #directory of the audio dataset
  tess = "./Audiofiles/TESS/"
  tess_emotion = []
  tess_path = []
  tess_folder = os.listdir(tess)
  for i in tess_folder:
    #print(f"Processing filename: {i}")
    emotion = i.split('_',1)[1]
    inner_files = os.listdir(tess+i+'/')
    for j in inner_files:
      tess_path.append(tess+i+'/'+j)
      tess_emotion.append(emotion)

  #convert to dataframe
  df_tess = pd.DataFrame([tess_path,tess_emotion]).T
  df_tess.columns = ["AudioPath","Label"]
  print("length of tess dataset",len(df_tess))

  return df_tess

#function to get savee dataset and labeling
def saveee_data():
  #directory of the audio dataset
  savee = "./Audiofiles/SAVEE/"
  emotion_savee = {'a':'anger','d':'disgust','f':'fear','h':'happiness','n':'neutral','sa':'sadness','su':'surprise'}
  savee_emotion = []
  savee_path = []
  savee_files = os.listdir(savee)
  for i in savee_files:
    emotion = i.split('_')[1]
    emotion = re.match(r"([a-z]+)([0-9]+)",emotion)[1]
    savee_emotion.append(emotion_savee[emotion])
    savee_path.append(savee+i)

  #convert to dataframe
  df_savee = pd.DataFrame([savee_path,savee_emotion]).T
  df_savee.columns = ["AudioPath","Label"]
  print("length of savee dataset",len(df_savee))

  return df_savee

@calc_time
def fetch_data():
  #get ravdess data
  df_ravdess = ravdess_data()
  #get crema data
  df_crema = crema_data()
  #get tess data
  df_tess = tess_data()
  #get savee data
  df_savee = saveee_data()
  #combine all four dataset into one single dataset and create a dataframe 
  frames = [df_ravdess,df_crema,df_tess,df_savee]
  final_combined = pd.concat(frames)
  final_combined.reset_index(drop=True,inplace=True)
  #save the information of datasets with their path and labels into a csv file
  final_combined.to_csv("./preprocesseddata.csv",index=False,header=True)
  print("Total length of the dataset is {}".format(len(final_combined)))
  return final_combined

#below are four data agumentation functions for noise, stretch, shift, pitch
#function to add noise to audio
def noise(data):
  noise_amp = 0.035*np.random.uniform()*np.amax(data)
  data = data + noise_amp*np.random.normal(size=data.shape[0])
  return data

#fuction to strech audio
def stretch(data, rate=0.8):
  return librosa.effects.time_stretch(data, rate=rate)

#fucntion to shift audio range
def shift(data):
  shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
  return np.roll(data, shift_range)

#function to change pitch
def pitch(data, sampling_rate, pitch_factor=0.7):
  return librosa.effects.pitch_shift(y=data, sr=sampling_rate, n_steps=pitch_factor)

#fuction to extract audio features from the audio files given the information of their path
#path and label information comes from fetch_data fucntion 
#also file preprocesseddata.csv stores the information of paths of audio files their label information
#the print statements are commented these statements were used to see the number of features returned as output
def extract_features(data,sample_rate):  
  
  #zero crossing rate
  result = np.array([])
  zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
  result = np.hstack((result, zcr)) 
  #print('zcr',result.shape)

  #chroma shift
  stft = np.abs(librosa.stft(data))
  chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
  result = np.hstack((result, chroma_stft))
  #print('chroma',result.shape)
  
  #mfcc
  mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
  result = np.hstack((result, mfcc))
  #print('mfcc',result.shape)
  
  #rmse
  rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
  result = np.hstack((result, rms)) 
  #print('rmse',result.shape)
  
  #melspectogram
  mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
  result = np.hstack((result, mel)) 
  #print('mel',result.shape)    

  #rollof 
  rollof = np.mean(librosa.feature.spectral_rolloff(y=data, sr=sample_rate).T, axis=0)
  result = np.hstack((result, rollof))
  #print('rollof',result.shape) 

  #centroids 
  centroid = np.mean(librosa.feature.spectral_centroid(y=data, sr=sample_rate).T, axis=0)
  result = np.hstack((result, centroid))
  #print('centroids',result.shape)

  #contrast
  contrast = np.mean(librosa.feature.spectral_contrast(y=data, sr=sample_rate).T, axis=0)
  result = np.hstack((result, contrast))
  #print('contrast',result.shape)

  #bandwidth
  bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=data, sr=sample_rate).T, axis=0)
  result = np.hstack((result, bandwidth))
  #print('bandwidth',result.shape)

  #tonnetz
  tonnetz = np.mean(librosa.feature.tonnetz(y=data, sr=sample_rate).T, axis=0)
  result = np.hstack((result, tonnetz))
  #print('tonnetz',result.shape) 

  return result

#function is used to get all augmented plus original features for given audio file
def get_features(path):
  #set the duration and offset
  #librosa.load takes audio file converts to array and returns array of audio file with its sampling rate
  print(path)
  data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)
    
  #get audio features without augmentation
  res1 = extract_features(data,sample_rate)
  result = np.array(res1)
    
  #get audio features with noise
  noise_data = noise(data)
  res2 = extract_features(noise_data,sample_rate)
  result = np.vstack((result, res2))
    
  #get audio features with stretching and pitching
  new_data = stretch(data)
  data_stretch_pitch = pitch(new_data, sample_rate)
  res3 = extract_features(data_stretch_pitch,sample_rate)
  result = np.vstack((result, res3))
    
  return result

#fucntion one by one takes aduio files from the path extracts features 
#extracted audio features along with their label information are stored in a csv file 
@calc_time
def Audio_features_extract():
  #this function is used to fetch the data from all the four datasets
  df = fetch_data()
  #count is used to keep a check of number of files processed
  count = 0
  #list to store audio features and their label information
  X_data, Y_label = [], []
  #zip audio path and label information and then iterate over them
  for path, emotion in zip(df["AudioPath"], df["Label"]):
    print("Number of files processed ",count)
    #get the features 
    #for one audio file it get three sets of features 
    #original features, features with noise(agumentation) and feature with change in stretch and pitch
    #so one audio file generates three output and the label is same for all the outputs
    feature = get_features(path)
    for ele in feature:
      X_data.append(ele)
      Y_label.append(emotion)
    count+=1
  #create a dataframe of aduio features
  Features = pd.DataFrame(X_data)
  #add label information 
  Features['Label'] = Y_label
  #store the extracted features in a csv file
  Features.to_csv('./Audiofiles/Audio_features_All_pr.csv',index=False)

#this is just one time process so call this function once only to get the features
#once the features are extracted then these features are used for making model
#Audio_features_extract()

#function to plot loss and accuracy curves on training set
def plotgraph(history):
  plt.figure(figsize=[8,6])
  plt.plot(history.history['loss'],'firebrick',linewidth=3.0)
  plt.plot(history.history['accuracy'],'turquoise',linewidth=3.0)
  plt.legend(['Training loss','Training Accuracy'],fontsize=18)
  plt.xlabel('Epochs ',fontsize=16)
  plt.ylabel('Loss and Accuracy',fontsize=16)
  plt.title('Loss Curves and Accuracy Curves',fontsize=16)

#function carries out additional preprocessing on data
#this function includes the EDA carried out on dataset 
#the selected emotions are kept and others are dropped 
#renaming of emotions is done to maintian uniformity
def additional_preprocess(filepath):
  #read the csv file of extrated features
  df = pd.read_csv(filepath)
  print("\nlabels or emotions present in dataset\n",df["Label"].unique())
  #replace label names with name common for each emotion 
  #this is done to maintain uniformity of label names
  df["Label"] = df["Label"].str.replace("sadness", "sad", case = True)
  df["Label"] = df["Label"].str.replace("happiness", "happy", case = True)
  df["Label"] = df["Label"].str.replace("Fear", "fear", case = True)
  df["Label"] = df["Label"].str.replace("Sad", "sad", case = True)
  df["Label"] = df["Label"].str.replace("Pleasant_surprise", "surprise", case = True)
  df["Label"] = df["Label"].str.replace("pleasant_surprised", "surprise", case = True)
  df["Label"] = df["Label"].str.replace("surprised", "surprise", case = True)
  df["Label"] = df["Label"].str.replace("fearful", "fear", case = True)
  df["Label"] = df["Label"].str.replace("anger", "angry", case = True)
  #drop labels surprized and clam
  #these label dosent contain sufficent amount of data and can lead to missclassification
  print("\nUnique count of labels or emotions\n",df["Label"].value_counts())
  #drop labels or emotions which can lead to misclassifications
  df.drop((np.where(df['Label'].isin(["surprise","calm"]))[0]), inplace = True)
  print("\nUnique count of labels or emotions after dropping selected labels\n",df["Label"].value_counts())
  print("\nlength of the total data is {}".format(len(df)))
  return df

#this fucntion is used to get audio features perform one hot encoding and split datasets into train, test and validation
@calc_time
def audio_features_final():
  df = additional_preprocess("./Audiofiles/Audio_features_All_pr.csv")
  #get all the aduio features as numpy array from the dataframe 
  #last column is label so last column is not fetched only 0to:-1
  data=df[df.columns[0:-1]].values
  #perform one hot encoding on labels
  encoder = OneHotEncoder()
  #fetch the last column of labels and perform one hot encoding on them
  label=df["Label"].values
  label = encoder.fit_transform(np.array(label).reshape(-1,1)).toarray()
  #min max scaler is used to normalize the data
  scaler = MinMaxScaler()
  data=scaler.fit_transform(data)
  #split the dataframe into train and test 80% train, 10% validation and 10% test datasets
  x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.20, random_state=42,shuffle=True)
  x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.50, random_state=42, shuffle=True)
  print("\nlength of train data is {}, test data is {} and validation set is {}".format(len(x_train),len(x_test),len(x_val))) 
  print("\n shape of train features and label is {}".format(x_train.shape, y_train.shape))
  print("\n shape of test features and label is {}".format(x_test.shape, y_test.shape))
  print("\n shape of validation features and label is {}".format(x_val.shape,y_val.shape))
  return x_train, x_test, y_train, y_test, x_val, y_val, encoder

#fucntion trains the model and saves the best model at the checkpoint
@calc_time
def emotion_recognition_model(x_train,y_train,x_val,y_val):
  #reduce the laerning rate if plateau is encountered
  reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=0.001)
  #early stopping method is used to montior the loss if there are no significant reductions in loss then halt the training
  es = EarlyStopping(monitor='loss', patience=20)
  #checkpoint to save the best model with highest validation accuracy
  filepath = "./emotion_model.h5"
  checkpoint = ModelCheckpoint("./emotion_model.h5", monitor="val_accuracy", verbose=1, save_best_only=True, mode="max")
  #create a combined list of reduce learning rate, early stopping and checkpoint
  callbacks_list = [reduce_lr,es,checkpoint]
  def residual_block(x, filters, conv_num=3, activation="relu"):
    #fucntion is used to create residual blocks and add residual blocks
    s = Conv1D(filters, 1, padding="same")(x)
    for i in range(conv_num - 1):
      x = Conv1D(filters, 3, padding="same")(x)
      x = Activation(activation)(x)
    x = Conv1D(filters, 3, padding="same")(x)
    x = Add()([x, s])
    x = Activation(activation)(x)
    return x
  
  #fucntion to build the model 
  def build_model():
    inputs =  Input(shape=(x_train.shape[1],1))
    x = Dense(256, activation="relu")(inputs)
    x = residual_block(x, 16, 2)
    x = residual_block(x, 32, 2)
    x = residual_block(x, 32, 2)
    x = residual_block(x, 64, 3)
    x = residual_block(x, 64, 3)
    x = residual_block(x, 128, 3)
    x = residual_block(x, 128, 3)
    #perform the average pooling after last residual block 
    x = AveragePooling1D(pool_size=3, strides=3)(x)
    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)
    x = Dense(128, activation="relu")(x)
    outputs = Dense(6, activation="softmax", name="output")(x)
    return Model(inputs=inputs, outputs=outputs)

  res_model = build_model() 
  #display the summary of the model
  res_model.summary() 
  #complie the model
  res_model.compile(loss='categorical_crossentropy',optimizer = Adam(learning_rate=1e-4, decay=1e-4 / 50) , metrics=['accuracy'])
  history = res_model.fit(np.expand_dims(x_train,-1),y_train,
                validation_data=(np.expand_dims(x_val, -1), y_val), 
                epochs=500,
                batch_size=32,
                shuffle=True,
                #workers=50,
                verbose=1,
                #use_multiprocessing=True,
                callbacks = callbacks_list)
  
  res_model.save('./trained_model/emotion_model.h5')  # This is the change made
  
  #plot loss and accuracy curves
  plotgraph(history)

#   # Function to record audio from microphone
def get_audio(duration=30, sr=22050):  # Default: 3 seconds, 22.05 kHz
    print("Recording... Speak now!")
    
    # Record audio
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()  # Wait for the recording to finish
    
    print("Recording finished.")
    
    # Convert recorded data to 1D array
    audio = np.squeeze(audio)
    
    return audio, sr

# Function to extract features from recorded audio (assuming extract_features, noise, stretch, and pitch exist)
def get_features_recorded(data, sr):
    res1 = extract_features(data, sr)
    result = np.array(res1)

    noise_data = noise(data)
    res2 = extract_features(noise_data, sr)
    result = np.vstack((result, res2))

    new_data = stretch(data)
    data_stretch_pitch = pitch(new_data, sr)
    res3 = extract_features(data_stretch_pitch, sr)
    result = np.vstack((result, res3))

    return result

# Function to evaluate performance of model on recorded audio
def test_realtime(encoder):
    # Load the model
    res_model = load_model("./trained_model/emotion_model.h5")  # Update path if needed

    # Record the audio
    audio, sr = get_audio()
    
    # Save audio in a file
    files = []
    os.makedirs("realtimetested", exist_ok=True)
    for file in glob.glob("realtimetested/*.npy"):
        files.append(file)

    np.save(f"realtimetested/audiorec{len(files)}.npy", audio)

    # Plot the recorded audio
    plt.figure(figsize=(5,5))
    plt.plot(audio)
    plt.show()
    
    plt.savefig(f"realtimetested/audiorec{len(files)}.png")

    # Convert int to float
    audio = audio.astype('float')

    # Get audio features
    feature = get_features_recorded(audio, sr)

    # Apply MinMax scaling
    scaler = MinMaxScaler()
    feature = scaler.fit_transform(feature)

    # Get raw probability scores
    probabilities = res_model.predict(feature)

    # Print all probability scores for debugging
    print("\nProbability Scores for Each Emotion:")
    for i, probs in enumerate(probabilities):
        print(f"Sample {i+1}: {probs}")

    # Keep the one-hot encoded format for inverse_transform
    label_predicted = encoder.inverse_transform(probabilities)  # Expecting (N, 6)

    # Get confidence scores
    confidence = np.max(probabilities, axis=1)  

    print("\nPredicted Emotion: {}".format(label_predicted[0]))
    print("Confidence Score: {:.2f}".format(confidence[0]))

    # Save results in a CSV file
    df = pd.DataFrame(index=range(0,3), columns=['path','label','audio'])
    for i in range(0,3):
        df["path"][i] = f"realtimetested/audiorec{len(files)}.npy"
        df["label"][i] = label_predicted[i]
        df["audio"][i] = feature[i]

    df.to_csv("realtimetested/real_time_predicted_audio_features.csv", mode='a', index=False)

#function to evaluate the model performance once the best model is saved
#it loads the best model and then evaluates the performance
@calc_time
def evaluate_model(x_train, x_test, y_train, y_test, x_val, y_val):
  #load the best model
  model = load_model("./trained_model/emotion_model.h5")
  #evaluate training accuracy 
  _,train_acc = model.evaluate(np.expand_dims(x_train,-1),y_train, batch_size=1)
  #evaluate testing acuracy
  _,test_acc = model.evaluate(np.expand_dims(x_test,-1),y_test, batch_size=1)
  #evaluate validation accuracy
  _,val_acc = model.evaluate(np.expand_dims(x_val,-1),y_val, batch_size=1)
  print("\n**********************************************")
  print("\n Training accuracy of the model is {}".format(np.round(float(train_acc*100),2)))
  print("\n Testing accuracy of the model is {}".format(np.round(float(test_acc*100),2)))
  print("\n Validation accuracy of the model is {}".format(np.round(float(val_acc*100),2)))
  print("**********************************************")
  #predict the outcome of the model
  y_pred = model.predict(x_test)
  y_pred=np.argmax(y_pred, axis=1)
  y_test=np.argmax(y_test, axis=1)
  #View the classification report for test data and predictions
  print("\nClassification report for Emotion Recognition")
  print(classification_report(y_test, y_pred))  
  #View confusion matrix for test data and predictions
  print("\nConfusion matrix for Emotion Recognition")
  print(confusion_matrix(y_test, y_pred))
  print("*****************************")

  #function calculates the above code in sequence
#it runs the model and also evaluates the model performance
@calc_time
def main():
  print("Emotion Recognition Model")
  #get train,test data and labels 
  x_train, x_test, y_train, y_test, x_val, y_val, encoder = audio_features_final()
  #test_audio_file("./Audiofiles/TESS/OAF_angry/OAF_chair_angry.wav", encoder)
  test_realtime(encoder)
  #call the emotion recognition model
  #emotion_recognition_model(x_train,y_train,x_val,y_val)
  #evaluate the model performance
  #evaluate_model(x_train, x_test, y_train, y_test, x_val, y_val)
  #print("\none hot encoding array\n",np.unique(y_train,axis=0))
  #print("\nOne hot encoding mapping to actual label\n",encoder.inverse_transform(np.unique(y_train,axis=0)))

if __name__ == "__main__":
    main()


#x_train, x_test, y_train, y_test, x_val, y_val, encoder = audio_features_final()
#mapping of the one hot ecoding with respect to their labels
#print("\none hot encoding array\n",np.unique(y_train,axis=0))
#print("\nOne hot encoding mapping to actual label\n",encoder.inverse_transform(np.unique(y_train,axis=0)))

#test_realtime(encoder)