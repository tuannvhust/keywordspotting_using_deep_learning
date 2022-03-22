from distutils.spawn import spawn
import math
from re import S
import re
import wave
import torch.nn as nn
from tensorflow.keras.models import load_model
import utils , nnet
from queue import Queue
import numpy as np
import pyaudio 
import tqdm 
import torchaudio
from utils import args
import os 
import torch
from scipy import spatial
from numpy import dot
from numpy.linalg import norm

class Stream_Prediction(nn.Module):
    def __init__(self,args):
        super().__init__()
        #load model
        self.args = args
        self.sampling_rate = 16000
        self.chunk_duration = 0.25
        self.window_duration = 1.0
        self.chunk_samples = int(self.sampling_rate*self.chunk_duration)
        self.window_samples = int(self.sampling_rate*self.window_duration)
        self.queue = Queue()
        self.data = np.zeros(self.window_samples,dtype='int16')



    def load_models(self):
        path = utils.get_ckpt_path(self.args)
        if torch.cuda.is_available():
            checkpoints = torch.load(path)
        else:
            checkpoints = torch.load(path,map_location=torch.device('cpu'))
        model = nnet.get_model(self.args)
        model.load_state_dict(checkpoints['model_state_dict'])
        return model 

    def start_stream(self):
        print("FIRST : Say your keyword {} times".format(3))
        self.record()
        #print(os.listdir("record"))
        self.X_demo = []
        self.Y_demo = []
        model = self.load_models()
        self.path = os.listdir("record")
        for line in self.path:
            file = os.path.join("record",line)
            audio,_ = torchaudio.load(file)
            #print("audio.shape",audio.shape)
            pad_audio = utils.padding(audio)
            #print("pad_audio.shape",pad_audio.shape)
            length = torch.tensor([audio.shape[1]/pad_audio.shape[1]])
            x = self.visualize_step(model=model,x=pad_audio,lens=length)
            self.X_demo.append(x)
            self.Y_demo.append(x)
        self.X_demo = np.concatenate(self.X_demo)
        #self.Y_demo = np.concatenate(self.Y_demo)
        #print("self.X_demo",self.X_demo.shape)
        #print("self.Y_demo",self.Y_demo)
        lower_threshold,higher_threshold = self.compute_threshold(self.X_demo)
        #print("lower_threshold",lower_threshold)
        #print("higher_threshold",higher_threshold)




        filename = "record/now.wav"
        chunk = 1000 # Record in chunks of 1024 samples
        sample_format = pyaudio.paInt16 # 16 bits per sample
        channels = 1 
        fs = 16000 # Record at 44100 samples per second 
        seconds = 2
        p = pyaudio.PyAudio() # Create an interface to PortAudio

        print('RECORDING QUERY.....')
        input("press ENTER to record")
            # p.going() = True

        stream = p.open(format = sample_format,
                            channels =channels,
                            rate = fs,
                            frames_per_buffer = chunk,
                            input = True)
                            # input_device_index = 1)
        frames = [] #initialize array to store frames 

            # Store data in chunks for 3 seconds 
        for i in range(0, int(fs / chunk * seconds)) : 
            data = stream.read(chunk)
            frames.append(data)
            
            # Stop and close the stream 
        stream.stop_stream()
        stream.close()
            #terminate the PortAudio interface 
        p.terminate()

        print('FINISHED')

            # Save record dato as a wave file 
        wf = wave.open(filename,'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(sample_format))
        wf.setframerate(fs)
        wf.writeframes(b''.join(frames))
        wf.close()

        audio1,_ = torchaudio.load(filename)
            #print("audio.shape",audio.shape)
        pad_audio1 = utils.padding(audio)
            #print("pad_audio.shape",pad_audio.shape)
        length1 = torch.tensor([audio1.shape[1]/pad_audio1.shape[1]])
        x_query = self.visualize_step(model=model,x=pad_audio1,lens=length1)
        total_distance = []

        for i in range(3):
            distance = 1 - spatial.distance.cosine(np.abs(x_query),np.abs(self.X_demo[i]))
            total_distance.append(distance)
        
        distance = np.sum(total_distance)/3
        
        #print("distance",distance)
        if  distance > lower_threshold and distance < higher_threshold:
            print("Its a keyword")
        else:
            print("It's not a keyword")

        #self.X_demo.append(x)
            #self.Y_demo.append(y)
        #self.X_demo = np.concatenate(self.X_demo)
        #self.Y_demo = np.concatenate(self.Y_demo)
        #print("self.X_demo",self.X_demo.shape)
    


    
    def record(self) : 
        chunk = 1000 # Record in chunks of 1024 samples
        sample_format = pyaudio.paInt16 # 16 bits per sample
        channels = 1 
        fs = 16000 # Record at 44100 samples per second 
        seconds = 2
        list_dir = ["record/example.wav","record/example1.wav","record/example2.wav",]
        for filename in list_dir:


            p = pyaudio.PyAudio() # Create an interface to PortAudio

            print('RECORDING.....')
            input("press ENTER to record")
            # p.going() = True

            stream = p.open(format = sample_format,
                            channels =channels,
                            rate = fs,
                            frames_per_buffer = chunk,
                            input = True)
                            # input_device_index = 1)
            frames = [] #initialize array to store frames 

            # Store data in chunks for 3 seconds 
            for i in range(0, int(fs / chunk * seconds)) : 
                data = stream.read(chunk)
                frames.append(data)
            
            # Stop and close the stream 
            stream.stop_stream()
            stream.close()
            #terminate the PortAudio interface 
            p.terminate()

            print('FINISHED')

            # Save record dato as a wave file 
            wf = wave.open(filename,'wb')
            wf.setnchannels(channels)
            wf.setsampwidth(p.get_sample_size(sample_format))
            wf.setframerate(fs)
            wf.writeframes(b''.join(frames))
            wf.close()
    # def create_database(self,path):
    #     for line in path:
    #         file = os.path.join("record",line)
    #         audio,_ = torchaudio.load(file)
    #         #print("audio.shape",audio.shape)
    #         pad_audio = utils.padding(audio)
    #         print("pad_audio.shape",pad_audio.shape)
    #         length = torch.tensor([audio.shape[1]/pad_audio.shape[1]])
    #             #print("pad_audio.shape",pad_audio.shape)
    #             #print("length",length.shape)
    #             #print("lens length",len(length))
    #         squeezed_pad_audio = pad_audio.squeeze(0)
    #             #print("pad_audio.shape",padded_audio.shape)
    #             #print("length.shape",length)
    #         features = utils.MelSpectrogram(self.args).transform(pad_audio,length)
    #             #print("features shape",features.shape)
    #             #model = self.load_models()
    #             #output = model(features)
    #             #print("output",output)
            


    def visualize_step(self,model,x,lens):
        model = model.embedding 
        model = model.eval()
        with torch.no_grad():
            #x,lens,y = batch 
            x = x.to(self.args.device)
            #y = y.to(self.args.device)
            lens = lens.to(self.args.device)
            x = model(utils.MelSpectrogram(self.args).transform(x,lens))
            x = x.cpu().detach().numpy()
            #y = y.cpu().detach().numpy()
        return x
    
    def compute_threshold(self,X_demo):
        dataset1 = np.abs(X_demo[1])
        dataset2 = np.abs(X_demo[2])
        dataset3 = np.abs(X_demo[0])
        result1 = 1 - spatial.distance.cosine(dataset1,dataset2,dataset3)
        result2 = 1 - spatial.distance.cosine(dataset2,dataset3,dataset1)
        result3 = 1 - spatial.distance.cosine(dataset3,dataset1,dataset2)
        phi = (result1 + result2 + result3)/3
        phi2 = math.sqrt((math.pow((result1-phi),2)+math.pow((result2-phi),2)+math.pow((result3-phi),2))/3)
        print("phi2",phi2)
        lower_boundary = phi - 1.96*phi2/math.sqrt(3)
        higher_boundary = phi + 1.96*phi2/math.sqrt(3)
        return lower_boundary,higher_boundary
























































def demo(args):
    demoer = Stream_Prediction(args)
    demoer.start_stream()