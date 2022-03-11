import os
import soundfile
import numpy as np


def prepare_data(args):

    #Check if data is dowloaded or not
    if not os.path.exists(os.path.join(args.data_dir,'speech_commands_v0.02.tar.gz')):
        print("downloading SGC")
        os.system(f'wget -P {args.data_dir} http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz')
    else:
        print("skipping downloading the speech commands")
    
    #Checking if rir-noises is downloaded or not
    if not os.path.exists(os.path.join(args.data_dir,"rirs_noises.zip")):
        print('downloading rir_noise.zip')
        os.system(f"wget -P {args.data_dir} https://www.openslr.org/resources/28/rirs_noises.zip")
    else:
        print("skipping downloading rir_noises")


    #Extracting data
    if args.extract_data:
        os.system(f'tar -xf {os.path.join(args.data_dir, "speech_commands_v0.02.tar.gz")} -C {args.data_dir}')
    else:
        print("skipping extracting data")
    if args.extract_rir_noise:
        os.system(f'unzip -n -q {os.path.join(args.data_dir, "rirs_noises.zip")} -d {args.data_dir}')
    else:
        print("skipping extracting rir_noise")

    #get validation set
    with open(os.path.join(args.data_dir,'validation_list.txt'),'r') as validation_file:
        validation_set = [line.strip('\n') for line in validation_file]
    #get testing_test
    with open(os.path.join(args.data_dir,'testing_list.txt'),'r') as testing_file:
        testing_set = [line.strip('\n') for line in testing_file]
    
    vocab = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go',
               'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
               'bed', 'bird', 'cat', 'dog', 'happy', 'house', 'marvin', 'sheila', 'tree', 'wow',
               'backward', 'forward', 'follow', 'learn', 'visual']
    
    data_set = []
    for word in vocab:
        temp_data = [os.path.join(word,file) for file in os.listdir(os.path.join(args.data_dir,word))]
        data_set = data_set + temp_data
    
    #training_set = data_set - validation_set - testing_set

    training_set = set(data_set).symmetric_difference(set(validation_set)).symmetric_difference(set(testing_set))
    training_set = sorted(list(training_set))

    #get training_set
    open(os.path.join(args.data_dir,"training_list.txt"), 'w')
    with open(os.path.join(args.data_dir,'training_list.txt'), 'a') as fin:
        [fin.write(line + '\n') for line in training_set]
    
    #get background data
    background_list = os.listdir(os.path.join(args.data_dir,"_background_noise_"))
    background_set = [file for file in background_list if file.endswith(".wav")]

    #Process background noise to create background data
    for file in background_set:
        background_folder = os.path.join(args.data_dir,"_background_noise_")
        sound_file = os.path.join(background_folder,file)
        os.makedirs(sound_file.replace(".wav",""),exist_ok= True)
        data,sampling_rate = soundfile.read(sound_file)
        #data là dữ liệu âm thanh, ví dụ sound_file có độ dài là 96 giây, sampling_rate là 16000, thì data có len là 96 x 16000
        #sampling_rate = 16000
        for i in range(600):
            offset = np.random.randint(len(data)-sampling_rate +1)
            sub_data = data[offset:offset+sampling_rate]
            #tưởng tượng nó như 1 miếng trượt, nếu offset > len(data) - sampling_rate +1 ,thi lúc đó sub_data có dữ liệu nằm ngoài khoảng data gốc
            #tạo ra các file âm thanh sub_data
            soundfile.write(os.path.join(args.data_dir,"_background_noise_",file.replace(".wav",""),file.replace(".wav","")+"_"+str(i)+".wav"),sub_data,sampling_rate)
            #lệnh này vừa tạo ra directory vừa tạo file âm thanh, trong đó, phần tử cuối  cùng của os.path.join là tên của file

    #write background data to background_list.txt
    open(os.path.join(args.data_dir,"background_list.txt"),'w')
    for folder,sub_folder,file in os.walk(os.path.join(args.data_dir,"_background_noise_")):
        #folder là _background_noise_
        #sub_folder là doing_the_dishes, dude_miaowing,....
        #file là : doing_the_dishes.wav, dude_miaowing.wav,....
        for a in sub_folder:
            files = os.listdir(os.path.join(folder,a))
            with open(os.path.join(args.data_dir,"background_list.txt"),'a') as ap:
                [ap.write(os.path.join('_background_noise_',a,line)+ "\n") for line in files]
        break
    
    #process rir_noise 
    rir_folder = os.path.join(args.data_dir,"RIRS_NOISES")
    noise_list = os.path.join(rir_folder,"pointsource_noises","noise_list")
    rir_list = os.path.join(rir_folder,"real_rirs_isotropic_noises","rir_list")

    #remove old noise and old reverb
    os.system(f'rm -rf {os.path.join(rir_folder, "noise")} {os.path.join(rir_folder, "reverb")}')

    #create new noise and reverb folder
    os.makedirs(os.path.join(rir_folder,"noise"))
    os.makedirs(os.path.join(rir_folder,"reverb"))

    #built noise_folder

    with open(noise_list,"r") as noise:
        for line in noise:
            noise_sound = os.path.join(args.data_dir,line.split()[-1])
            data,sampling_rate = soundfile.read(noise_sound)
            try:
                if data.shape[1]:
                    data = data[:,0] #make sure data has one channel
            except:
                pass 
            if len(data) > sampling_rate:
                for i in range(int(len(data)/sampling_rate)):
                    start = int(i*sampling_rate)
                    stop = int(start + sampling_rate)
                    sub_data = data[start:stop]
                    soundfile.write(os.path.join(rir_folder,"noise",noise_sound.split('/')[-1].replace(".wav","")+"_"+str(i)+".wav"),sub_data,sampling_rate)
            else:
                soundfile.write(os.path.join(rir_folder,"noise",noise_sound.split('/')[-1]),data,sampling_rate)


    #built reverb folder
    with open(rir_list,"r") as rir:
        for line in rir:
            rir_sound = os.path.join(args.data_dir,line.split()[-1])
            data,sampling_rate = soundfile.read(noise_sound)
            try:
                if data.shape[1]:
                    data = data[:,0] #make sure data has one channel
            except:
                pass 
            soundfile.write(os.path.join(rir_folder,"reverb",rir_sound.split('/')[-1]),data,sampling_rate)

        


