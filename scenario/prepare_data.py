import numpy as np
import soundfile
import os

def prepare_data(args):
    
    # download data
    if not os.path.exists(os.path.join(args.data_dir, 'speech_commands_v0.02.tar.gz')):
        print('Downloading GSC dataset ...')
        os.system(f'wget -P {args.data_dir} http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz')
    else:
        print('Skipping download GSC dataset ...')
    if not os.path.exists(os.path.join(args.data_dir, 'rirs_noises.zip')):
        print('Downloading RIR and Noise dataset ...')
        os.system(f'wget -P {args.data_dir} https://www.openslr.org/resources/28/rirs_noises.zip')
    else:
        print('Skipping download RIR and Noise dataset ...')
    
    # extract data
    if args.no_extract:
        print('Extracing GSC dataset ...')
        os.system(f'tar -xf {os.path.join(args.data_dir, "speech_commands_v0.02.tar.gz")} -C {args.data_dir}')
        print('Extracting RIR and Noise dataset ...')
        os.system(f'unzip -n -q {os.path.join(args.data_dir, "rirs_noises.zip")} -d {args.data_dir}')
    else:
        print('Skipping extract dataset ...')

    print('Preparing dataset ...')
    # get validation list
    with open(os.path.join(args.data_dir, 'validation_list.txt'), 'r') as validation_file:
        validation_set = [line.strip('\n') for line in validation_file]
    # get test list
    with open(os.path.join(args.data_dir, 'testing_list.txt'), 'r') as testing_file:
        testing_set = [line.strip('\n') for line in testing_file]

    # get data list
    keyword = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go',
               'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
               'bed', 'bird', 'cat', 'dog', 'happy', 'house', 'marvin', 'sheila', 'tree', 'wow',
               'backward', 'forward', 'follow', 'learn', 'visual']
    
    data_set = []
    for word in keyword:
        word_file = [os.path.join(word, file) for file in os.listdir(os.path.join(args.data_dir, word))]
        data_set += word_file
    
    # training list = data list - validation list - test list
    training_set = set(data_set).symmetric_difference(set(validation_set)).symmetric_difference(set(testing_set))
    training_set = sorted(list(training_set))

    # write training list to file
    open(os.path.join(args.data_dir, 'training_list.txt'), 'w')
    with open(os.path.join(args.data_dir, 'training_list.txt'), 'a') as fin:
        [fin.write(path + '\n') for path in training_set]
    
    # processing background noise
    background_list = os.listdir(os.path.join(args.data_dir, '_background_noise_'))
    background_set = [file for file in background_list if file.endswith('.wav')] # 6 files
    for file in background_set:
        folder_path = os.path.join(args.data_dir, '_background_noise_')
        file_path = os.path.join(folder_path, file)
        os.makedirs(file_path.replace('.wav',''), exist_ok=True)
        signal, rate = soundfile.read(file_path)
        for i in range(600): # 6*600=3600 to balancing data
            offset = np.random.randint(0, len(signal)-rate-1)
            sub_signal = signal[offset:offset+rate]
            soundfile.write(os.path.join(folder_path, file.replace('.wav',''), file.replace('.wav','')+'_'+str(i)+'.wav'), sub_signal, rate)
    
    # write background list to file
    open(os.path.join(args.data_dir, 'background_list.txt'), 'w')
    for root, dirs, _ in os.walk(os.path.join(args.data_dir, '_background_noise_')):
        for folder in dirs:
            files = os.listdir(os.path.join(root, folder))
            with open(os.path.join(args.data_dir, 'background_list.txt'), 'a') as fin:
                [fin.write(os.path.join('_background_noise_', folder, file) + '\n') for file in files]
        break

    # processing reverberation and rir noise
    rir_folder = os.path.join(args.data_dir, 'RIRS_NOISES')
    noise_file = os.path.join(rir_folder, 'pointsource_noises', 'noise_list')
    reverb_file = os.path.join(rir_folder, 'real_rirs_isotropic_noises', 'line')
    
    # remove old noise and reverb
    os.system(f'rm -rf {os.path.join(rir_folder, "noise")} {os.path.join(rir_folder, "reverb")}')
    
    # make new folder to save noise and reverb
    os.makedirs(os.path.join(rir_folder, 'noise'))
    os.makedirs(os.path.join(rir_folder, 'reverb'))
    
    # build noise folder
    with open(noise_file, 'r') as file:
        for line in file:
            file_path = os.path.join(args.data_dir, line.split()[-1])
            signal, rate = soundfile.read(file_path)
            try:
                if signal.shape[1]:
                    signal = signal[:,0] # ensure 1 channel
            except:
                pass
            if len(signal) > rate: # wav is too long
                for i in range(int(len(signal)/rate)):
                    start = int(i * rate)
                    stop = int(start + rate)
                    sub_signal = signal[start:stop]
                    soundfile.write(os.path.join(rir_folder, 'noise', file_path.split('/')[-1].replace('.wav','')+'_'+str(i)+'.wav'), sub_signal, rate)
            else:
                soundfile.write(os.path.join(rir_folder, 'noise', file_path.split('/')[-1]), signal, rate)

    # build reverb folder
    with open(reverb_file, 'r') as file:
        for line in file:
            file_path = os.path.join(args.data_dir, line.split()[-1])
            signal, rate = soundfile.read(file_path)
            try:
                if signal.shape[1]:
                    signal = signal[:,0] # ensure 1 channel
            except:
                pass
            soundfile.write(os.path.join(rir_folder, 'reverb', file_path.split('/')[-1]), signal, rate)