

from ast import arg
import json
import os
from random import shuffle
from sys import path



def build_json(args):
    if args.n_keyword == 35:
        vocab = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go',
               'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
               'bed', 'bird', 'cat', 'dog', 'happy', 'house', 'marvin', 'sheila', 'tree', 'wow',
               'backward', 'forward', 'follow', 'learn', 'visual']
    elif args.n_keyword == 12:
        vocab = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go','unknown','silence']
    else:
        raise ValueError("Number o keyword must be 35 or 12")
    #mở các list training, validation, testing,bajckground
    with open(os.path.join(args.data_dir,'training_list.txt'),'r') as training_list:
        training_set = [line.strip('\n') for line in training_list]
    with open(os.path.join(args.data_dir,'validation_list.txt'),'r') as validation_list:
        validation_set = [line.strip('\n') for line in validation_list]
    with open(os.path.join(args.data_dir,'testing_list.txt'),'r') as testing_list:
        testing_set = [line.strip('\n') for line in testing_list] 
    with open(os.path.join(args.data_dir,'background_list.txt'),'r') as background_list:
        background_set = [line.strip('\n') for line in background_list]  

    print(len(training_set) ,len(validation_set), len(testing_set), len(background_set))
    print(f"Building json files for {args.n_keyword} keywords")

    #tạo file json
    ftrain = open(os.path.join(args.data_dir,'train'+'_'+ str(args.n_keyword)+'.json'),'w')
    fval = open(os.path.join(args.data_dir,'validation'+'_'+ str(args.n_keyword)+'.json'),'w')
    ftest = open(os.path.join(args.data_dir,'test'+'_'+ str(args.n_keyword)+'.json'),'w')
    fnoise = open(os.path.join(args.data_dir,'noise.json'),'w')
    freverb = open(os.path.join(args.data_dir,'reverb.json'),'w')

    #tạo nội dung cho file json
    for word in vocab:
        if word == 'unknown':
            unknown_word = ['backward', 'bed', 'bird', 'cat', 'dog',
                            'eight', 'five', 'follow', 'forward', 'four',
                            'happy', 'house', 'learn', 'marvin', 'nine',
                            'one', 'seven', 'sheila', 'six', 'three',
                            'tree', 'two', 'visual', 'wow', 'zero']

            for unk_word in unknown_word:
                count = 0
                for line in training_set:
                    if unk_word + '/' in line:
                        dir = os.path.join(os.path.join(args.data_dir,line))
                        json_dict = {'file':dir,'text':word}
                        json.dump(json_dict,ftrain)
                        ftrain.write('\n')
                        count += 1
                        if count == 120:
                            break
            for unk_word in unknown_word:
                count = 0
                for line in validation_set:
                    if unk_word + '/' in line:
                        dir = os.path.join(os.path.join(args.data_dir,line))
                        json_dict = {'file':dir,'text':word}
                        json.dump(json_dict,fval)
                        fval.write('\n')
                        count += 1
                        if count == 15:
                            break  

            for unk_word in unknown_word:
                count = 0
                for line in testing_set:
                    if unk_word + '/' in line:
                        dir = os.path.join(os.path.join(args.data_dir,line))
                        json_dict = {'file':dir,'text':word}
                        json.dump(json_dict,ftest)
                        ftest.write('\n')
                        count += 1
                        if count == 15:
                            break     

        elif word =='silence':
            shuffle(background_set)
            for part in background_set[:2800]:
                dir = os.path.join(os.path.join(args.data_dir,part))
                json_dict = {'file':dir,'text':word}
                json.dump(json_dict,ftrain)
                ftrain.write('\n')
            for part in background_set[2800:3200]:
                dir = os.path.join(os.path.join(args.data_dir,part))
                json_dict = {'file':dir,'text':word}
                json.dump(json_dict,fval)
                fval.write('\n')
            for part in background_set[3200:3600]:
                dir = os.path.join(os.path.join(args.data_dir,part))
                json_dict={'file':dir,'text': word}
                json.dump(json_dict,ftest)
                ftest.write('\n')
        
        else:
            for part in training_set:
                if word +'/' in part:
                    dir = os.path.join(os.path.join(args.data_dir,part))
                    json_dict = {'file':dir,'text':word}
                    json.dump(json_dict,ftrain)
                    ftrain.write('\n')
            for part in validation_set:
                if word +'/' in part:
                    dir = os.path.join(os.path.join(args.data_dir,part))
                    json_dict = {'file':dir,'text':word}
                    json.dump(json_dict,fval)
                    fval.write('\n')
            for part in testing_set:
                if word +'/' in part:
                    dir = os.path.join(os.path.join(args.data_dir,part))
                    json_dict = {'file':dir,'text':word}
                    json.dump(json_dict,ftest)
                    ftest.write('\n')
    noise_folder = os.path.join(args.data_dir,"RIRS_NOISES",'noise')
    reverb_folder = os.path.join(args.data_dir,"RIRS_NOISES","reverb")
    for part in os.listdir(noise_folder):
        dir = os.path.join(os.path.join(noise_folder,part))
        json_dict = {'file':dir}
        json.dump(json_dict,fnoise)
        fnoise.write('\n')
    for part in os.listdir(reverb_folder):
        dir = os.path.join(os.path.join(reverb_folder,part))
        json_dict = {'file':dir}
        json.dump(json_dict,freverb)
        freverb.write('\n')   
    
    ftrain.close()
    ftest.close()
    fval.close()
    fnoise.close()
    freverb.close()

    return ftrain,fval,ftest,fnoise,freverb
    

            




