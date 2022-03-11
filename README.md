# Keyword Spotting
## 1. Prepare data:
python3 main.py --scenario=prepare_data <br/>
python3 main.py --scenario=build_json --n_keyword=12 <br/>
python3 main.py --scenario=build_json --n_keyword=35 <br/>

## 2. Training:
### 12 keyword:
CUDA_VISIBLE_DEVICES=0 python3 main.py --metric=softmax --batch_size=128 --no_evaluate <br/>
CUDA_VISIBLE_DEVICES=0 python3 main.py --metric=adacos --batch_size=128 --no_evaluate <br/>
CUDA_VISIBLE_DEVICES=0 python3 main.py --metric=arcface --batch_size=128 --no_evaluate <br/>
CUDA_VISIBLE_DEVICES=0 python3 main.py --metric=cosface --batch_size=128 --no_evaluate <br/>
CUDA_VISIBLE_DEVICES=0 python3 main.py --metric=sphereface --batch_size=128 --no_evaluate <br/>

### 35 keyword
CUDA_VISIBLE_DEVICES=0 python3 main.py --metric=softmax --n_keyword=35 --batch_size=128 --no_evaluate <br/>
CUDA_VISIBLE_DEVICES=0 python3 main.py --metric=adacos --n_keyword=35 --batch_size=128 --no_evaluate <br/>
CUDA_VISIBLE_DEVICES=0 python3 main.py --metric=arcface --n_keyword=35 --batch_size=128 --no_evaluate <br/>
CUDA_VISIBLE_DEVICES=0 python3 main.py --metric=cosface --n_keyword=35 --batch_size=128 --no_evaluate <br/>
CUDA_VISIBLE_DEVICES=0 python3 main.py --metric=sphereface --n_keyword=35 --batch_size=128 --no_evaluate <br/>
## 2. Visualize:
### 12 keyword:
CUDA_VISIBLE_DEVICES=0 python3 main.py --scenario='visualize_data' --metric=softmax --batch_size=128 --no_evaluate <br/>
CUDA_VISIBLE_DEVICES=0 python3 main.py --scenario='visualize_data' --metric=adacos --batch_size=128 --no_evaluate <br/>
CUDA_VISIBLE_DEVICES=0 python3 main.py --scenario='visualize_data' --metric=arcface --batch_size=128 --no_evaluate <br/>
CUDA_VISIBLE_DEVICES=0 python3 main.py --scenario='visualize_data' --metric=cosface --batch_size=128 --no_evaluate <br/>
CUDA_VISIBLE_DEVICES=0 python3 main.py --scenario='visualize_data' --metric=sphereface --batch_size=128 --no_evaluate <br/>

### 35 keyword
CUDA_VISIBLE_DEVICES=0 python3 main.py --scenario='visualize_data' --metric=softmax --n_keyword=35 --batch_size=128 --no_evaluate <br/>
CUDA_VISIBLE_DEVICES=0 python3 main.py --scenario='visualize_data' --metric=adacos --n_keyword=35 --batch_size=128 --no_evaluate <br/>
CUDA_VISIBLE_DEVICES=0 python3 main.py --scenario='visualize_data' --metric=arcface --n_keyword=35 --batch_size=128 --no_evaluate <br/>
CUDA_VISIBLE_DEVICES=0 python3 main.py --scenario='visualize_data' --metric=cosface --n_keyword=35 --batch_size=128 --no_evaluate <br/>
CUDA_VISIBLE_DEVICES=0 python3 main.py --scenario='visualize_data' --metric=sphereface --n_keyword=35 --batch_size=128 --no_evaluate <br/>