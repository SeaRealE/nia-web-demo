# nia-web-demo
Surgical image segmentation and action classification using PyTorch and Flask
## Environments
- python==3.6.12  
- pytorch==1.6.0  
- CUDA==10.2  
- Flask==1.1.2

<img src =https://user-images.githubusercontent.com/33483699/106849247-b5ea8b80-66f5-11eb-8e0b-ce60e5f3be89.png width="100%" height="auto" />  

---

## Install 
Using ```Imbalanced Dataset Sampler``` from [ufoym/imbalanced-dataset-sampler](https://github.com/ufoym/imbalanced-dataset-sampler)
```bash
$ git clone https://github.com/ufoym/imbalanced-dataset-sampler.git
$ cd imbalanced-dataset-sampler
$ python setup.py install
$ pip install .
$ cd ..
```
```bash
$ git clone https://github.com/SeaRealE/nia-web-demo.git ./nia-web-demo
$ cd nia-web-demo
$ pip3 install -r requirements.txt
```

## Prepare
Before do this, you need ```cp_seg.pth``` and ```cp_act.pth``` in the directory ```./_checkpoint```
```bash
$ python tools/load_model.py
```
Set port allow
```bash
$ sudo ufw allow 8893
```

## Run
```bash
$ ./run_nia_demo.sh
```
, or
```bash
$ python app.py
```
