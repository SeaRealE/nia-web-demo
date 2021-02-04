# nia-web-demo
## Environments
- python==3.6.12  
- pytorch==1.6.0  
- CUDA==10.2  
- Flask==1.1.2
<div style="text-align: center;">
<img src =https://user-images.githubusercontent.com/33483699/106848826-0c0aff00-66f5-11eb-9632-3dce78efd304.png width="300" height="300" />  
<img src =https://user-images.githubusercontent.com/33483699/106848576-8e46f380-66f4-11eb-9e4d-a892d9f58612.png width="300" height="300" />  
<img src =https://user-images.githubusercontent.com/33483699/106848578-8edf8a00-66f4-11eb-90df-63da10ed5f6d.png width="300" height="300" />  
</div>

---

## Install 
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
$ ./run.sh
```
, or
```bash
$ python app.py
```