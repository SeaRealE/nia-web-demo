from flask import Flask, request, render_template, url_for, redirect

import sys
import os

import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from segmentation.model import *

import cv2
import numpy as np


# load model
sys.path.append(os.getcwd()+'/model/')
model = MyModel('./model/model.pth')

model_action = models.resnet101()
model_action.fc = nn.Sequential(nn.Linear(2048, 2048),
                         nn.BatchNorm1d(num_features=2048),
                         nn.ReLU(),
                         nn.Linear(2048, 1024),
                         nn.BatchNorm1d(num_features=1024),
                         nn.ReLU(),
                         nn.Linear(1024, 2))
model_action = model_action.cuda()
model_action.eval()
model_action.load_state_dict(torch.load('./action/all.pth'))

transform_val = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


# load server
HOST = '0.0.0.0'
PORT = 8893
app = Flask(__name__)

# no cache
@app.after_request
def set_response_headers(r):
    r.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    r.headers['Pragma'] = 'no-cache'
    r.headers['Expires'] = '0'
    return r

# catch all route to 'login'
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def index(path):
    return redirect(url_for('login'))

# login-page
@app.route('/login')
def login():
    return render_template("login.html")

# login val
@app.route('/image', methods=['POST'])
def login_success():
    # 'switch_page' returns 0 or 1, if no exists None
    if request.form.get('switch_page') != None:
        return render_template("img-inference.html")
    
    # password validation
    userfile = open('PASSWORD', 'r')
    user_result = userfile.read()
    if request.method == 'POST':
        user_pw = request.form['password']
    
        if user_pw == user_result:
            # print('----------- log) login success')
            return render_template("img-inference.html")
        else:
            # print('----------- log) login failed')
            return redirect(url_for('login'))

# image-inference page        
@app.route('/img-inference', methods=['POST'])
def image_predict():
    # print('----------- log) inference image')
    # remove 'result.png'
    if os.path.isfile('./static/result.png'):
        os.remove('./static/result.png')
        
    if request.method == 'POST':
        # call 'inference' again
        if request.files['image'].filename == '':
            file = request.form['image_path']
        # get image from remote
        else :
            file = request.files['image']
        
        # inference
        img = Image.open(file).convert('RGB')
        img.save('./static/input.png')

        res, crop_list =  model.predict(img)
        
        output_action_list = {}
        
        for crop in crop_list:
            img_crop, label_crop = crop

            model_action.load_state_dict(torch.load('./action/' + label_crop + '.pth'))

            with torch.no_grad():
                output_action = model_action(transform_val(Image.fromarray(img_crop)).unsqueeze(0).cuda())
                _, pred_action = output_action.max(1)

                output_action_list[label_crop] = pred_action.data.cpu().numpy()[0]

        
        
        res.save('./static/result.png')
        # Image.fromarray(res).save('./static/result.png')

        # return a path of 'input.png', 'result.png' 
        input_url = url_for('static', filename='input.png')
        output_url = url_for('static', filename='result.png')
            
        return render_template("img-inference.html", input_img='.'+input_url, output_img='.'+output_url,
                              output_action_list = output_action_list)

# video page
@app.route('/video', methods=['POST'])
def video_page():
    return render_template("vid-inference.html")

# video-inference page 
@app.route('/vid-inference', methods=['POST'])
def video_predict():
    global model_action
    
    # print('----------- log) inference video')
    if request.method == 'POST':
        # call 'inference' again
        file_path = ''
        if request.files['video'].filename == '':
            file = request.form['video_path']
            file_path = file
        # get image from remote
        else :
            file = request.files['video']
            file.save('./static/input.mp4')
            file_path = './static/input.mp4'
        
        # read video file
        cap = cv2.VideoCapture(file_path)
        
        width  = cap.get(3) # float
        height = cap.get(4) # float
        frameNum = cap.get(5)
        frameAllNum = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        
        fourcc = cv2.VideoWriter_fourcc('V','P','8','0')
        out = cv2.VideoWriter('./static/output.webm', fourcc, 30, (int(width),int(height)))
        
        count = 0
        
        output_action_list = {'11':[0,0],'12':[0,0],'13':[0,0],
                                     '14':[0,0],'15':[0,0],'17':[0,0]}
           
        model_action.load_state_dict(torch.load('./action/all.pth'))
        
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                                               
                res, crop_list =  model.predict(img)
                
                for crop in crop_list:
                    img_crop, label_crop = crop

                    # model_action.load_state_dict(torch.load('./action/' + label_crop + '.pth'))

                    with torch.no_grad():
                        output_action = model_action(transform_val(Image.fromarray(img_crop)).unsqueeze(0).cuda())
                        _, pred_action = output_action.max(1)

                        act_idx = pred_action.data.cpu().numpy()[0]
                        output_action_list[label_crop][act_idx] += 1

                # frameAllNum -= 1
                # print('log) remain :',frameAllNum)
            
                out.write(np.array(res))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        out.release()
        cap.release()

        # return a path of 'input.mp4', 'output.webm' 
        input_url = url_for('static', filename='input.mp4')
        output_url = url_for('static', filename='output.webm')
        
        act_result = np.array(list(output_action_list.values()))
        act_result_close = act_result[:,0].tolist()
        act_result_open = act_result[:,1].tolist()
        
        act_sum = act_result.sum()
        act_result_close_per = (act_result[:,0] / act_sum * 100).tolist()
        act_result_open_per = (act_result[:,1] / act_sum * 100).tolist()
        
    
    
    return render_template("vid-inference.html", input_video=input_url, output_video=output_url,
                           act_result = [act_result_close, act_result_open, act_result_close_per, act_result_open_per] )

# @app.route('/test')
# def test():  
#     return render_template("test.html")

# run server
if __name__ == '__main__':
    app.run(host=HOST, debug=True, port=PORT)
