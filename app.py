import sys
import os

now_dir = os.getcwd()
target_dir = '/model'
cp_dir = '/_checkpoint/'
sys.path.append(now_dir + target_dir)

from flask import Flask, request, render_template, url_for, redirect

import torch
from model.model import *

from PIL import Image
import cv2
import numpy as np


# load model
model = MyModel(now_dir + target_dir + '/model.pth')
model_action = torch.load(now_dir + target_dir + '/action.pth')
model_action = model_action.cuda()
model_action.eval()

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
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

# catch all route to 'login'
@app.route('/', defaults={'path': ''})
@app.route('/<string:path>')
@app.route('/<path:path>')
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
    if os.path.isfile(now_dir + '/static/result.png'):
        os.remove(now_dir + '/static/result.png')
        
    if request.method == 'POST':
        # call 'inference' again
        if request.files['image'].filename == '':
            file = request.form['image_path']
        # get image from remote
        else :
            file = request.files['image']
        
        # inference
        img = Image.open(file).convert('RGB')
        img.save(now_dir + '/static/input.png')

        res, crop_list =  model.predict(img)
        
        output_action_list = {}
        for crop in crop_list:
            img_crop, label_crop = crop

            with torch.no_grad():
                output_action = model_action(act_transfrom(Image.fromarray(img_crop)).unsqueeze(0).cuda())
                _, pred_action = output_action.max(1)

                output_action_list[label_crop] = pred_action.data.cpu().numpy()[0]

        res.save(now_dir + '/static/result.png')

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
    # print('----------- log) inference video')
    if request.method == 'POST':
        # call 'inference' again
        file_path = ''
        if request.files['video'].filename == '':
            file = request.form['video_path']
            file_path = file
        # get video from remote
        else :
            file = request.files['video']
            file.save(now_dir + '/static/input.mp4')
            file_path = now_dir + '/static/input.mp4'
        
        # read video file
        cap = cv2.VideoCapture(file_path)
        
        width  = cap.get(3) # float
        height = cap.get(4) # float
        frameNum = cap.get(5)
        # frameAllNum = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        
        fourcc = cv2.VideoWriter_fourcc('V','P','8','0')
        out = cv2.VideoWriter(now_dir + '/static/output.webm', fourcc, frameNum, (int(width),int(height)))
        
        output_action_list = {'11':[0,0],'12':[0,0],'13':[0,0],
                                     '14':[0,0],'15':[0,0],'17':[0,0]}
        
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                                               
                res, crop_list =  model.predict(img)
                
                for crop in crop_list:
                    img_crop, label_crop = crop

                    with torch.no_grad():
                        output_action = model_action(act_transfrom(Image.fromarray(img_crop)).unsqueeze(0).cuda())
                        _, pred_action = output_action.max(1)

                        act_idx = pred_action.data.cpu().numpy()[0]
                        output_action_list[label_crop][act_idx] += 1
                        
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
        
        # result of action
        act_result = np.array(list(output_action_list.values()))
        act_result_close = act_result[:,0].tolist()
        act_result_open = act_result[:,1].tolist()
        
        # convert to % rate
        # act_sum = act_result.sum()
        # act_result_close_per = (act_result[:,0] / act_sum * 100).tolist()
        # act_result_open_per = (act_result[:,1] / act_sum * 100).tolist()
        
    return render_template("vid-inference.html", input_video=input_url, output_video=output_url,
                           act_result = [act_result_close, act_result_open]) #, act_result_close_per, act_result_open_per] )


# run server
if __name__ == '__main__':
    app.run(host=HOST, debug=True, port=PORT)
