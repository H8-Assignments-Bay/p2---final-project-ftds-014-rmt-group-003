import streamlit as st
from detect import run
from PIL import Image
from io import *
from datetime import datetime
import os
import time
import subprocess

## CFG
cfg_model_path = "best.pt" 

def imageInput(device, src):
    
    if src == 'Upload your own data':
        image_file = st.file_uploader("Upload An Image", type=['png', 'jpeg', 'jpg'])
        col1, col2 = st.columns(2)
        if image_file is not None:
            img = Image.open(image_file)
            with col1:
                st.image(img, caption='Uploaded Image', use_column_width='always')
            ts = datetime.timestamp(datetime.now())
            imgpath = os.path.join('data/uploads', str(ts)+image_file.name).replace('\\','/')
            outputpath = os.path.join('data/outputs', os.path.basename(imgpath)).replace('\\','/')
            with open(imgpath, mode="wb") as f:
                f.write(image_file.getbuffer())

            #call Model prediction--
            run(weights=cfg_model_path ,source=imgpath)

            #--Display predicton
            img_ = Image.open(outputpath)
            with col2:
                st.image(img_, caption='Model Prediction(s)', use_column_width='always')

def webcam(src):
    if src == 'Webcam': 
        # Image selector slider
        subprocess.run(['python', 'detect.py', '--source', '0','--conf', '0.4'])

def videoInput(device, src):
    uploaded_video = st.file_uploader("Upload Video", type=['mp4', 'mpeg', 'mov'])
    if uploaded_video != None:

        ts = datetime.timestamp(datetime.now())
        imgpath = os.path.join('data/uploads', str(ts)+uploaded_video.name).replace('\\','/')
        outputpath = os.path.join('data/outputs', str(ts)+uploaded_video.name).replace('\\','/')

        with open(imgpath, mode='wb') as f:
            f.write(uploaded_video.read())  # save video to disk

        st_video = open(imgpath, 'rb')
        video_bytes = st_video.read()
        st.video(video_bytes)
        st.write("Uploaded Video")
        run(weights=cfg_model_path, source=imgpath)
        st.write("Model Prediction")
        st_video2 = open(outputpath, 'rb')
        video_bytes2 = st_video2.read()
        st.video(video_bytes2)

def main():
    # -- Sidebar
    st.sidebar.title('⚙️Options')
    datasrc = st.sidebar.radio("Select input source.", ['Upload your own data','Webcam'])
                
    option = st.sidebar.radio("Select input type.", ['Image', 'Video'], disabled = False)

    st.header('📦Logo Detection Model Demo')
    st.subheader('👈🏽 Select the options')
    if (option == "Image") & (datasrc == 'Upload your own data'):    
        imageInput('cpu', datasrc)
    elif (option == "Video") & (datasrc == 'Upload your own data'): 
        videoInput('cpu', datasrc)
    elif datasrc == 'Webcam':
        webcam(datasrc)


if __name__ == '__main__':
    main()