import streamlit as st
import numpy as np
import pandas as pd
import altair as al
from test_sia import SampleTest
import matplotlib.pyplot as plt
import streamlit as st
import altair as alt
from test_sia import SampleTest
from keras import backend as K
from PIL import Image
from configure import *
import csv
import cv2 as cv
from Image_preprocessing import image_process
from auxiliary_page_face_validation import *
from auxiliary_page_face_recognition import *
from auxiliary_page_homepage import *
from test_sia import SiameseNet
import socket


def page_homepage():
    process = st.sidebar.radio(
        'Go through with me !',
        ('Situation',
         'Overview'))
    if process == 'Situation':
        page_homepage_introduction()
    elif process == 'Overview':
        page_home_overview()


def page_face_validation():
    process = st.sidebar.radio(
        'Go through with me !',
        ('Introduction-Verification', 'Image Preprocess', 'Builded Model', 'Show Test', 'Summary'))
    if process == 'Show Test':
        page_face_validation_show_test()
    elif process == 'Introduction-Verification':
        page_face_validation_introduction()
    elif process == 'Image Preprocess':
        page_face_validation_image_preprocess()
    elif process == 'Builded Model':
        page_face_valition_builded_model()
    elif process == 'Summary':
        page_face_validation_show_test_show_all()


def page_face_recogition():
    process = st.sidebar.radio(
        'Go through with me !',
        ('Introduction-Recognition', 'Image Preprocess', 'Builded Model', 'Show Test', 'Summary'))
    if process == 'Show Test':
        page_face_recognition_show_test()
    elif process == 'Introduction-Recognition':
        page_face_recognition_introduction()
    elif process == 'Image Preprocess':
        page_face_recognition_image_process()
    elif process == 'Builded Model':
        page_face_recognition_builded_model()
    elif process == 'Summary':
        page_face_recognition_show_test_show_all()


def page_summary():
    process = st.sidebar.radio(
        'How would you like to be contacted?',
        ('Experiment Enviroment', 'Improve Way', 'Refernce','Thanks'))
    if process == 'Experiment Enviroment':
        st.subheader('Hardware condition')
        st.image('source/summary/hardenv.PNG', width=500)
        st.subheader('Software condition')
        st.image('source/summary/softenv.PNG', width=500)
    elif process == 'Refernce':
        st.write('Blank is too small to answer')
        agree_fig = st.checkbox('Show possible reference papers?')
        if agree_fig:
            st.image('source/summary/reference.PNG',width=900)
    elif process == 'Improve Way':
        st.balloons()
        st.markdown(base_css.format('left', '35') +
                    'The best way to improve:',
                    unsafe_allow_html=True)
        st.markdown(base_css.format('left', '35') +
                    'To find large-scale high-quality data',
                    unsafe_allow_html=True)
    elif process == 'Thanks':
        st.markdown(':barely_sunny:')
        st.markdown(':family:')
        st.markdown(':male-teacher:')
        st.markdown(':rabbit2:')
        st.markdown(':bear:')
        st.markdown(':wink:')



def page_application():
    process = st.sidebar.radio(
        'How would you like to be contacted?',
        ('Introduction', 'Face Recognition', 'Face Verification'))
    if process == 'Face Recognition':
        st.markdown(":barely_sunny:")
        st.write('Sorry not implemented yet')
        st.write('The initial idea is that according to the top-5 probability value of a photo')
        st.write('first the highest probability is at least greater than a certain threshold, such as 60%')
        st.write('secondly, the difference between the highest probability value and this high probability '
                 'value is worth greater than a certain threshold, such as 35%')
        st.write('According to calculations, the sum of the remaining three is about 15%')
        st.write('Then load the record (success or failure) into the database according to the judgment result.')
        st.write('The data table can be designed as follows: '
                 'time, judgment of the person who may be effective (name of the person with the highest probability),'
                 ' judgment of the first five probability values and corresponding names, judgment result the reason')

        st.subheader('Show demo')
        testmodel = Alextest()
        testmodel.load_model()

        img_1_1 = st.file_uploader('pic_to_recognition', type='jpg')
        if img_1_1 is not None:
            img_1_2 = Image.open(img_1_1).resize((227, 227), Image.ANTIALIAS)
            figure, info = testmodel.one_test(img_1_2)
            # st.markdown(base_css.format('center', '25') + str(testmodel.pair_test(pairs=[img_1_1, img_2_1])),
            #             unsafe_allow_html=True)
            st.pyplot(figure)
            max_proba_name = info[0][0][0].split(':')[0]
            max_probability = float(info[0][0][0].split(':')[1].split('%')[0]) / 100
            secend_probability = float(info[0][1][0].split(':')[1].split('%')[0]) / 100
            if max_probability < 0.80:
                st.markdown(base_css.format('center', '25') + 'Warning, In doubt with ' + max_proba_name +
                            ' resident' + ', maybe close',
                            unsafe_allow_html=True)
            else:
                if secend_probability < 0.07:
                    st.markdown(base_css.format('center', '25') + 'Successfully, You are ' + max_proba_name +
                                ' resident' + ', max_probability:' + str(max_probability),
                                unsafe_allow_html=True)
                else:
                    st.markdown(base_css.format('center', '25') + 'Warning, In doubt with ' + max_proba_name +
                                ' resident' + ', maybe close',
                                unsafe_allow_html=True)

    elif process == 'Face Verification':
        test = SampleTest()
        process2 = st.sidebar.radio(
            'How would you like to be contacted?',
            ('Temporary Staff', 'Householder'))
        if process2 == 'Temporary Staff':
            page_face_validation_show_test_new_pair(test)
        elif process2 == 'Householder':
            page_face_validation_show_test_new_pair_2(test)
    elif process == 'Introduction':
        st.subheader('Face Recognition Process')
        st.image('source/application_demo/face_recognition_process.PNG')
        st.subheader('Face Validation Process')
        st.image('source/application_demo/face_validation_process.PNG')
