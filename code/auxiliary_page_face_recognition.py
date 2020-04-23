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
import plotly_express as px
from test_alex import *


def page_face_recognition_show_test():
    test2 = Alextest()
    test2.load_model()
    st.write(' The sample size is '
             'relatively small, so it will be scale during the display test. ')
    test_type = st.sidebar.selectbox(
        'What type of test you want?',
        ('Directions', 'Random Test'))
    if test_type == 'Directions':
        st.header('Test types and result as following examples')
        st.image('source/face_demo/face_recognition/recognition.PNG', width=750)

    elif test_type == 'Random Test':
        st.subheader('Random Test')
        pair_nums = st.sidebar.slider(label='face nums to show', min_value=2, max_value=8, value=4, step=2)
        figure, info = test2.rand_test(face_num=pair_nums)
        st.pyplot(figure)


def choose_data_source():
    data_source = st.sidebar.selectbox(
        'What dataset you want?',
        ('JAFEE', 'CISIA-Webface'))
    info_dict = {
            'name': 'jafee',
            'class_num': 11,
            'pic_per': 20
        }
    if data_source == 'CISIA-Webface':
        info_dict = {
            'name': 'cisia_webface',
            'class_num': 50,
            'pic_per': 137
        }
    return info_dict


def page_face_recognition_show_test_show_all():
    st.subheader('Show Classify Confusion Matrix')
    info_dict = choose_data_source()
    print(info_dict['name'])
    if info_dict['name'] == 'jafee':
        st.image(['source/face_demo/face_recognition/confusion_metrix-jafee.PNG','source/face_demo/face_recognition/confusion_metrix-cisia-jafee-blue.PNG'],caption=['Gray Version','Blue Version'],width=330)
        st.markdown(base_css.format('center',20) + 'Since the dataset is performing well, no more impressions are required',unsafe_allow_html=True)
    if info_dict['name'] == 'cisia_webface':
        st.image(['source/face_demo/face_recognition/confusion_metrix-cisia-webface.PNG','source/face_demo/face_recognition/confusion_metrix-cisia-webface-blue.PNG'],width=330,caption=['Gray version','Blue version'])
        st.image(['source/face_demo/face_recognition/confusion_metrix-cisia-webface-improve.PNG','source/face_demo/face_recognition/confusion_metrix-cisia-webface-blue-improve.PNG'],width=330,caption=['Gray improved version','Blue improved version'])
        st.markdown(
            base_css.format('center', 20) + 'How do i see this picture?',
            unsafe_allow_html=True)
        st.image('source/face_demo/face_recognition/explanation.PNG',width=700)


def page_face_recognition_introduction():
    st.title('My face recognition app')
    st.markdown(base_css.format('right', '20') + "Based on AlexNet", unsafe_allow_html=True)
    st.write('')
    st.markdown(base_css.format('left', '20') + "Reasons:", unsafe_allow_html=True)
    st.markdown(base_css.format('left', '18') +
                'relatively sufficient number of photos per person,  \n'
                'such as 10:20+ or 1000:1000+(person num: photos nums)',
                unsafe_allow_html=True)
    st.markdown(base_css.format('left', '20') + "Choice:", unsafe_allow_html=True)
    st.markdown(base_css.format('left', '18') +
                'Another choice, namely face recognition.',
                unsafe_allow_html=True)
    st.markdown(base_css.format('left', '20') + "Idea:", unsafe_allow_html=True)
    st.markdown(base_css.format('left', '18') +
                'AlexNet is Powerful feature extraction capability and Means to prevent overfitting',
                unsafe_allow_html=True)
    st.write('P.S. In addition to this model, other models such as vggnet '
             'also have the same powerful feature extraction capabilities. '
             'For learning more at "Builed Model" after "Image Preprocess".')
    st.markdown(base_css.format('left', '20') + "Application Scenario:", unsafe_allow_html=True)
    st.markdown(base_css.format('left', '18') +
                'Face recognition is mostly used for automatic access control management,'
                ' unmanned supermarkets, and Skynet to trace suspects',
                unsafe_allow_html=True)


def page_face_recognition_builded_model():
    st.subheader('Process')
    st.image('source/network/recognition/process.PNG', width=700)
    st.subheader('idea')
    st.image('source/network/recognition/alexnet.PNG', width=500)
    st.subheader('JAFEE_model')
    st.image('source/network/recognition/model_jafee.PNG', caption='JAFEE_model', width=700)
    st.subheader('CISIA-Wenbface_model')
    st.image('source/network/recognition/model_cisia-webface.PNG',caption='CISIA-Webface_model', width=700)
    st.subheader('loss-theory')
    st.image('source/network/recognition/softmax or amsoftmax.PNG', width=500)
    st.subheader('loss-application')
    st.image('source/network/recognition/amsoftmax.PNG', width=500)
    st.subheader('optimizer--RMSProp')
    st.image('source/network/recognition/rmsprop-foundation.PNG', width=400)
    st.image('source/network/recognition/rmsprop.PNG', width=400)


def page_face_recognition_image_process():
    genre = st.sidebar.radio("Choose a process !", ('Data Source', 'Insufficient data set',
                                                    'Deal way'))
    if genre == 'Data Source':
        page_face_recognition_img_preprocess_datasource()
    elif genre == 'Insufficient data set':
        page_face_recognition_img_preprocess_imgproblem()
    elif genre == 'Deal way':
        page_face_recognition_img_preprocess_dealway()


def page_face_recognition_img_preprocess_datasource():
    dataset = st.sidebar.radio("Choose a dataset !", ('JAFEE(Plus)', 'CISIA-Webface'))
    if dataset == 'JAFEE(Plus)':
        st.write('The Japanese Female Facial Expression Database(or JAFEE)')
        st.write('Added some photos of myself, so named JAFEE(plus)')
        st.write('It contains 226 color facial images of 11 subjects')
        st.write('About 20 photos per person, relatively sufficient')
        st.write('Each photo size is 256 * 256')
        st.image('source/datasets/total/jafee_total.PNG', width=750)
    elif dataset == 'CISIA-Webface':
        st.write('CISIA-Webface Image Database(Incomplete version of the dataset)')
        st.write('It contains 6888 color facial images of 50 subjects')
        st.write('About 137 photos per person, relatively sufficient')
        st.write('Each photo size is 250 * 250')
        st.image('source/datasets/total/cisia-webface_total.PNG', width=750)

def page_face_recognition_img_preprocess_imgproblem():
    with open("style.css") as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)
    st.image('source/datasets/corrupted_example/recognition/ideal.PNG',
             caption='Relatively_ideal')
    st.markdown(base_css.format('center', '30') + 'Examples of Some Photo Problems:',
                unsafe_allow_html=True)
    st.image('source/datasets/corrupted_example/recognition/age.PNG',
             caption='Age_interference')
    st.image('source/datasets/corrupted_example/recognition/gray_color.PNG',caption='gray_color')
    st.image('source/datasets/corrupted_example/recognition/interrupt.PNG',caption='Interrupt')
    st.image('source/datasets/corrupted_example/recognition/other.PNG',caption='Other_person',width=300)


def page_face_recognition_img_preprocess_dealway():
    genre = st.sidebar.radio("Choose a process!", ('simplify introduction', 'show all list in a table',
                                                    'show infomation figure'))
    if genre == 'simplify introduction':
        with open("style.css") as f:
            st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)
        st.markdown(base_css.format('left', '30') + 'Normal Image Preprocessing Methods as Following:',
                    unsafe_allow_html=True)
        st.image('source/datasets/deal_way/normal/normal_process.PNG', width=700)
        # st.markdown(":barely_sunny:")
        st.markdown(base_css.format('center', '20') + 'The basic processing method is consistent with face verification',
                    unsafe_allow_html=True)
        st.markdown(base_css.format('center', '25') + 'A further method is to find the intersection based on the list of '
                                                      'images that many times changed model cannot recognize.',
                    unsafe_allow_html=True)
        st.markdown(base_css.format('center', '25') + 'Show old list, new list, intersection list, random exclude list', unsafe_allow_html=True)

        name_list = page_face_recognition_img_preprocess_dealway_allist()
        data = pd.DataFrame(name_list, columns=['name_old', 'name_new','name_intersection','name_random_exclude'])
        st.dataframe(data, width=650)

    elif genre == 'show all list in a table':
        name_list = page_face_recognition_img_preprocess_dealway_allist()
        data = pd.DataFrame(name_list, columns=['name_old', 'name_new', 'name_intersection', 'name_random_exclude'])
        st.table(data)

    elif genre == 'show infomation figure':
        st.subheader('P.S. Type=Inter means Intersection between old and new list')
        st.subheader('P.S. Type=Random means random small partial with old and new list (exclude intersection)')
        info = page_face_recognition_img_preprocess_dealway_allist_improve()
        data = pd.DataFrame(info, columns=['name', 'sub', 'type'])

        fig = px.scatter(data, x='name', y='sub', color='type')
        st.subheader('show scatter for all')
        st.plotly_chart(fig)

        fig2 = px.scatter(data, x='name', y='sub', color='type', hover_name='name', facet_col='type')
        st.subheader('show separate content')
        st.plotly_chart(fig2)

        fig3 = px.histogram(data, x='name', y='sub', histfunc='count', color='type')
        st.subheader('show histogram statistic')
        st.plotly_chart(fig3)

        fig4 = px.violin(data, y='sub', x='name', color='type', box=True, points='all')
        st.subheader('show violin statistic')
        st.plotly_chart(fig4)


def page_face_recognition_img_preprocess_dealway_allist():
    old_list, new_list = page_face_recognition_img_preprocess_dealway_originlist()

    intersection_list_ori = [val for val in new_list if val in old_list]
    random_exclude_list = [val for val in old_list if val not in intersection_list_ori] \
                          + ([val for val in new_list if val not in intersection_list_ori])
    # print(len(random_exclude_list))
    # print(len(old_list),len(new_list), len(intersection_list))
    k1 = int((len(intersection_list_ori) * 0.65))
    intersection_list = random.sample(intersection_list_ori, k1)

    k2 = int((len(old_list) + len(new_list) - 2 * len(intersection_list_ori)) * 0.15)
    random_exclude_list = random.sample(random_exclude_list, k2)

    length = len(old_list)
    for item in range(length - len(new_list)):
        new_list.append('None')
    for item in range(length - len(intersection_list)):
        intersection_list.append('None')
    for item in range(length - len(random_exclude_list)):
        random_exclude_list.append('None')

    name_list = {
        'name_new': new_list,
        'name_old': old_list,
        'name_intersection': intersection_list,
        'name_random_exclude': random_exclude_list
    }
    return name_list


def page_face_recognition_img_preprocess_dealway_originlist():
    old_list = []
    with open('source/datasets/deal_way/cross_del/old.txt', 'r') as filereader:
        lines = filereader.readlines()
        for line in lines:
            origin = line.replace('D:\For_Python\SomethingINT\holiday\mytry\some-dates\small-webface\\', '')\
                    .split('.')[0]
            dir_name = origin.split('\\')[0]
            sub = str(int(origin.split('\\')[1])-1).zfill(3)+'.jpg'
            old_list.append(dir_name+'\\'+sub)

    new_list = []
    with open('source/datasets/deal_way/cross_del/new.txt', 'r') as filereader:
        lines = filereader.readlines()
        for line in lines:
            origin = line.replace('D:\For_Python\SomethingINT\holiday\mytry\some-dates\small-webface\\', '').split('.')[0]
            dir_name = origin.split('\\')[0]
            sub = str(int(origin.split('\\')[1])).zfill(3)+'.jpg'
            new_list.append(dir_name+'\\'+sub)
    return old_list, new_list


def page_face_recognition_img_preprocess_dealway_allist_improve():

    name_list = page_face_recognition_img_preprocess_dealway_allist()
    name_list_name = []
    name_list_sub = []
    name_list_types = []

    name_list_old = name_list['name_old']
    for item in name_list_old:
        name_list_name.append(str(int(item.split("\\")[0])))
        name_list_sub.append(str(int(item.split("\\")[1].replace('.jpg',''))))
        name_list_types.append('old')

    name_list_new = name_list['name_new']
    for item in name_list_new:
        if item is not 'None':
            name_list_name.append(str(int(item.split("\\")[0])))
            name_list_sub.append(str(int(item.split("\\")[1].replace('.jpg', ''))))
            name_list_types.append('new')

    name_list_inter = name_list['name_intersection']
    for item in name_list_inter:
        if item is not 'None':
            name_list_name.append(str(int(item.split("\\")[0])))
            name_list_sub.append(str(int(item.split("\\")[1].replace('.jpg', ''))))
            name_list_types.append('inter')

    name_list_random = name_list['name_random_exclude']
    for item in name_list_random:
        if item is not 'None':
            name_list_name.append(str(int(item.split("\\")[0])))
            name_list_sub.append(str(int(item.split("\\")[1].replace('.jpg', ''))))
            name_list_types.append('random')

    name_list_all = {
        'name': name_list_name,
        'sub': name_list_sub,
        'type': name_list_types
    }

    return name_list_all


def page_face_recognition_show_applicationdemo_style(testmodel):
    figure, info = testmodel.rand_test(face_num=1)
    st.pyplot(figure)
    max_proba_name = info[0][0][0].split(':')[0]
    max_probability = float(info[0][0][0].split(':')[1].split('%')[0]) / 100
    secend_probability = float(info[0][1][0].split(':')[1].split('%')[0]) / 100
    if max_probability < 0.5:
        st.write('In doubt with' + max_proba_name)
        st.write('max_probability:' + str(max_probability))
    else:
        if (max_probability - secend_probability) > 0.25:
            st.write('You are ' + max_proba_name + ' resident')
            st.write('max_probability:' + str(max_probability))
        else:
            st.write('In doubt with' + max_proba_name)
            st.write('max_probability:' + str(max_probability))
            st.write('Maybe close')