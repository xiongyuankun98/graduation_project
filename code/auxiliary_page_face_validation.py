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
import math
from Image_preprocessing import image_process
from configure import *
import os

def choose_data_source():
    data_source = st.sidebar.selectbox(
        'What dataset you want?',
        ('Feret', 'CISIA-FACEV5'))
    info_dict = {
            'name': 'feret',
            'source_path': 'D:/For_Python/SomethingINT/holiday/mytry/some-dates/Feret(副本)/FERET_80_80-人脸数据库/',
            'class_num': 200,
            'pic_per': 7
        }
    if data_source == 'CISIA-FACEV5':
        info_dict = {
            'name': 'cisia_facev5',
            'source_path': 'D:/For_Python/SomethingINT/holiday/mytry/some-dates/64_CASIA-FaceV5/CASIA-FaceV5 1st/',
            'class_num': 500,
            'pic_per': 5
        }
    return info_dict


def page_face_validation_show_test():
    test = SampleTest()
    print(id(test))
    test_type = st.sidebar.selectbox(
        'What type of test you want?',
        ('Directions', 'Random Test', 'Inner Test', 'Select Threshold'))
    if test_type == 'Directions':
        st.header('Test types and result as following examples')
        st.subheader('Random Test')
        st.image('source/face_demo/face_validation/rand_test.PNG', width=750)
        st.subheader('Inner Test')
        st.image('source/face_demo/face_validation/inner_test.PNG', width=750)

    elif test_type == 'Random Test':
        st.subheader('Random Test (with threshold 0.5)')
        info_dict = choose_data_source()
        pair_nums = st.sidebar.slider(label='pair nums to show', min_value=2, max_value=9, value=3, step=1)
        st.pyplot(test.rand_test(pair_nums, face_dataset=info_dict['source_path'],
                                  nb_class=int(info_dict['class_num']), per_pic=int(info_dict['pic_per'])))
    elif test_type == 'Inner Test':
        st.subheader('Inner Test (with threshold 0.5)')
        info_dict = choose_data_source()
        pair_nums = st.sidebar.slider(label='pair nums to show', min_value=2, max_value=9, value=3, step=1)
        st.pyplot(test.unit_test(pair_nums, face_dataset=info_dict['source_path'],
                                  nb_class=int(info_dict['class_num']), per_pic=int(info_dict['pic_per'])))
    elif test_type == 'Select Threshold':
        page_face_validation_show_test_multi_test(test)


def page_face_validation_show_test_new_pair(testmodel):
    st.subheader('New Pair Validation(with threshold 0.5)')
    img_1_1 = st.file_uploader('Pic_1', type='bmp')
    img_2_1 = st.file_uploader('Pic_2', type='bmp')
    if img_1_1 and img_2_1 is not None:
        img_1_2 = Image.open(img_1_1).resize((input_size_height, input_size_weight), Image.ANTIALIAS)
        img_2_2 = Image.open(img_2_1).resize((input_size_height, input_size_weight), Image.ANTIALIAS)
        print(type(img_1_2))
        img_inter = np.zeros((input_size_height, int(input_size_weight / 6), 3), np.uint8)
        img_inter.fill(255)
        img_inter = Image.fromarray(img_inter.astype('uint8'))
        st.image([img_inter, img_1_2, img_inter, img_2_2])
        st.markdown(base_css.format('center', '25') + str(testmodel.pair_test(pairs=[img_1_1, img_2_1])), unsafe_allow_html=True)


def page_face_validation_show_test_new_pair_2(testmodel):
    st.subheader('Householder Validation(with threshold 0.5)')
    st.markdown(base_css.format('center', '20') + '默认显示0号照片，初始为0号住户0号照片',
                unsafe_allow_html=True)
    name_number = st.number_input('Insert serial number', min_value=0, max_value=49, value=0, step=1)
    empty = st.empty()
    path = webface_path + '\\' + str(name_number).zfill(4) + '\\' + str(0).zfill(3) + '.jpg'
    img_exist = Image.open(path).resize((input_size_height, input_size_weight), Image.ANTIALIAS)
    empty.image(img_exist)
    img_test = st.file_uploader('Pic_2', type='jpg')
    if img_test is not None:
        img_2_2 = Image.open(img_test).resize((input_size_height, input_size_weight), Image.ANTIALIAS)
        st.image(img_2_2)
        st.markdown(base_css.format('center', '25') + str(testmodel.pair_test(pairs=[img_exist, img_test])), unsafe_allow_html=True)


def page_face_validation_show_test_multi_test(testmodel):
    st.subheader('Multi Test')
    info_dict = choose_data_source()
    a = info_dict['class_num']
    b = info_dict['pic_per']
    minimal_num = int(a * b / math.gcd(a, b))
    margins = st.sidebar.number_input(label='threshold to segmentation', min_value=0.1, max_value=1.0, value=0.5,
                                      step=0.1)
    pair_nums = minimal_num if a == 500 else int(minimal_num/2)
    st.write('pair nums = {}(with threshold {})'.format(pair_nums * 2,margins)
             if a == 500 else 'pair nums = 1300(with threshold {})'.format(margins))

    rate_index = testmodel.confusion_metricx(pair_num=pair_nums, margin=margins,
                                             photo_source=info_dict['source_path'],
                                             nb_class=int(info_dict['class_num']), per_pic=int(info_dict['pic_per']))
    st.pyplot()
    data = {
        'Name': ['accuracy(ACC)', 'precision(PPV)', 'recall(TPR)', 'FDR(1-PPV)', 'F1score'],
        'Value': [rate_index[0], rate_index[1], rate_index[2], rate_index[3], rate_index[4]],
        'Calculation formula': ['(TP+ TN) / ALL', 'TP / (TP+ FP)',
                                'TP / (TP+ FN)', 'TP / (TP+ FN)', '2* TPR* PPV / (TPR+ PPV)']
    }
    columns = ['Name', 'Value', 'Calculation formula']
    show_data = pd.DataFrame(data=data, index=['准确度', '精度', '召回率', '错误发现率', 'F1得分'], columns=columns)
    st.table(show_data)


def page_face_validation_show_test_show_all():
    st.subheader('Something to show')
    info_dict = choose_data_source()
    agree_data = st.sidebar.checkbox('show data with different margin?')
    agree_fig = st.sidebar.checkbox('show fig with ROC?')
    print(info_dict['name'])
    if info_dict['name'] == 'feret':
        if agree_data:
            tmp_lst = []
            with open('source/face_demo/face_validation/judge.csv', 'r') as filereader:
                reader = csv.reader(filereader)
                for row in reader:
                    tmp_lst.append(row)
            show_data = pd.DataFrame(tmp_lst[1:], columns=tmp_lst[0])
            st.dataframe(show_data)
        if agree_fig:
            SampleTest.statics_roc_auc(path='source/face_demo/face_validation/judge.csv', show_roc=True)
            st.pyplot()
    if info_dict['name'] == 'cisia_facev5':
        if agree_data:
            tmp_lst = []
            with open('source/face_demo/face_validation/judge2.csv', 'r') as filereader:
                reader = csv.reader(filereader)
                for row in reader:
                    tmp_lst.append(row)
            show_data = pd.DataFrame(tmp_lst[1:], columns=tmp_lst[0])
            st.dataframe(show_data)
        if agree_fig:
            SampleTest.statics_roc_auc(path='source/face_demo/face_validation/judge2.csv', show_roc=True)
            st.pyplot()


def page_face_validation_image_preprocess():
    # st.markdown("<p style='text-align: center; color: black;'>
    # The main image preprocessing methods used in this experiment are:</p>", unsafe_allow_html=True)
    genre = st.sidebar.radio("Choose a process !", ('Data Source', 'Insufficient data set',
                                                    'Deal way(CISIA-FACEV5 example)'))
    if genre == 'Data Source':
        page_face_validation_img_preprocess_datasource()
    elif genre == 'Insufficient data set':
        page_face_validation_img_preprocess_imgproblem()
    elif genre == 'Deal way(CISIA-FACEV5 example)':
        page_face_validation_img_preprocess_dealway()


def page_face_validation_introduction():
    st.title('My face validation app')
    st.markdown(base_css.format('right', '20') + "Based on SiameseNet", unsafe_allow_html=True)
    st.write('')
    st.markdown(base_css.format('left', '20') + "Reasons:", unsafe_allow_html=True)
    st.markdown(base_css.format('left', '18') +
                'Limited by hardware limitations and data set capacity Insufficient photos per capita.',
                unsafe_allow_html=True)
    st.markdown(base_css.format('left', '20') + "Choice:", unsafe_allow_html=True)
    st.markdown(base_css.format('left', '18') +
                'Easier way, namely face verification.',
                unsafe_allow_html=True)
    st.markdown(base_css.format('left', '20') + "Idea:", unsafe_allow_html=True)
    st.markdown(base_css.format('left', '18') +
                'Siamese network is a "joint neural network". '
                'The "joint" of a neural network is realized by sharing weights. '
                'Mainly to measure the similarity of the two inputs, '
                'and apply to the number of categories is large, '
                'and the amount of data in each category is small, '
                'such as signature verification, face verification tasks and so on.',
                unsafe_allow_html=True)
    st.write('P.S. There are also variations of this network: Pseudo-SiamaseNet, '
             'but this experiment used is the original version. '
             'For this time learn more at "Builed Model" after "Image Preprocess".')
    st.markdown(base_css.format('left', '20') + "Application Scenario:", unsafe_allow_html=True)
    st.markdown(base_css.format('left', '18') +
                'Face verification is usually in dormitory access control, hotel registration, etc. '
                'Compare the existing photos in the database with the face photos taken by '
                'the live camera to verify whether it is the person.',
                unsafe_allow_html=True)


def page_face_validation_img_preprocess_datasource():
    dataset = st.sidebar.radio("Choose a dataset !", ('CASIA-FACEV5', 'FERET'))
    if dataset == 'CASIA-FACEV5':
        st.write('CASIA Face Image Database Version 5.0 (or CASIA-FaceV5)')
        st.write('It contains 2,500 color facial images of 500 subjects')
        st.write('5 photos per person, too few to suitable for face recognition training')
        st.write('Each photo size is 640 * 480')
        st.image('source/datasets/total/cisia-facev5_total.PNG', width=750)
    elif dataset == 'FERET':
        st.write('FERET Image Database(Incomplete version of the dataset)')
        st.write('It contains 1,400 color facial images of 200 subjects')
        st.write('7 photos per person, too few to suitable for face recognition training')
        st.write('Each photo size is 80 * 80')
        st.image('source/datasets/total/feret_total.PNG', width=750)


def page_face_validation_img_preprocess_dealway():
    with open("style.css") as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)
    st.markdown(base_css.format('left', '30') + 'Normal Image Preprocessing Methods as Following:',
                unsafe_allow_html=True)
    st.image('source/datasets/deal_way/normal/normal_process.PNG', width=700)
    # st.markdown(":barely_sunny:")
    st.write('P.S. Normalized: For the convenience of data processing, '
             'it is more convenient and fast to map the data to the range of 0 to 1 for processing.'
             '  \nP.S. Standardization: The mean is 0 and the variance is 1')
    st.markdown(base_css.format('left', '30') +
                'There Is A Simplified Example:',
                unsafe_allow_html=True)
    st.write('')
    st.image('source/datasets/deal_way/normal/000_0.bmp', caption='Original Image', width=400)
    st.write('')
    st.image(np.array(Image.open('source/datasets/deal_way/normal/000_0_1.bmp')), caption='Face Detection', width=400)
    st.write('')
    st.image(np.array(Image.open('source/datasets/deal_way/normal/000_0_2.bmp')), caption='Position & Crop', width=100)
    st.write('')
    st.image(np.array(Image.open('source/datasets/deal_way/normal/000_0_3.bmp')), caption='Face Scaled', width=150)
    st.write('')
    st.image(Image.open('source/datasets/deal_way/normal/000_0_4.bmp'),
             caption='Face GrayScale: Yes', width=150)
    st.image(Image.open('source/datasets/deal_way/normal/000_0_5.bmp'),
             caption='Face Histogram Equalization(Optional, Not actually adopted)', width=150)
    df = pd.DataFrame(
        image_process('source/datasets/deal_way/normal/000_0_4.bmp', gray=True, he=True, z_score=True, blur=False)[1])
    st.dataframe(df)
    st.markdown("<p style='text-align: center; color: black;'>"
                "Normalized Iamge Data</p>", unsafe_allow_html=True)
    df = pd.DataFrame(
        image_process('source/datasets/deal_way/normal/000_0_4.bmp', gray=True, he=True, z_score=True, blur=False)[0])
    st.dataframe(df)
    st.markdown("<p style='text-align: center; color: black;'>"
                "Standardization Iamge Data</p>", unsafe_allow_html=True)


def page_face_validation_img_preprocess_imgproblem():
    with open("style.css") as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)
    st.image('source/datasets/corrupted_example/verification/Relatively_ideal.PNG',
             caption='Relatively_ideal')
    st.markdown(base_css.format('center', '30') + 'Examples of Some Photo Problems:',
                unsafe_allow_html=True)
    st.image(['source/datasets/corrupted_example/verification/Blurred.PNG',
              'source/datasets/corrupted_example/verification/Detection_error.PNG',
              ],
             caption=['Blurred', 'Detection_error'])
    st.image(['source/datasets/corrupted_example/verification/Glasses_interference.PNG',
              'source/datasets/corrupted_example/verification/Insufficient_light.PNG'],
             caption=['Glasses_interference', 'Insufficient_light']
             )


def page_face_valition_builded_model():
    st.subheader('Process')
    st.image('source/network/validation/process.PNG', width=700)
    st.subheader('idea')
    st.image('source/network/validation/siamese.PNG', width=500)
    st.subheader('model')
    st.image('source/network/validation/model.PNG', width=700)
    st.image('source/network/validation/model_sequencial.PNG', width=700)
    st.subheader('loss-theory')
    st.image('source/network/validation/Contrastive_loss.PNG', width=500)
    st.subheader('loss-application')
    st.image('source/network/validation/loss_application.PNG', width=700)
    st.subheader('optimizer--Adadelta')
    st.image('source/network/validation/adadelta.PNG', width=400)
# TODO 验证的  数据集问题  标题选取  预处理流程图  model 一系列 总结过渡
# TODO 识别的  数据集问题  标题选取  预处理流程图  model 一系列  读取txt  dataframe  读取csv dataframe  多分类评估器并换成图像
# TODO application demo + CURD  + @st.cache
# TODO model  cache