import streamlit as st
from configure import *
# TODO 大致布局、整合规划、vggnet暂时放一放(可以考虑直接提取的特征---平均特征----均值差异-----阈值)、
#  对alexnet多分类性能评估，绘图 ，(多次实验结果汇总、最优结果曲线、出现的问题、改进办法)
#  有无必要CUDR？


def page_homepage_introduction():
    st.title('My Face application')
    st.markdown(base_css.format('right', '20') + 'A Simple Tutorial for My Project', unsafe_allow_html=True)
    st.markdown(base_css.format('center', '20') + 'A kind of biometrics technology for identity recognition based on '
                                                'human facial feature information', unsafe_allow_html=True)
    st.markdown(base_css.format('center', '25') + '1:1(Verifacation) or 1:N(Recognition)</p>', unsafe_allow_html=True)
    st.image(['source/introduction/face_validation.jpg', 'source/introduction/face_recognition.bmp'],
             caption=['face_validation', 'face_recognition'], width=320)
    st.markdown(base_css.format('center', '25') + 'Face Verifacation', unsafe_allow_html=True)
    st.write('A process that the computer compares the current face'
             ' with the database designation and finds whether it matches, which can be simply understood'
             ' as proving that you are you.')
    st.markdown(base_css.format('center', '25') + 'Face Recognition', unsafe_allow_html=True)
    st.write('A process that after collecting a photo of someone,'
             'the system finds an image that matches the current user ’s face data '
             'from the portrait database and matches it.')
    st.markdown(base_css.format('center', '25') + 'Recommended Reading Order: in Natural Order', unsafe_allow_html=True)


def page_home_overview():
    st.markdown("<h1 style='text-align: center; color: black;'>About the Following Learning Curve</h1>", unsafe_allow_html=True)
    st.write('For face verification, you can mainly look at loss:'
             '  \nbecause the main judgment is based on distance and correctness.')
    st.write('For face recognition, it should have been based on a large amount of face data to train '
             "the network model's ability to extract features, "
             "but due to hardware conditions, the quality of the pictures "
             "and so on, the scope of this concept was reduced to a simple feature extractor plus classifier. "
             "so in addition to loss, you must also see the accuracy and top-k (k default is 5): "
             "because the main judgment is based on distance and the classification effect.")
    st.markdown("<h1 style='text-align: center; color: black;'>Face Validation</h1>", unsafe_allow_html=True)
    st.image('source/experiment_records/loss_cisia-facev5.PNG', caption='CISIA-FaceV5', width=750)
    st.image('source/experiment_records/loss_feret.PNG', caption='Feret(Small Partial)', width=750)
    st.markdown("<p style='text-align: center; color: black;'>"
                "Summary: The overall situation is good, the loss of "
                "training and verification has dropped well</p>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: black;'>Face Recognition</h1>", unsafe_allow_html=True)
    st.image('source/experiment_records/learn_jafee.PNG', caption='JAFEE', width=750)
    st.write('Summary: In the case of significantly fewer categories, '
             'the number and quality of photos per capita can already meet the training requirements, '
             'and the overall situation is good')
    st.image('source/experiment_records/learn_webface.PNG', caption='CISIA-Webface(Small Partial)', width=750)
    st.write("Summary: For more category the training situation is good, the model is complex enough,"
                "But the training sample per capita seems to be insufficient,"
                "and the same time the quality of the photos needs to be improved."
                "so there is a more obvious overfitting phenomenon,"
                "and the verification set can reach an accuracy rate close to 68% -- 70% at most"
                "(The initial situation is 30%-34%)")
    st.markdown(base_css.format('center', '25') + 'Please Try Different Pages on The Left', unsafe_allow_html=True)

