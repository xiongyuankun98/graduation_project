import streamlit as st
import pandas as pd


def page_summary_enviroment():
    st.title('Summarise and ')
    st.markdown("<p style='text-align: right; font-size:20px;"
                "color: black;'>A Simple Tutorial for My Project</p>", unsafe_allow_html=True)
    st.write('')
    st.write('In life, there are many places where face recognition (verification) is needed')
    st.write('Generally speaking :')
    st.write('A kind of biometrics technology for identity recognition based on'
             ' human facial feature information')
    st.write('')
    st.markdown(base_css.format('center', '25') + '1:1(Verifacation) or 1:N(Recognition)</p>', unsafe_allow_html=True)
    st.write('')
    st.image(['face_recognition.bmp', 'face_validation.jpg'],
             caption=['face_recognition', 'face_validation'], width=320)
    st.markdown(base_css.format('center', '25') + 'Face Verifacation', unsafe_allow_html=True)
    st.write('A process that the computer compares the current face'
             ' with the database designation and finds whether it matches, which can be simply understood'
             ' as proving that you are you.')
    st.markdown(base_css.format('center', '25') + 'Face Recognition', unsafe_allow_html=True)
    st.write('A process that after collecting a photo of someone,'
             'the system finds an image that matches the current user ’s face data '
             'from the portrait database and matches it.')