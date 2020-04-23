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
from auxiliary_streamlit import *

page = st.sidebar.radio("Choose a page",
                        ["Homepage", "Face Verifacation", "Face Recognition",
                         "Application Demo", "Summary"])

if page == "Homepage":
    page_homepage()

elif page == "Face Verifacation":
    page_face_validation()

elif page == "Face Recognition":
    page_face_recogition()

elif page == "Application Demo":
    page_application()

elif page == 'Summary':
    page_summary()

