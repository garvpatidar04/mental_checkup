import streamlit as st
import pickle
import numpy as np


illnesses = {
    0: 'Somatization',
    1: 'Obsessive-Compulsive Disorder',
    2: 'Interpersonal Sensitivity',
    3: 'Depression',
    4: 'Anxiety',
    5: 'Anger-Hostility',
    6: 'Phobic Anxiety',
    7: 'Paranoid Ideation',
    8: 'Psychoticism',
    9: 'Additional Items'
}


st.title(':orange[Mental health predictor]')

c1, c2, c3 = st.columns(3)

with c1:
    ans1 = st.radio(':red[Q1]: How often do you experience headaches?', ['NOT AT ALL', 'A LITTLE BIT', 'MODERATELY', 'QUITE A BIT', 'EXTREMELY'])
    ans2 = st.radio(':red[Q2]: How often do you feel nervous or have inner shakiness?', ['NOT AT ALL', 'A LITTLE BIT', 'MODERATELY', 'QUITE A BIT', 'EXTREMELY'])
    ans3 = st.radio(':red[Q3]: How frequently do you have intrusive thoughts or ideas that you can\'t get rid of?', ['NOT AT ALL', 'A LITTLE BIT', 'MODERATELY', 'QUITE A BIT', 'EXTREMELY'])
    ans4 = st.radio(':red[Q4]: How often do you feel faint or dizzy?', ['NOT AT ALL', 'A LITTLE BIT', 'MODERATELY', 'QUITE A BIT', 'EXTREMELY'])
    ans5 = st.radio(':red[Q5]: How often do you notice a loss of interest or pleasure in sexual activities?', ['NOT AT ALL', 'A LITTLE BIT', 'MODERATELY', 'QUITE A BIT', 'EXTREMELY'])
    
with c2:
    ans6 = st.radio(':red[Q6]: How often do you find yourself being critical of others?', ['NOT AT ALL', 'A LITTLE BIT', 'MODERATELY', 'QUITE A BIT', 'EXTREMELY'])
    ans7 = st.radio(':red[Q7]: How frequently do you feel like someone else can control your thoughts?', ['NOT AT ALL', 'A LITTLE BIT', 'MODERATELY', 'QUITE A BIT', 'EXTREMELY'])
    ans8 = st.radio(':red[Q8]: How often do you feel that others are to blame for most of your problems?', ['NOT AT ALL', 'A LITTLE BIT', 'MODERATELY', 'QUITE A BIT', 'EXTREMELY'])
    ans9 = st.radio(':red[Q9]: How often do you have difficulty remembering things?', ['NOT AT ALL', 'A LITTLE BIT', 'MODERATELY', 'QUITE A BIT', 'EXTREMELY'])
    ans10 = st.radio(':red[Q10]: How often are you worried about being sloppy or careless?', ['NOT AT ALL', 'A LITTLE BIT', 'MODERATELY', 'QUITE A BIT', 'EXTREMELY'])

with c3:
    ans11 = st.radio(':red[Q11]: How often do you find yourself easily annoyed or irritated?', ['NOT AT ALL', 'A LITTLE BIT', 'MODERATELY', 'QUITE A BIT', 'EXTREMELY'])
    ans12 = st.radio(':red[Q12]: How frequently do you experience pains in your heart or chest?', ['NOT AT ALL', 'A LITTLE BIT', 'MODERATELY', 'QUITE A BIT', 'EXTREMELY'])
    ans13 = st.radio(':red[Q13]: How often do you feel afraid when you are in open spaces or on the streets', ['NOT AT ALL', 'A LITTLE BIT', 'MODERATELY', 'QUITE A BIT', 'EXTREMELY'])
    ans14 = st.radio(':red[Q14]: How often do you feel low in energy or slowed down?', ['NOT AT ALL', 'A LITTLE BIT', 'MODERATELY', 'QUITE A BIT', 'EXTREMELY'])
    ans15 = st.radio(':red[Q15]: How frequently do you have thoughts about ending your life?', ['NOT AT ALL', 'A LITTLE BIT', 'MODERATELY', 'QUITE A BIT', 'EXTREMELY'])


ans_arr = [ans1, ans2, ans3, ans4, ans5, ans6, ans7, ans8, ans9, ans10, ans11, ans12, ans13, ans14, ans15]

options = ['NOT AT ALL', 'A LITTLE BIT', 'MODERATELY', 'QUITE A BIT', 'EXTREMELY']

ans_arr_index = []

for ans in ans_arr:
    ans_arr_index.append(options.index(ans)) 

# if st.button('Show'):
#     st.write(ans_arr_index)


with open(r'pca.pkl', 'rb') as f:
    pca = pickle.load(f)

ans_arr_index_pca = pca.transform(np.expand_dims(np.array(ans_arr_index), 0))

with open(r'model1.pkl', 'rb') as f:
    model = pickle.load(f)


st.markdown("""
    <style>
    .stButton {
        display: flex;
        justify-content: center;
    }
    </style>
""", unsafe_allow_html=True)

# Using st.button and it will be centered due to the applied CSS
# if st.button('Click Me!'):
#     st.write('Button clicked!')

if st.button(':green[Get your analysis]'):
    st.write(ans_arr_index_pca[0])
    y_pred = model.predict(ans_arr_index_pca)
    
    disease_name = [illnesses[ind] for ind, val in enumerate(y_pred[0]) if val == 1 ]


    st.header(':red[Deseases:]')   
    for d in disease_name:
        st.subheader(d)




