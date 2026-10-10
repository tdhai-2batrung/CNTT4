import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title='Sàng lọc nguy cơ tiểu đường', page_icon='🩺')
st.title('Sàng lọc nguy cơ tiểu đường')
st.caption('Công cụ nghiên cứu hỗ trợ sàng lọc, không thay thế chẩn đoán y khoa.')

@st.cache_resource
def load_model():
    return joblib.load('diabetes_model.joblib')

model = load_model()

with st.form('patient_form'):
    age = st.number_input('Tuổi', min_value=0, max_value=120, value=40)
    gender = st.selectbox('Giới tính', ['Nam', 'Nữ'])
    st.subheader('Kết quả xét nghiệm')
    st.info('Chỉ nhập số liệu khi đã xác nhận đúng đơn vị với dữ liệu huấn luyện. Ví dụ: Glucose mmol/L; HbA1C % nếu bộ dữ liệu gốc dùng các đơn vị này.')
    columns = ['Glucose', 'HbA1C', 'Cholesterol', 'HDL', 'LDL', 'Triglycerid', 'Creatinine', 'Ure', 'AcidUric', 'AST', 'ALT']
    values = {}
    for i, col in enumerate(columns):
        values[col] = st.number_input(col, min_value=0.0, value=None, format='%.3f', placeholder='Nhập kết quả xét nghiệm', key=col)
    submitted = st.form_submit_button('Dự đoán nguy cơ', type='primary')

if submitted:
    if any(v is None for v in values.values()):
        st.error('Vui lòng nhập đầy đủ các chỉ số xét nghiệm trước khi dự đoán.')
    else:
        patient = pd.DataFrame([{'Tuoi': age, 'GioiTinh': gender, **values}])
        prediction = int(model.predict(patient)[0])
        probability = float(model.predict_proba(patient)[0, list(model.classes_).index(1)])
        st.subheader('Kết quả sàng lọc')
        st.metric('Điểm nguy cơ do mô hình ước tính', f'{probability:.1%}')
        if prediction == 1:
            st.warning('Mô hình phân loại kết quả vào nhóm có nguy cơ. Bạn nên trao đổi với bác sĩ để được đánh giá và tư vấn phù hợp.')
        else:
            st.success('Mô hình phân loại kết quả vào nhóm nguy cơ thấp hơn. Điều này không loại trừ bệnh tiểu đường.')
        st.write('**Giải thích:** Mô hình sử dụng tuổi, giới tính và các kết quả xét nghiệm đã nhập để đưa ra dự đoán. Điểm trên là đầu ra của mô hình, không phải kết luận chẩn đoán hay xác suất mắc bệnh đã được kiểm chứng lâm sàng.')
        st.write('**Lưu ý:** Không tự ý thay đổi thuốc hoặc trì hoãn khám bệnh dựa trên kết quả này. Nếu có triệu chứng bất thường, hãy liên hệ nhân viên y tế.')
