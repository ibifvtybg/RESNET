import streamlit as st
import joblib
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties

# 检查字体文件路径是否存在
font_path = "SimHei.ttf"
if fm.findfont(FontProperties(fname=font_path)):
    font_prop = FontProperties(fname=font_path, size=20)
    plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
    plt.rcParams['axes.unicode_minus'] = False
else:
    st.warning("未找到字体文件 SimHei.ttf，可能会影响中文显示。")

# 添加蓝色主题的 CSS 样式
st.markdown("""
    <style>
  .main {
        background-color: #007BFF;
        background-image: url('https://www.transparenttextures.com/patterns/light_blue_fabric.png');
        color: #ffffff;
        font-family: 'Arial', sans-serif;
    }
  .title {
        font-size: 48px;
        color: #808080;
        font-weight: bold;
        text-align: center;
        margin-bottom: 30px;
    }
  .subheader {
        font-size: 28px;
        color: #99CCFF;
        margin-bottom: 25px;
        text-align: center;
        border-bottom: 2px solid #80BFFF;
        padding-bottom: 10px;
        margin-top: 20px;
    }
  .input-label {
        font-size: 18px;
        font-weight: bold;
        color: #ADD8E6;
        margin-bottom: 10px;
    }
  .footer {
        text-align: center;
        margin-top: 50px;
        font-size: 16px;
        color: #D8BFD8;
        background-color: #0056b3;
        padding: 20px;
        border-top: 1px solid #6A5ACD;
    }
  .button {
        background-color: #0056b3;
        border: none;
        color: white;
        padding: 12px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 18px;
        margin: 20px auto;
        cursor: pointer;
        border-radius: 10px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.5);
        transition: background-color 0.3s, box-shadow 0.3s;
    }
  .button:hover {
        background-color: #003366;
        box-shadow: 0px 6px 10px rgba(0, 0, 0, 0.7);
    }
  .stSelectbox,.stNumberInput,.stSlider {
        margin-bottom: 20px;
    }
  .stSlider > div {
        padding: 10px;
        background: #E6E6FA;
        border-radius: 10px;
    }
  .prediction-result {
        font-size: 24px;
        color: #ffffff;
        margin-top: 30px;
        padding: 20px;
        border-radius: 10px;
        background: #4682B4;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.3);
    }
  .advice-text {
        font-size: 20px;
        line-height: 1.6;
        color: #ffffff;
        background: #5DADE2;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.3);
        margin-top: 15px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main'>", unsafe_allow_html=True)

# 页面标题
st.markdown('<div class="title">空气质量指数预测</div>', unsafe_allow_html=True)

# 定义 ResidualBlock 类
class ResidualBlock(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, input_size)
        self.leaky_relu = nn.LeakyReLU(0.01)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        residual = x.clone()
        out = self.leaky_relu(self.fc1(x))
        out = out.clone()
        out = self.dropout(out)
        out = self.fc2(out)
        out += residual
        return out

class MultiDimensionalResNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MultiDimensionalResNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.residual_block1 = ResidualBlock(hidden_size, hidden_size)
        self.residual_block2 = ResidualBlock(hidden_size, hidden_size)
        self.residual_block3 = ResidualBlock(hidden_size, hidden_size)
        self.residual_block4 = ResidualBlock(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.dropout(self.fc1(x))
        x = x.clone()
        x = self.residual_block1(x)
        x = self.residual_block2(x)
        x = self.residual_block3(x)
        x = self.residual_block4(x)
        x = self.fc2(x)
        return x

# 加载模型和标准化器
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('RESNET.pkl')
        scaler = joblib.load('scaler.pkl')
        X_train_scaled = joblib.load('X_train.pkl')
        st.session_state['X_train_scaled'] = X_train_scaled
        st.session_state['is_data_loaded'] = True
        return model, scaler
    except:
        return None, None
model, scaler = load_artifacts()

# 特征顺序需与训练时一致
FEATURES = ['TEMP', 'DEWP', 'SLP', 'STP', 'VISIB', 'WDSP', 'MXSPD', 'MAX', 'MIN', 'PRCP', 'CO', 'NO2', 'SO2', 'O3', 'PM2.5', 'PM10']

# 定义空气质量类别映射
category_mapping = {
    5: '严重污染',
    4: '重度污染',
    3: '中度污染',
    2: '轻度污染',
    1: '良',
    0: '优'
}

st.markdown('<div class="subheader">请填写以下气象和污染物数据：</div>', unsafe_allow_html=True)

# 输入组件
TEMP = st.number_input("温度（℃）", min_value=-30.0, value=15.0)
DEWP = st.number_input("露点温度（℃）", min_value=-30.0, value=10.0)
SLP = st.number_input("海平面气压（hPa）", min_value=900.0, value=1013.0)
STP = st.number_input("本站气压（hPa）", min_value=0.0, value=1010.0)
VISIB = st.number_input("能见度（km）", min_value=0.0, value=10.0)
WDSP = st.number_input("风速（m/s）", min_value=0.0, value=3.0)
MXSPD = st.number_input("最大风速（m/s）", min_value=0.0, value=8.0)
MAX = st.number_input("最高温度（℃）", min_value=-20.0, value=25.0)
MIN = st.number_input("最低温度（℃）", min_value=-30.0, value=5.0)
PRCP = st.number_input("降水量（mm）", min_value=0.0, value=0.0)
CO = st.number_input("一氧化碳（CO）浓度", min_value=0.0, value=0.5)
NO2 = st.number_input("二氧化氮（NO2）浓度", min_value=0.0, value=20.0)
SO2 = st.number_input("二氧化硫（SO2）浓度", min_value=0.0, value=10.0)
O3 = st.number_input("臭氧（O3）浓度", min_value=0.0, value=80.0)
PM2_5 = st.number_input("PM2.5 浓度", min_value=0.0, value=35.0)
PM10 = st.number_input("PM10 浓度", min_value=0.0, value=70.0)

def predict():
    if model is None or scaler is None:
        st.write(f"<div style='color: red;'>模型或标准化器加载失败</div>", unsafe_allow_html=True)
        return

    # 获取用户输入并构建特征数组
    user_inputs = {f: globals()[f] for f in FEATURES}
    features_array = np.array([list(user_inputs.values())])
    
    # 标准化输入
    features_scaled = scaler.transform(features_array)
    features_tensor = torch.tensor(features_scaled, dtype=torch.float32).clone()

    # 模型预测
    with torch.no_grad():
        prediction_logits = model(features_tensor)
        predicted_proba = torch.softmax(prediction_logits, dim=1).numpy()[0]
        predicted_class = np.argmax(predicted_proba)
    
    predicted_category = category_mapping[predicted_class]
    st.markdown(f"<div class='prediction-result'>预测类别：{predicted_category}</div>", unsafe_allow_html=True)

    # 生成建议
    probability = predicted_proba[predicted_class] * 100
    advice = {
        '严重污染': f"建议：空气质量为严重污染，模型预测概率 {probability:.1f}%。请减少户外活动并做好防护。",
        '重度污染': f"建议：空气质量为重度污染，模型预测概率 {probability:.1f}%。敏感人群应避免外出。",
        '中度污染': f"建议：空气质量为中度污染，模型预测概率 {probability:.1f}%。建议减少户外运动时间。",
        '轻度污染': f"建议：空气质量为轻度污染，模型预测概率 {probability:.1f}%。可适当活动但需注意防护。",
        '良': f"建议：空气质量为良，模型预测概率 {probability:.1f}%。适合正常户外活动。",
        '优': f"建议：空气质量为优，模型预测概率 {probability:.1f}%。可放心享受户外。",
    }[predicted_category]
    st.markdown(f"<div class='advice-text'>{advice}</div>", unsafe_allow_html=True)

    # SHAP分析（简化版）
    if st.session_state.get('is_data_loaded', False):
        X_train_scaled = st.session_state['X_train_scaled']
        background_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).clone()
        
        # 使用KernelExplainer
        def model_predict(x):
            return model(torch.tensor(x, dtype=torch.float32)).detach().numpy()
        
        explainer = shap.KernelExplainer(model_predict, background_tensor[:100].numpy())
        shap_values = explainer.shap_values(features_scaled, nsamples=100)
        
        # 处理多分类SHAP值
        if isinstance(shap_values, list):
            shap_values = shap_values[predicted_class]
        shap_values = np.array(shap_values).flatten()
        
        # 绘制瀑布图
        shap_importance = pd.DataFrame({'feature': FEATURES, 'shap_value': shap_values})
        shap_importance = shap_importance.sort_values('shap_value', ascending=False)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        shap_importance.plot.barh(x='feature', y='shap_value', ax=ax, color=['#66b3ff' if v>0 else '#ff5050' for v in shap_values])
        plt.title('特征重要性瀑布图', fontproperties=font_prop, size=18)
        plt.xlabel('SHAP值', fontproperties=font_prop, size=14)
        plt.ylabel('特征', fontproperties=font_prop, size=14)
        plt.tight_layout()
        st.pyplot(fig)

if st.button("预测"):
    predict()

st.markdown('<div class="footer">© 2024 All rights reserved.</div>', unsafe_allow_html=True)
