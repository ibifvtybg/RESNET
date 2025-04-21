# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 19:03:57 2025

@author: 18657
"""

# -*- coding: utf-8 -*-
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

# 定义 ResidualBlock 类
class ResidualBlock(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, input_size)
        self.leaky_relu = nn.LeakyReLU(0.01)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        residual = x
        out = self.leaky_relu(self.fc1(x))
        out = self.dropout(out)
        out = self.fc2(out)
        out += residual
        return out


# 定义 MultiDimensionalResNet 类
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
        x = self.residual_block1(x)
        x = self.residual_block2(x)
        x = self.residual_block3(x)
        x = self.residual_block4(x)
        x = self.fc2(x)
        return x
        
# 设置中文字体
font_path = "SimHei.ttf"
font_prop = FontProperties(fname=font_path, size=20)

# 确保 matplotlib 使用指定的字体
plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 添加蓝色主题的 CSS 样式，修复背景颜色问题
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
st.markdown('<div class="title">浦东新区监测站 CO 浓度预测</div>', unsafe_allow_html=True)

# 加载 ResNet 模型
try:
    model = joblib.load('RESNET.pkl')
except Exception as e:
    st.write(f"<div style='color: red;'>Error loading model: {e}</div>", unsafe_allow_html=True)
    model = None

# 特征顺序需与训练时一致
FEATURES = ['TEMP', 'DEWP', 'SLP', 'STP', 'VISIB', 'WDSP', 'MXSPD', 'MAX', 'MIN', 'PRCP',
            'CO', 'NO2', 'SO2', 'O3', 'PM2.5', 'PM10']

# 特征名称映射
feature_name_mapping = {
    'TEMP': '温度', 'DEWP': '露点温度', 'SLP': '海平面气压',
    'STP': '本站气压', 'VISIB': '能见度', 'WDSP': '风速',
    'MXSPD': '最大风速', 'MAX': '最高温度', 'MIN': '最低温度',
    'PRCP': '降水量', 'CO': '一氧化碳', 'NO2': '二氧化氮',
    'SO2': '二氧化硫', 'O3': '臭氧', 'PM2.5': 'PM2.5', 'PM10': 'PM10'
}

st.markdown('<div class="subheader">请填写以下气象和污染物数据：</div>', unsafe_allow_html=True)

# 输入组件
TEMP = st.number_input("温度（℃）", min_value=-30.0, value=15.0)
DEWP = st.number_input("露点温度（℃）", min_value=-30.0, value=10.0)
SLP = st.number_input("海平面气压（hPa）", min_value=900.0, value=1013.0)
STP = st.number_input("本站气压（hPa）", min_value=900.0, value=1010.0)
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
FSP = st.number_input("PM2.5 浓度", min_value=0.0, value=35.0)
RSP = st.number_input("PM10 浓度", min_value=0.0, value=70.0)


def predict():
    try:
        if model is None:
            st.write("<div style='color: red;'>模型加载失败</div>", unsafe_allow_html=True)
            return

        # 获取用户输入并构建特征数组
        user_inputs = {
            'TEMP': TEMP, 'DEWP': DEWP, 'SLP': SLP, 'STP': STP, 'VISIB': VISIB,
            'WDSP': WDSP, 'MXSPD': MXSPD, 'MAX': MAX, 'MIN': MIN, 'PRCP': PRCP,
            'CO': CO, 'NO2': NO2, 'SO2': SO2, 'O3': O3, 'PM2.5': FSP, 'PM10': RSP
        }
        feature_values = [user_inputs[feat] for feat in FEATURES]
        features_array = np.array([feature_values])

        # 这里简单假设使用均值为 0，标准差为 1 的标准化（实际需根据训练数据情况）
        mean = np.mean(features_array, axis=0)
        std = np.std(features_array, axis=0)
        std = np.where(std == 0, 1e-8, std)  # 避免除零错误
        features_scaled = (features_array - mean) / std
        features_tensor = torch.tensor(features_scaled, dtype=torch.float32)

        # 模型预测
        with torch.no_grad():
            prediction = model(features_tensor).numpy()[0][0]

        # 显示预测结果
        st.markdown(f"<div class='prediction-result'>预测 CO 浓度：{prediction:.2f} 毫克/立方米</div>", unsafe_allow_html=True)

        # 生成建议
        advice = f"当前预测 CO 浓度为 {prediction:.2f}。"
        if prediction > 3.0:
            advice += " 浓度较高，建议减少户外活动。"
        else:
            advice += " 浓度正常，可正常活动。"
        st.markdown(f"<div class='advice-text'>{advice}</div>", unsafe_allow_html=True)

        # 创建虚拟背景数据（这里简单使用用户输入作为背景）
        background = features_tensor.repeat(100, 1)
        explainer = shap.DeepExplainer(model, background)
        shap_values = explainer.shap_values(features_tensor)

        # 整理特征重要性
        shap_importance = pd.DataFrame({
            'feature': FEATURES,
            'shap_value': shap_values[0]
        })
        shap_importance['abs_value'] = np.abs(shap_importance['shap_value'])
        shap_importance = shap_importance.sort_values('abs_value', ascending=False)

        # 准备瀑布图数据
        features_sorted = shap_importance['feature'].tolist()
        contributions_sorted = shap_importance['shap_value'].tolist()
        features_sorted = [feature_name_mapping[f] for f in features_sorted]

        # 绘制瀑布图
        fig, ax = plt.subplots(figsize=(12, 8))
        start = prediction
        prev_contributions = [start]

        for i in range(len(contributions_sorted)):
            color = '#66b3ff' if contributions_sorted[i] > 0 else '#ff5050'
            if i == 0:
                ax.barh(features_sorted[i], contributions_sorted[i], left=start - contributions_sorted[i],
                        color=color, edgecolor='black', height=0.6)
            else:
                ax.barh(features_sorted[i], contributions_sorted[i], left=prev_contributions[i],
                        color=color, edgecolor='black', height=0.6)
            prev_contributions.append(prev_contributions[i] - contributions_sorted[i])
            plt.text(prev_contributions[i] + contributions_sorted[i] / 2, i,
                     f"{contributions_sorted[i]:+.2f}", ha='center', va='center', color='black')

        plt.title('特征贡献度瀑布图', fontsize=20, fontproperties=font_prop)
        plt.xlabel('CO 浓度贡献值（毫克/立方米）', fontsize=16, fontproperties=font_prop)
        plt.ylabel('特征', fontsize=16, fontproperties=font_prop)
        plt.yticks(fontsize=14, fontproperties=font_prop)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.xlim(left=prev_contributions[-1] - 0.5, right=start + 0.5)
        plt.text(prev_contributions[0], -0.5, f'预测值: {prediction:.2f}',
                 ha='right', va='center', fontsize=14, fontweight='bold', color='#333')

        plt.tight_layout()
        st.pyplot(fig)

    except Exception as e:
        st.write(f"<div style='color: red;'>Error: {e}</div>", unsafe_allow_html=True)


if st.button("预测", key="predict_button"):
    predict()

st.markdown('<div class="footer">© 2024 All rights reserved.</div>', unsafe_allow_html=True)
    
