import streamlit as st
import joblib
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import shap
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 设置中文字体
font_path = "SimHei.ttf"
font_prop = FontProperties(fname=font_path, size=20)
plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
plt.rcParams['axes.unicode_minus'] = False


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


# 加载模型和标准化器
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('RESNET.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"加载模型或标准化器失败：{str(e)}")
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

st.title("空气质量指数预测")

# 输入组件
st.header("请填写以下气象和污染物数据：")
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
PM2_5 = st.number_input("PM2.5 浓度", min_value=0.0, value=35.0)
PM10 = st.number_input("PM10 浓度", min_value=0.0, value=70.0)


def predict():
    try:
        if model is None or scaler is None:
            st.error("模型或标准化器加载失败，请检查文件。")
            return

        # 获取用户输入并构建特征数组
        user_inputs = {
            'TEMP': TEMP,
            'DEWP': DEWP,
            'SLP': SLP,
            'STP': STP,
            'VISIB': VISIB,
            'WDSP': WDSP,
            'MXSPD': MXSPD,
            'MAX': MAX,
            'MIN': MIN,
            'PRCP': PRCP,
            'CO': CO,
            'NO2': NO2,
            'SO2': SO2,
            'O3': O3,
            'PM2.5': PM2_5,
            'PM10': PM10
        }
        feature_values = [user_inputs[feat] for feat in FEATURES]
        features_array = np.array([feature_values])

        # 标准化输入
        features_scaled = scaler.transform(features_array)
        features_tensor = torch.tensor(features_scaled, dtype=torch.float32)

        # 模型预测
        with torch.no_grad():
            prediction_logits = model(features_tensor)
            predicted_proba = torch.softmax(prediction_logits, dim=1).numpy()[0]

        # 获取预测类别
        predicted_class = np.argmax(predicted_proba)
        predicted_category = category_mapping[predicted_class]

        # 根据预测结果生成建议
        probability = predicted_proba[predicted_class] * 100
        probability_str = " ".join([f"{category_mapping[i]}: {predicted_proba[i] * 100:.1f}%" for i in range(len(category_mapping))])
        advice = {
            '严重污染': f"建议：根据我们的库，该日空气质量为严重污染。模型预测该日为严重污染的概率为 {probability:.1f}%。建议采取防护措施，减少户外活动。",
            '重度污染': f"建议：根据我们的库，该日空气质量为重度污染。模型预测该日为重度污染的概率为 {probability:.1f}%。建议减少外出，佩戴防护口罩。",
            '中度污染': f"建议：根据我们的库，该日空气质量为中度污染。模型预测该日为中度污染的概率为 {probability:.1f}%。敏感人群应减少户外活动。",
            '轻度污染': f"建议：根据我们的库，该日空气质量为轻度污染。模型预测该日为轻度污染的概率为 {probability:.1f}%。可以适当进行户外活动，但仍需注意防护。",
            '良': f"建议：根据我们的库，此日空气质量为良。模型预测此日空气质量为良的概率为 {probability:.1f}%。可以正常进行户外活动。",
            '优': f"建议：根据我们的库，该日空气质量为优。模型预测该日空气质量为优的概率为 {probability:.1f}%。空气质量良好，尽情享受户外时光。",
        }[predicted_category]

        # 显示预测结果
        st.subheader("预测结果")
        st.markdown(f"预测类别：**{predicted_category}**")
        st.write(f"预测概率：{probability_str}")

        # 显示建议
        st.subheader("建议")
        st.write(advice)

        # 计算 SHAP 值
        if 'X_train' not in st.session_state:
            st.error("未找到训练数据 X_train，请确保数据已正确加载。")
            return
        X_train = st.session_state['X_train']
        background = torch.tensor(scaler.transform(X_train), dtype=torch.float32)
        explainer = shap.DeepExplainer(model, background)
        shap_values = explainer.shap_values(features_tensor)

        # 打印 shap_values 的形状和类型
        st.write("shap_values 的类型:", type(shap_values))
        if isinstance(shap_values, list):
            for i, value in enumerate(shap_values):
                st.write(f"shap_values[{i}] 的形状:", value.shape)
        else:
            st.write("shap_values 的形状:", shap_values.shape)

        # 修正代码逻辑
        if isinstance(shap_values, list):
            if len(shap_values) == 1:
                shap_values = shap_values[0]
            else:
                shap_values = shap_values[predicted_class]

        # 整理特征重要性
        shap_importance = pd.DataFrame({
            'feature': FEATURES,
            'shap_value': shap_values[0]  # 取第一个样本的 SHAP 值
        })
        shap_importance['abs_value'] = np.abs(shap_importance['shap_value'])
        shap_importance = shap_importance.sort_values('abs_value', ascending=False)

        # 准备瀑布图数据
        features_sorted = shap_importance['feature'].tolist()
        contributions_sorted = shap_importance['shap_value'].tolist()

        # 初始化绘图
        fig, ax = plt.subplots(figsize=(12, 8))

        # 初始化累积值
        start = predicted_proba[predicted_class]
        prev_contributions = [start]

        # 计算每一步的累积值
        for i in range(1, len(contributions_sorted)):
            prev_contributions.append(prev_contributions[-1] + contributions_sorted[i - 1])

        # 绘制瀑布图
        for i in range(len(contributions_sorted)):
            color = '#66b3ff' if contributions_sorted[i] > 0 else '#ff5050'
            if i == 0:
                ax.barh(features_sorted[i], contributions_sorted[i], left=start - contributions_sorted[i],
                        color=color, edgecolor='black', height=0.6)
            else:
                ax.barh(features_sorted[i], contributions_sorted[i], left=prev_contributions[i],
                        color=color, edgecolor='black', height=0.6)
            plt.text(prev_contributions[i] + contributions_sorted[i] / 2, i,
                     f"{contributions_sorted[i]:+.2f}", ha='center', va='center', color='black')

        plt.title(f'预测类型为{predicted_category}时的特征贡献度瀑布图', fontsize=20, fontproperties=font_prop)
        plt.xlabel('贡献度（SHAP 值）', fontsize=16, fontproperties=font_prop)
        plt.ylabel('特征', fontsize=16, fontproperties=font_prop)
        plt.yticks(fontsize=14, fontproperties=font_prop)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.xlim(left=prev_contributions[-1] - 0.1, right=start + 0.1)
        plt.text(prev_contributions[0], -0.5, f'预测概率: {start * 100:.1f}%',
                 ha='right', va='center', fontsize=14, fontweight='bold', color='#333')

        plt.tight_layout()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"预测过程中出现错误：{str(e)}")


if st.button("预测"):
    predict()
