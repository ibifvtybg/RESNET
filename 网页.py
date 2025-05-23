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
st.markdown('<div class="title">上海市空气质量指数预测</div>', unsafe_allow_html=True)

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
        model_path = 'RESNET.pkl'
        scaler_path = 'scaler.pkl'
        X_train_path = 'X_train.pkl'
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        X_train_scaled = joblib.load(X_train_path)
        if 'X_train_scaled' not in st.session_state:
            st.session_state['X_train_scaled'] = X_train_scaled
            st.session_state['is_data_loaded'] = True
        return model, scaler
    except Exception as e:
        st.error(f"加载模型或标准化器失败：{str(e)}")
        return None, None


model, scaler = load_artifacts()

# 特征顺序需与训练时一致
FEATURES = ['TEMP', 'SLP', 'STP', 'VISIB', 'WDSP', 'CO', 'NO2', 'SO2', 'O3', 'PM2.5', 'PM10']

# 定义空气质量类别映射
category_mapping = {
    5: '严重污染',
    4: '重度污染',
    3: '中度污染',
    2: '轻度污染',
    1: '良',
    0: '优'
}

st.markdown('<div class="subheader">请填写以下气象和污染物数据以进行空气质量预测：</div>', unsafe_allow_html=True)

# 输入组件
TEMP = st.number_input("温度（℃）", min_value=-30.0, value=15.0)
SLP = st.number_input("海平面气压（hPa）", min_value=900.0, value=1013.0)
STP = st.number_input("气压（hPa）", min_value=0.0, value=1010.0)
VISIB = st.number_input("能见度（km）", min_value=0.0, value=10.0)
WDSP = st.number_input("风速（m/s）", min_value=0.0, value=3.0)
CO = st.number_input("一氧化碳（CO）浓度", min_value=0.0, value=0.5)
NO2 = st.number_input("二氧化氮（NO2）浓度", min_value=0.0, value=20.0)
SO2 = st.number_input("二氧化硫（SO2）浓度", min_value=0.0, value=10.0)
O3 = st.number_input("臭氧（O3）浓度", min_value=0.0, value=80.0)
PM2_5 = st.number_input("PM2.5 浓度", min_value=0.0, value=35.0)
PM10 = st.number_input("PM10 浓度", min_value=0.0, value=70.0)


def predict():
    try:
        if model is None or scaler is None:
            st.error("模型或标准化器未正确加载")
            return

        # 获取用户输入并构建特征数组
        user_inputs = {
            'TEMP': TEMP,
            'SLP': SLP,
            'STP': STP,
            'VISIB': VISIB,
            'WDSP': WDSP,
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
        features_tensor = torch.tensor(features_scaled, dtype=torch.float32, requires_grad=True).clone()

        # 模型预测
        with torch.no_grad():
            prediction_logits = model(features_tensor)
            predicted_proba = torch.softmax(prediction_logits, dim=1).numpy()[0]
            predicted_class = np.argmax(predicted_proba)
        predicted_category = category_mapping[predicted_class]
        st.markdown(f"<div class='prediction-result'>预测类别：{predicted_category}</div>", unsafe_allow_html=True)

        # 根据预测结果生成建议
        probability = predicted_proba[predicted_class] * 100
        advice = {
            '严重污染': f"建议：根据我们的库，该日空气质量为严重污染。模型预测该日为严重污染的概率为 {probability:.1f}%。建议采取防护措施，减少户外活动。",
            '重度污染': f"建议：根据我们的库，该日空气质量为重度污染。模型预测该日为重度污染的概率为 {probability:.1f}%。建议减少外出，佩戴防护口罩。",
            '中度污染': f"建议：根据我们的库，该日空气质量为中度污染。模型预测该日为中度污染的概率为 {probability:.1f}%。敏感人群应减少户外活动。",
            '轻度污染': f"建议：根据我们的库，该日空气质量为轻度污染。模型预测该日为轻度污染的概率为 {probability:.1f}%。可以适当进行户外活动，但仍需注意防护。",
            '良': f"建议：根据我们的库，此日空气质量为良。模型预测此日空气质量为良的概率为 {probability:.1f}%。可以正常进行户外活动。",
            '优': f"建议：根据我们的库，该日空气质量为优。模型预测该日空气质量为优的概率为 {probability:.1f}%。空气质量良好，尽情享受户外时光。",
        }[predicted_category]
        st.markdown(f"<div class='advice-text'>{advice}</div>", unsafe_allow_html=True)

        # 加载背景数据
        if 'X_train_scaled' not in st.session_state or not st.session_state['is_data_loaded']:
            X_train_path = 'X_train.pkl'
            X_train_scaled = joblib.load(X_train_path)
            st.session_state['X_train_scaled'] = X_train_scaled
            st.session_state['is_data_loaded'] = True
        X_train_scaled = st.session_state['X_train_scaled']

        # 确保 background_tensor 不需要梯度
        background_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).clone()

        # 确保 features_tensor 不需要梯度
        features_tensor = torch.tensor(features_scaled, dtype=torch.float32).clone()

        # 使用更小的背景数据集样本
        background_sample = background_tensor[:100]  # 使用前100个样本作为背景

        # 创建解释器前确保模型在eval模式
        model.eval()

        # 使用 KernelExplainer 替代 DeepExplainer
        # 首先定义一个包装函数
        def model_predict(x):
            if isinstance(x, np.ndarray):
                x = torch.tensor(x, dtype=torch.float32)
            with torch.no_grad():
                return model(x).detach().numpy()

        # 使用 KernelExplainer
        explainer = shap.KernelExplainer(model_predict, background_sample.numpy())

        # 计算 SHAP 值
        shap_values = explainer.shap_values(features_tensor.numpy(), nsamples=100)

        # 处理多分类SHAP值
        if isinstance(shap_values, list):
            shap_values = shap_values[predicted_class]
        elif len(shap_values.shape) == 3:  # 处理(1,16,6)形状
            shap_values = shap_values[0, :, predicted_class]  # 取第一个样本，所有特征，当前类别的SHAP值

        # 确保最终是一维数组(11,)
        if len(shap_values.shape) > 1:
            shap_values = shap_values.flatten()

        # 验证形状是否正确
        if shap_values.shape != (11,):
            st.error(f"SHAP值形状异常 ({shap_values.shape})，无法继续")
            return

        # 创建DataFrame
        shap_importance = pd.DataFrame({
            'feature': FEATURES,
            'shap_value': shap_values
        })
        shap_importance['abs_value'] = np.abs(shap_importance['shap_value'])
        shap_importance = shap_importance.sort_values('abs_value', ascending=False)

        # 准备绘制瀑布图的数据
        features = shap_importance['feature'].tolist()
        contributions = shap_importance['shap_value'].tolist()

        # 确保瀑布图的数据是按贡献度绝对值降序排列的
        sorted_indices = np.argsort(np.abs(contributions))[::-1]
        features_sorted = [features[i] for i in sorted_indices]
        contributions_sorted = [contributions[i] for i in sorted_indices]

        min_contribution = min(contributions_sorted)

        # 初始化绘图
        fig, ax = plt.subplots(figsize=(14, 8))

        # 初始化累积值
        start = 0
        prev_contributions = [start]

        # 计算每一步的累积值
        for i in range(1, len(contributions_sorted)):
            prev_contributions.append(prev_contributions[-1] + contributions_sorted[i - 1])

        # 绘制瀑布图
        for i in range(len(contributions_sorted)):
            color = '#ff5050' if contributions_sorted[i] < 0 else '#66b3ff'
            if i == len(contributions_sorted) - 1:
                ax.barh(features_sorted[i], contributions_sorted[i], left=prev_contributions[i], color=color, edgecolor='black', height=0.5, zorder=2, hatch='/')
            else:
                ax.barh(features_sorted[i], contributions_sorted[i], left=prev_contributions[i], color=color, edgecolor='black', height=0.5, zorder=2)

            plt.text(prev_contributions[i] + contributions_sorted[i] / 2, i, f"{contributions_sorted[i]:.2f}",
                     ha='center', va='center', fontsize=10, fontproperties=font_prop, color='black')

        # 设置图表属性
        plt.title(f'预测类型为{predicted_category}时的特征贡献度瀑布图', size=20, fontproperties=font_prop)
        plt.xlabel('贡献度 (SHAP 值)', fontsize=20, fontproperties=font_prop)
        plt.ylabel('特征', fontsize=20, fontproperties=font_prop)
        plt.yticks(size=20, fontproperties=font_prop)
        plt.xticks(size=20, fontproperties=font_prop)
        plt.grid(axis='x', linestyle='--', alpha=0.7)

        # 增加边距避免裁剪
        plt.xlim(left=min_contribution * 1.2, right=max(prev_contributions) + max(contributions_sorted) * 0.8)
        fig.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.2)

        plt.tight_layout()

        # 保存并在 Streamlit 中展示
        plt.savefig("shap_waterfall_plot.png", bbox_inches='tight', dpi=1200)
        st.image("shap_waterfall_plot.png")

    except ValueError as ve:
        st.write(f"<div style='color: red;'>预测错误：{ve}</div>", unsafe_allow_html=True)
    except IndexError as ie:
        st.write(f"<div style='color: red;'>预测过程中索引错误：{ie}</div>", unsafe_allow_html=True)
    except Exception as e:
        st.write(f"<div style='color: red;'>预测过程中出现意外错误：{e}</div>", unsafe_allow_html=True)


if st.button("预测"):
    predict()

st.markdown('<div class="footer">© 2025 All rights reserved.</div>', unsafe_allow_html=True)
