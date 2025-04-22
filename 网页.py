import streamlit as st
import joblib
import numpy as np
import torch
import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties
from retry import retry

# 检查字体文件路径是否存在
font_path = "SimHei.ttf"
font_prop = FontProperties(fname=font_path, size=14) if fm.findfont(FontProperties(fname=font_path)) else None
if font_prop:
    plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
    plt.rcParams['axes.unicode_minus'] = False
else:
    st.warning("未找到字体文件 SimHei.ttf，将使用默认字体，可能影响中文显示")

# 添加蓝色主题的 CSS 样式
st.markdown("""
    <style>
  .main {background-color:#007BFF;background-image:url('https://www.transparenttextures.com/patterns/light_blue_fabric.png');color:#ffffff;font-family:'Arial', sans-serif;}
  .title {font-size:48px;color:#808080;font-weight:bold;text-align:center;margin-bottom:30px;}
  .subheader {font-size:28px;color:#99CCFF;margin-bottom:25px;text-align:center;border-bottom:2px solid #80BFFF;padding-bottom:10px;margin-top:20px;}
  .input-label {font-size:18px;font-weight:bold;color:#ADD8E6;margin-bottom:10px;}
  .footer {text-align:center;margin-top:50px;font-size:16px;color:#D8BFD8;background-color:#0056b3;padding:20px;border-top:1px solid #6A5ACD;}
  .button {background-color:#0056b3;border:none;color:white;padding:12px 24px;text-align:center;text-decoration:none;display:inline-block;font-size:18px;margin:20px auto;cursor:pointer;border-radius:10px;box-shadow:0px 4px 6px rgba(0,0,0,0.5);transition:background-color 0.3s, box-shadow 0.3s;}
  .button:hover {background-color:#003366;box-shadow:0px 6px 10px rgba(0,0,0,0.7);}
  .stSelectbox,.stNumberInput,.stSlider {margin-bottom:20px;}
  .stSlider > div {padding:10px;background:#E6E6FA;border-radius:10px;}
  .prediction-result {font-size:24px;color:#ffffff;margin-top:30px;padding:20px;border-radius:10px;background:#4682B4;box-shadow:0px 4px 8px rgba(0,0,0,0.3);}
  .advice-text {font-size:20px;line-height:1.6;color:#ffffff;background:#5DADE2;padding:20px;border-radius:10px;box-shadow:0px 4px 8px rgba(0,0,0,0.3);margin-top:15px;}
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
        return model, scaler, X_train_scaled
    except Exception as e:
        st.error(f"加载模型或数据失败：{str(e)}")
        return None, None, None

model, scaler, X_train_scaled = load_artifacts()

# 确保训练数据加载到session_state
if model and scaler and X_train_scaled is not None:
    st.session_state['model'] = model
    st.session_state['scaler'] = scaler
    st.session_state['X_train_scaled'] = X_train_scaled
    st.session_state['is_data_loaded'] = True
else:
    st.session_state['is_data_loaded'] = False

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
STP = st.number_input("本站气压（hPa）", min_value=0.0, value=1010.0)
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
        if not st.session_state.get('is_data_loaded', False):
            st.error("训练数据未加载，请刷新页面重新加载")
            return

        model = st.session_state['model']
        scaler = st.session_state['scaler']
        X_train_scaled = st.session_state['X_train_scaled']

        # 构建特征数组
        user_inputs = {
            'TEMP': TEMP, 'SLP': SLP, 'STP': STP, 'VISIB': VISIB,
            'WDSP': WDSP, 'CO': CO, 'NO2': NO2, 'SO2': SO2,
            'O3': O3, 'PM2.5': PM2_5, 'PM10': PM10
        }
        features_array = np.array([user_inputs[feat] for feat in FEATURES]).reshape(1, -1)

        # 标准化输入
        features_scaled = scaler.transform(features_array)
        features_tensor = torch.tensor(features_scaled, dtype=torch.float32)

        # 模型预测
        with torch.no_grad():
            outputs = model(features_tensor)
            predicted_proba = torch.softmax(outputs, dim=1).numpy()[0]
            predicted_class = np.argmax(predicted_proba)
        predicted_category = category_mapping[predicted_class]
        st.markdown(f"<div class='prediction-result'>预测类别：{predicted_category}</div>", unsafe_allow_html=True)

        # 生成建议
        probability = predicted_proba[predicted_class] * 100
        advice = {
            '严重污染': f"建议：模型预测为严重污染的概率为 {probability:.1f}%，请尽量减少外出并做好防护。",
            '重度污染': f"建议：模型预测为重度污染的概率为 {probability:.1f}%，敏感人群应避免户外活动。",
            '中度污染': f"建议：模型预测为中度污染的概率为 {probability:.1f}%，建议减少露天活动。",
            '轻度污染': f"建议：模型预测为轻度污染的概率为 {probability:.1f}%，可适当户外活动但需注意防护。",
            '良': f"建议：模型预测为良的概率为 {probability:.1f}%，适合正常户外活动。",
            '优': f"建议：模型预测为优的概率为 {probability:.1f}%，空气质量极佳，可放心享受户外。"
        }[predicted_category]
        st.markdown(f"<div class='advice-text'>{advice}</div>", unsafe_allow_html=True)

        # 计算SHAP值
        background_sample = X_train_scaled[:100]  # 使用前100个样本作为背景
        def model_predict(x):
            x_tensor = torch.tensor(x, dtype=torch.float32)
            with torch.no_grad():
                return model(x_tensor).detach().numpy()

        explainer = shap.KernelExplainer(model_predict, background_sample)
        shap_values = explainer.shap_values(features_array, nsamples=500)

        # 处理多分类SHAP值
        if isinstance(shap_values, list):
            shap_values = shap_values[predicted_class]
        shap_values = shap_values.flatten()

        # 生成瀑布图数据
        shap_importance = pd.DataFrame({
            '特征': FEATURES,
            '贡献度': shap_values
        })
        shap_importance['绝对值'] = np.abs(shap_importance['贡献度'])
        shap_importance = shap_importance.sort_values('绝对值', ascending=False).reset_index(drop=True)

        # 绘制瀑布图
        fig, ax = plt.subplots(figsize=(12, 6), facecolor='#E6E6FA')
        bars = ax.barh(
            shap_importance['特征'], 
            shap_importance['贡献度'], 
            color=['#ff5050' if v < 0 else '#66b3ff' for v in shap_importance['贡献度']],
            edgecolor='black',
            height=0.6
        )

        # 添加数值标签
        for i, bar in enumerate(bars):
            width = bar.get_width()
            if width > 0:
                ax.text(width + 0.05, i, f'{width:.2f}', va='center', fontproperties=font_prop if font_prop else None)
            else:
                ax.text(width - 0.3, i, f'{width:.2f}', va='center', ha='right', fontproperties=font_prop if font_prop else None)

        # 设置图表属性
        ax.set_title(f'{predicted_category}特征贡献度分析', fontsize=16, fontproperties=font_prop if font_prop else None)
        ax.set_xlabel('贡献度 (SHAP值)', fontsize=14, fontproperties=font_prop if font_prop else None)
        ax.set_ylabel('特征', fontsize=14, fontproperties=font_prop if font_prop else None)
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        plt.gca().invert_yaxis()  # 按贡献度从大到小排列
        plt.tight_layout()

        # 显示图表
        st.pyplot(fig)

    except Exception as e:
        st.error(f"预测过程中出现错误：{str(e)}")
        st.error("请检查输入数据是否符合规范或重新加载模型")

if st.button("预测", type="primary"):
    predict()

st.markdown('<div class="footer">© 2025 空气质量预测系统 | 支持：XXX团队</div>', unsafe_allow_html=True)
