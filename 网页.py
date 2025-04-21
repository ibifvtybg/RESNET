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
font_prop = FontProperties(fname=font_path, size=14)
plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
plt.rcParams['axes.unicode_minus'] = False

# 全局样式设置
st.set_page_config(page_title="空气质量预测", layout="wide")
st.markdown("""
<style>
.error-box {
    background-color: #ffe6e6;
    border-left: 4px solid #ff4d4d;
    padding: 15px;
    margin: 15px 0;
    border-radius: 5px;
}
.info-box {
    background-color: #e6f7ff;
    border-left: 4px solid #1890ff;
    padding: 15px;
    margin: 15px 0;
    border-radius: 5px;
}
</style>
""", unsafe_allow_html=True)

# 定义模型结构
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

# 模型加载与错误处理
@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load('RESNET.pkl')
        scaler = joblib.load('scaler.pkl')
        st.success("模型与标准化器加载成功", icon="✅")
        return model, scaler
    except FileNotFoundError:
        st.error("未找到模型或标准化器文件 (RESNET.pkl/scaler.pkl)", icon="🚨")
        return None, None
    except Exception as e:
        st.error(f"模型加载失败: {str(e)}", icon="🚨")
        return None, None

model, scaler = load_model_and_scaler()

# 特征配置
FEATURES = [
    'TEMP', 'DEWP', 'SLP', 'STP', 'VISIB', 
    'WDSP', 'MXSPD', 'MAX', 'MIN', 'PRCP', 
    'CO', 'NO2', 'SO2', 'O3', 'PM2.5', 'PM10'
]
category_mapping = {5: '严重污染', 4: '重度污染', 3: '中度污染',
                    2: '轻度污染', 1: '良', 0: '优'}

# 输入表单
st.title("空气质量指数预测系统", anchor=False)
with st.form("input_form", clear_on_submit=True):
    st.subheader("气象与污染物数据输入")
    
    # 输入验证函数
    def validate_input(value, min_val, max_val, name):
        if value < min_val or value > max_val:
            st.error(f"{name} 输入无效，范围应为 {min_val}~{max_val}", icon="⚠️")
            return False
        return True

    # 分栏布局
    col1, col2 = st.columns(2)
    with col1:
        TEMP = st.number_input("温度（℃）", min_value=-30.0, max_value=50.0, value=15.0)
        DEWP = st.number_input("露点温度（℃）", min_value=-30.0, max_value=50.0, value=10.0)
        SLP = st.number_input("海平面气压（hPa）", min_value=900.0, max_value=1100.0, value=1013.0)
        STP = st.number_input("本站气压（hPa）", min_value=900.0, max_value=1100.0, value=1010.0)
        VISIB = st.number_input("能见度（km）", min_value=0.0, max_value=50.0, value=10.0)
        WDSP = st.number_input("风速（m/s）", min_value=0.0, max_value=30.0, value=3.0)
    
    with col2:
        MXSPD = st.number_input("最大风速（m/s）", min_value=0.0, max_value=50.0, value=8.0)
        MAX = st.number_input("最高温度（℃）", min_value=-30.0, max_value=50.0, value=25.0)
        MIN = st.number_input("最低温度（℃）", min_value=-30.0, max_value=50.0, value=5.0)
        PRCP = st.number_input("降水量（mm）", min_value=0.0, max_value=500.0, value=0.0)
        CO = st.number_input("一氧化碳（CO）浓度", min_value=0.0, max_value=50.0, value=0.5)
        NO2 = st.number_input("二氧化氮（NO2）浓度", min_value=0.0, max_value=500.0, value=20.0)
        SO2 = st.number_input("二氧化硫（SO2）浓度", min_value=0.0, max_value=500.0, value=10.0)
        O3 = st.number_input("臭氧（O3）浓度", min_value=0.0, max_value=500.0, value=80.0)
        PM2_5 = st.number_input("PM2.5 浓度", min_value=0.0, max_value=1000.0, value=35.0)
        PM10 = st.number_input("PM10 浓度", min_value=0.0, max_value=1000.0, value=70.0)
    
    # 输入数据校验
    input_valid = all([
        validate_input(TEMP, -30, 50, "温度"),
        validate_input(DEWP, -30, 50, "露点温度"),
        validate_input(SLP, 900, 1100, "海平面气压"),
        validate_input(STP, 900, 1100, "本站气压"),
        validate_input(VISIB, 0, 50, "能见度"),
        validate_input(WDSP, 0, 30, "风速"),
        validate_input(MXSPD, 0, 50, "最大风速"),
        validate_input(MAX, -30, 50, "最高温度"),
        validate_input(MIN, -30, 50, "最低温度"),
        validate_input(PRCP, 0, 500, "降水量"),
        validate_input(CO, 0, 50, "一氧化碳浓度"),
        validate_input(NO2, 0, 500, "二氧化氮浓度"),
        validate_input(SO2, 0, 500, "二氧化硫浓度"),
        validate_input(O3, 0, 500, "臭氧浓度"),
        validate_input(PM2_5, 0, 1000, "PM2.5浓度"),
        validate_input(PM10, 0, 1000, "PM10浓度"),
    ])
    
    submitted = st.form_submit_button("生成预测", type="primary", use_container_width=True)

def main():
    if not submitted:
        return
    
    if not input_valid:
        st.stop()
    
    try:
        # 数据预处理
        user_input = np.array([TEMP, DEWP, SLP, STP, VISIB, WDSP, MXSPD, MAX, MIN, PRCP, CO, NO2, SO2, O3, PM2_5, PM10]).reshape(1, -1)
        user_input_scaled = scaler.transform(user_input)
        features_tensor = torch.tensor(user_input_scaled, dtype=torch.float32)
        
        # 模型预测
        with torch.no_grad():
            logits = model(features_tensor)
            probs = torch.softmax(logits, dim=1).numpy()[0]
            pred_class = np.argmax(probs)
            pred_category = category_mapping[pred_class]
            
        # 结果展示
        st.subheader("预测结果", anchor=False)
        st.markdown(f"""
        <div style="font-size: 20px; padding: 15px; background-color: #e6f7ff; border-radius: 5px;">
            <b>预测类别:</b> {pred_category}<br>
            <b>预测概率:</b> {probs[pred_class]*100:.1f}%<br>
            <b>各等级概率:</b> {", ".join([f"{k}: {v*100:.1f}%" for k, v in zip(category_mapping.values(), probs)])}
        </div>
        """, unsafe_allow_html=True)
        
        # 建议生成
        advice_dict = {
            '严重污染': "建议避免户外活动，关闭门窗，敏感人群及时就医",
            '重度污染': "建议减少外出，佩戴防颗粒物口罩，减少户外锻炼",
            '中度污染': "建议儿童、老年人及敏感人群减少户外活动",
            '轻度污染': "建议适当减少户外活动，易感人群注意防护",
            '良': "适宜户外活动，注意适时增减衣物",
            '优': "非常适宜户外活动，享受清新空气"
        }
        st.info(f"⚠️ 空气质量建议: {advice_dict[pred_category]}", icon="💡")
        
        # SHAP值分析
        st.subheader("特征重要性分析 (SHAP值)", anchor=False)
        if 'X_train' not in st.session_state or st.session_state['X_train'].empty:
            st.error("训练数据未加载，请先上传训练数据集", icon="🚨")
            return
            
        X_train = st.session_state['X_train']
        if X_train.shape[0] < 10:
            st.warning("训练数据样本量较少，SHAP值计算可能不准确", icon="⚠️")
            
        # 背景数据处理
        background_scaled = scaler.transform(X_train)
        background_tensor = torch.tensor(background_scaled, dtype=torch.float32)
        explainer = shap.DeepExplainer(model, background_tensor)
        shap_values = explainer.shap_values(features_tensor)
        
        # SHAP值维度校验
        if isinstance(shap_values, list) and len(shap_values) != len(category_mapping):
            st.error(f"SHAP值维度异常，预期类别数 {len(category_mapping)}，实际 {len(shap_values)}", icon="🚨")
            return
            
        # 提取预测类别SHAP值
        if isinstance(shap_values, list):
            shap_values = shap_values[pred_class]
        shap_values = np.array(shap_values).flatten()
        
        # 重要性排序
        importance_df = pd.DataFrame({
            '特征': FEATURES,
            'SHAP值': shap_values,
            '绝对值': np.abs(shap_values)
        }).sort_values('绝对值', ascending=False)
        
        # 数据展示
        st.dataframe(importance_df.style.format({'SHAP值': '{:.4f}', '绝对值': '{:.4f}'}), use_container_width=True)
        
        # 瀑布图绘制
        st.subheader("特征贡献度瀑布图", anchor=False)
        features_sorted = importance_df['特征'].tolist()
        contributions = importance_df['SHAP值'].tolist()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        y_pos = np.arange(len(features_sorted))
        colors = ['#1890ff' if c > 0 else '#ff4d4d' for c in contributions]
        
        ax.barh(y_pos, contributions, color=colors, edgecolor='white')
        ax.set_xlabel('SHAP值贡献度', fontproperties=font_prop)
        ax.set_ylabel('特征', fontproperties=font_prop)
        ax.set_title(f'{pred_category}预测的特征贡献分析', fontproperties=font_prop, fontsize=16)
        ax.set_yticks(y_pos, features_sorted, fontproperties=font_prop)
        
        # 添加数值标签
        for i, val in enumerate(contributions):
            ax.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=10)
        
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        st.pyplot(fig)
        
    except torch.cuda.OutOfMemoryError:
        st.error("内存不足，请减少输入数据量或重启应用", icon="🚨")
    except shap.ExplanationError as e:
        st.error(f"SHAP值计算失败: {str(e)}", icon="🚨")
    except Exception as e:
        st.error(f"预测过程中发生错误: {str(e)}", icon="🚨")
        st.exception(e)  # 开发环境显示详细堆栈

if __name__ == "__main__":
    if model and scaler:
        main()
    else:
        st.stop()
