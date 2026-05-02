import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="福建地铁地产经济数据分析系统",
    page_icon="🚇",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 样式 ====================
st.markdown("""
<style>
    footer {visibility: hidden;}
    section[data-testid="stSidebar"] {width: 280px;}
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        margin-bottom: 30px;
        text-align: center;
    }
    .paper-card {
        background-color: #ffffff;
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.1);
        border-left: 5px solid #667eea;
        margin: 20px 0;
    }
    .decoration {
        text-align: center;
        font-size: 2.5rem;
        margin: 15px 0;
    }
</style>
""", unsafe_allow_html=True)

# ==================== 初始化 session ====================
if 'in_main_system' not in st.session_state:
    st.session_state.in_main_system = False

# ==================== 预定义换乘站列表 ====================
TRANSFER_STATIONS = {
    "厦门": ["吕厝", "火炬园", "体育中心", "五缘湾", "湖滨东路", "厦门火车站", "官任"],
    "福州": ["南门兜", "东街口", "树兜", "金山", "紫阳", "福州火车站", "前屿", "董屿·福建师大"]
}


def is_transfer_station(city, station_name):
    """判断是否为换乘站"""
    if city not in TRANSFER_STATIONS:
        return False
    for ts in TRANSFER_STATIONS[city]:
        if ts in station_name:
            return True
    return False


# ==================== 数据加载与处理函数 ====================
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
    else:
        df = pd.DataFrame(columns=['city', 'sale_price_num', 'distance_to_station',
                                   'distance_bin', 'community_district', 'station_name'])
    return df


def preprocess(df):
    required = ['sale_price_num', 'distance_to_station', 'distance_bin',
                'community_district', 'city', 'station_name']
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"缺少必要列: {missing}")
        return None
    df['sale_price_num'] = pd.to_numeric(df['sale_price_num'], errors='coerce')
    df['distance_to_station'] = pd.to_numeric(df['distance_to_station'], errors='coerce')
    df = df.dropna(subset=['sale_price_num']).copy()
    if len(df) == 0:
        st.error("无有效房价数据")
        return None
    if 'distance_bin' in df.columns:
        order = ['0-500米', '500-1000米', '1000-1500米', '1500-2000米', '2000米以外']
        order = [o for o in order if o in df['distance_bin'].unique()]
        df['distance_bin'] = pd.Categorical(df['distance_bin'], categories=order, ordered=True)
    # 添加换乘站标记
    df['is_transfer'] = df.apply(lambda row: is_transfer_station(row['city'], row['station_name']), axis=1)
    return df


def remove_outliers(df, method='iqr', upper_limit=None):
    if method == 'none' or len(df) == 0:
        return df, 0, None, None
    price = df['sale_price_num']
    low, high = None, None
    if method == 'iqr':
        q1 = price.quantile(0.25)
        q3 = price.quantile(0.75)
        iqr = q3 - q1
        low = max(0, q1 - 1.5 * iqr)
        high = q3 + 1.5 * iqr
        mask = (price >= low) & (price <= high)
    elif method == 'cap':
        if upper_limit is None or upper_limit <= 0:
            return df, 0, None, None
        low = price.min()
        high = upper_limit
        mask = price <= upper_limit
    else:
        return df, 0, None, None
    removed = len(df) - mask.sum()
    return df[mask].copy(), removed, low, high


# ==================== 主分析系统 ====================
def main_analysis_system():
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/subway.png", width=80)
        st.markdown("### 欢迎，访客")
        if st.button("🏠 返回首页"):
            st.session_state.in_main_system = False
            st.rerun()

        st.markdown("---")
        menu = st.radio(
            "功能导航",
            ["📊 数据概览", "📏 距离分析", "🚉 站点类型分析", "🏙️ 城市对比", "📋 数据详情"],
            index=0
        )
        st.markdown("---")
        st.subheader("数据设置")
        uploaded_file = st.file_uploader("上传CSV数据文件", type=["csv"])
        outlier_method = st.radio(
            "异常值剔除",
            ['不过滤', 'IQR法（1.5倍四分位距）', '固定上限法'],
            index=0
        )
        upper_limit = None
        if outlier_method == '固定上限法':
            upper_limit = st.number_input("房价上限(元/㎡)", min_value=1000, max_value=200000, value=50000, step=5000)
        method_map = {'不过滤': 'none', 'IQR法（1.5倍四分位距）': 'iqr', '固定上限法': 'cap'}
        method = method_map[outlier_method]

        df_raw = load_data(uploaded_file)
        if df_raw is None or df_raw.empty:
            st.warning("请上传数据文件")
            return
        df_clean = preprocess(df_raw)
        if df_clean is None:
            return

        cities = sorted(df_clean['city'].dropna().unique())
        options = ["全部"]
        for city in cities:
            options.append(f"{city} - 全部")
            districts = sorted(df_clean[df_clean['city'] == city]['community_district'].dropna().unique())
            for dist in districts:
                options.append(f"{city} - {dist}")
        selected = st.selectbox("地区选择", options=options)

        if selected == "全部":
            filter_city = None
            filter_district = None
        else:
            parts = selected.split(" - ")
            filter_city = parts[0]
            filter_district = None if parts[1] == "全部" else parts[1]

        df_filtered, removed, low_th, high_th = remove_outliers(df_clean, method=method, upper_limit=upper_limit)
        if removed > 0:
            st.success(f"已剔除 {removed} 条异常值 (阈值: {low_th:.0f} - {high_th:.0f})")

        if filter_city:
            df_filtered = df_filtered[df_filtered['city'] == filter_city]
        if filter_district:
            df_filtered = df_filtered[df_filtered['community_district'] == filter_district]

        if len(df_filtered) == 0:
            st.error("当前筛选无数据")
            return

        st.session_state.df = df_filtered
        if 'distance_bin' in df_filtered.columns and df_filtered['distance_bin'].notna().any():
            existing_cats = [c for c in df_filtered['distance_bin'].cat.categories if
                             c in df_filtered['distance_bin'].unique()]
            df_filtered['distance_bin'] = pd.Categorical(df_filtered['distance_bin'], categories=existing_cats,
                                                         ordered=True)
            st.session_state.df = df_filtered

    if 'df' not in st.session_state:
        st.warning("请先在侧边栏上传数据")
        return
    df = st.session_state.df

    if menu == "📊 数据概览":
        st.header("核心统计指标")
        overall = pd.DataFrame({
            '指标': ['总样本量', '平均房价', '中位数房价', '最低房价', '最高房价', '标准差'],
            '数值': [
                len(df),
                f"{df['sale_price_num'].mean():.0f} 元/㎡",
                f"{df['sale_price_num'].median():.0f} 元/㎡",
                f"{df['sale_price_num'].min():.0f} 元/㎡",
                f"{df['sale_price_num'].max():.0f} 元/㎡",
                f"{df['sale_price_num'].std():.0f} 元/㎡"
            ]
        })
        st.table(overall)
        city_avg = df.groupby('city')['sale_price_num'].mean().round(0).reset_index()
        city_avg.columns = ['城市', '平均房价 (元/㎡)']
        st.subheader("各城市平均房价")
        st.table(city_avg)
        dist_avg = df.groupby('community_district')['sale_price_num'].mean().round(0).sort_values(ascending=False).head(
            10).reset_index()
        dist_avg.columns = ['行政区', '平均房价 (元/㎡)']
        st.subheader("行政区均价TOP10")
        st.table(dist_avg)
        fig_box = px.box(df, y='sale_price_num', title="房价箱线图（异常值检测）")
        st.plotly_chart(fig_box, use_container_width=True)

    elif menu == "📏 距离分析":
        st.header("地铁站点步行距离对房价的影响")
        if 'distance_bin' in df.columns and df['distance_bin'].notna().any():
            dist_summary = []
            for dist in df['distance_bin'].cat.categories:
                sub = df[df['distance_bin'] == dist]
                if len(sub) == 0:
                    continue
                p = sub['sale_price_num']
                dist_summary.append({
                    '距离区间': dist,
                    '样本量': len(sub),
                    '均价': f"{p.mean():,.0f}",
                    '中位数': f"{p.median():,.0f}",
                    '价格范围': f"{p.min():,.0f} - {p.max():,.0f}"
                })
            st.table(pd.DataFrame(dist_summary))
            dist_mean = df.groupby('distance_bin', observed=False)['sale_price_num'].mean().reset_index()
            fig_line = px.line(dist_mean, x='distance_bin', y='sale_price_num',
                               markers=True, text='sale_price_num',
                               labels={'distance_bin': '步行距离区间', 'sale_price_num': '平均房价 (元/㎡)'},
                               title="距离衰减趋势图")
            fig_line.update_traces(texttemplate='%{text:.0f}', textposition='top center')
            st.plotly_chart(fig_line, use_container_width=True)
            fig_box_dist = px.box(df, x='distance_bin', y='sale_price_num',
                                  labels={'distance_bin': '步行距离区间', 'sale_price_num': '房价 (元/㎡)'},
                                  title="各距离区间房价分布")
            st.plotly_chart(fig_box_dist, use_container_width=True)
            fig_scatter = px.scatter(df, x='distance_to_station', y='sale_price_num',
                                     opacity=0.6, trendline="lowess",
                                     labels={'distance_to_station': '步行距离 (米)', 'sale_price_num': '房价 (元/㎡)'},
                                     title="步行距离与房价散点图（含趋势线）")
            st.plotly_chart(fig_scatter, use_container_width=True)
            dist_counts = df['distance_bin'].value_counts().reset_index()
            dist_counts.columns = ['距离区间', '样本量']
            fig_pie = px.pie(dist_counts, names='距离区间', values='样本量',
                             title="各距离区间样本量占比", hole=0.3)
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("数据缺少距离区间字段")

    elif menu == "🚉 站点类型分析":
        st.header("换乘站与普通站溢价特征对比")

        # 检查是否有换乘站数据
        if 'is_transfer' not in df.columns or df['is_transfer'].sum() == 0:
            st.warning("当前数据中未识别到换乘站，请检查换乘站列表或站点名称。预定义的换乘站列表见代码。")
        else:
            # 1. 整体房价分布对比（箱线图）
            fig_box_type = px.box(df, x='is_transfer', y='sale_price_num',
                                  labels={'is_transfer': '站点类型', 'sale_price_num': '房价 (元/㎡)'},
                                  title="换乘站 vs 普通站 周边房价分布对比",
                                  category_orders={'is_transfer': [True, False]})
            fig_box_type.update_xaxes(tickvals=[True, False], ticktext=['换乘站', '普通站'])
            st.plotly_chart(fig_box_type, use_container_width=True)

            # 2. 按距离区间分别统计换乘站和普通站的平均房价表格
            st.subheader("各距离区间平均房价对比（元/㎡）")
            transfer_avg = df[df['is_transfer'] == True].groupby('distance_bin', observed=False)[
                'sale_price_num'].mean().round(0)
            normal_avg = df[df['is_transfer'] == False].groupby('distance_bin', observed=False)[
                'sale_price_num'].mean().round(0)
            compare_df = pd.DataFrame({
                '距离区间': transfer_avg.index,
                '换乘站均价': transfer_avg.values,
                '普通站均价': normal_avg.values,
                '溢价额 (换乘-普通)': (transfer_avg.values - normal_avg.values).astype(int),
                '溢价率 (%)': ((transfer_avg.values - normal_avg.values) / normal_avg.values * 100).round(1)
            })
            # 格式化数值
            compare_df['换乘站均价'] = compare_df['换乘站均价'].apply(lambda x: f"{x:,}")
            compare_df['普通站均价'] = compare_df['普通站均价'].apply(lambda x: f"{x:,}")
            compare_df['溢价额 (换乘-普通)'] = compare_df['溢价额 (换乘-普通)'].apply(lambda x: f"{x:,}")
            st.table(compare_df)

            # 3. 柱状图对比换乘站和普通站各距离区间均价
            plot_df = pd.DataFrame({
                '距离区间': list(transfer_avg.index) * 2,
                '平均房价': list(transfer_avg.values) + list(normal_avg.values),
                '站点类型': ['换乘站'] * len(transfer_avg) + ['普通站'] * len(normal_avg)
            })
            fig_bar = px.bar(plot_df, x='距离区间', y='平均房价', color='站点类型',
                             barmode='group', title="换乘站与普通站各距离区间平均房价对比",
                             labels={'平均房价': '房价 (元/㎡)', '距离区间': '步行距离区间'})
            st.plotly_chart(fig_bar, use_container_width=True)

            # 4. 额外统计：整体平均房价对比
            avg_transfer = df[df['is_transfer'] == True]['sale_price_num'].mean()
            avg_normal = df[df['is_transfer'] == False]['sale_price_num'].mean()
            st.info(
                f"📊 整体来看，换乘站周边住宅平均房价为 **{avg_transfer:,.0f} 元/㎡**，普通站周边为 **{avg_normal:,.0f} 元/㎡**，换乘站溢价约 **{(avg_transfer - avg_normal):,.0f} 元/㎡（{(avg_transfer / avg_normal - 1) * 100:.1f}%）**。")

    elif menu == "🏙️ 城市对比":
        st.header("城市间房价对比分析")

        st.subheader("各城市及主要行政区平均房价")
        district_summary = df.groupby(['city', 'community_district']).agg(
            平均房价=('sale_price_num', 'mean'),
            样本量=('sale_price_num', 'count')
        ).round(0).reset_index()
        district_summary['平均房价'] = district_summary['平均房价'].astype(int).apply(lambda x: f"{x:,} 元/㎡")
        district_summary = district_summary.sort_values(['city', '平均房价'], ascending=[True, False])
        st.dataframe(district_summary, use_container_width=True, hide_index=True)

        if len(df['city'].unique()) > 1:
            fig_city_box = px.box(df, x='city', y='sale_price_num',
                                  labels={'city': '城市', 'sale_price_num': '房价 (元/㎡)'},
                                  title="各城市房价分布对比")
            st.plotly_chart(fig_city_box, use_container_width=True)
            if 'distance_bin' in df.columns:
                pivot = df.groupby(['city', 'distance_bin'], observed=False)['sale_price_num'].mean().round(0).unstack()
                fig_heat = px.imshow(pivot, text_auto=True, aspect="auto",
                                     labels=dict(x="距离区间", y="城市", color="平均房价 (元/㎡)"),
                                     title="各城市不同距离区间平均房价热力图")
                st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.info("当前数据仅包含一个城市")

    elif menu == "📋 数据详情":
        st.header("原始数据示例（前50行）")
        display_cols = ['city', 'community_district', 'community_name', 'sale_price_num',
                        'distance_bin', 'station_name', 'is_transfer']
        exist_cols = [c for c in display_cols if c in df.columns]
        st.dataframe(df[exist_cols].head(50), use_container_width=True, hide_index=True)
        csv = df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button("下载筛选后数据 (CSV)", data=csv, file_name="filtered_data.csv", mime="text/csv")


# ==================== 论文首页 ====================
def paper_home_page():
    st.markdown("""
    <div class="main-header">
        <h1>🚇 福建地铁地产经济数据分析系统</h1>
        <p>—— 基于Python的地铁站点周边住宅价格空间分异分析</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="paper-card">
        <h2 style="color:#667eea; text-align:center;">📄 论文信息</h2>
        <hr>
        <p><strong>论文题目：</strong> 福建地铁地产经济的数据分析</p>
        <p><strong>姓　　名：</strong> 朱思明</p>
        <p><strong>学　　号：</strong> 2024349256</p>
        <p><strong>年　　级：</strong> 2024级</p>
        <p><strong>专　　业：</strong> 数据科学与大数据技术</p>
        <p><strong>指导教师：</strong> 林锋</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="decoration">🏙️ 🚇 📊 🏘️ 📈 🗺️</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.info("""
        **系统功能**
        - 数据上传与异常值剔除（IQR法/固定上限）
        - 步行距离分区统计（0-500米,500-1000米,1000-1500米,1500-2000米,2000米以外）
        - 距离衰减折线图/箱线图/散点图
        - 换乘站与普通站溢价对比（箱线图、分区间表格、柱状图）
        - 各城市房价对比、行政区TOP10、各行政区明细表
        """)
    with col2:
        st.success("""
        **技术栈**
        - Python + Pandas + NumPy
        - Streamlit（交互式Web框架）
        - Plotly（动态可视化）
        - 高德地图API + 安居客爬虫
        """)

    st.markdown("---")
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        if st.button("✨ 进入系统 ✨", use_container_width=True):
            st.session_state.in_main_system = True
            st.rerun()

    st.markdown("""
    <div style="text-align:center; margin-top:30px; color:#666;">
        <hr>
        <p>阳光学院 人工智能学院 | 数据科学与大数据技术 2024级</p>
        <p>© 2025 朱思明 版权所有</p>
    </div>
    """, unsafe_allow_html=True)


# ==================== 主程序 ====================
def main():
    if st.session_state.in_main_system:
        main_analysis_system()
    else:
        paper_home_page()


if __name__ == "__main__":
    main()