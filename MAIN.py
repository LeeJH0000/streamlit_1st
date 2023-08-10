import streamlit as st
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
from tkinter.tix import COLUMN
from pyparsing import empty
from datetime import datetime, timedelta
import altair as alt
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from matplotlib import font_manager, rc
import base64
import plotly.express as px


# 데이터 불러오기
data = pd.read_csv('801_df_done.csv', encoding='cp949')

# 레이아웃 설정하기
st.set_page_config(layout='wide')

# 페꾸!!
st.markdown('<style>body { background-color: #ece6cc; }</style>', unsafe_allow_html=True)

# 금요일조 이름 넣기!
image_path = 'title.PNG'
sub_path = 'subtitle.PNG'
home_button_image = f'<a href="." style="text-decoration: none;"><img src="data:image/jpeg;base64,{base64.b64encode(open(image_path, "rb").read()).decode()}" alt="Image" width="450" height="110" style="margin-right: 10px;"></a>'
st.markdown(f'<h1 style="text-align: center; margin-top: -100px;">{home_button_image}</h1>', unsafe_allow_html=True)
st.markdown(f'<h2 style="text-align: center; margin-top: -20px;"><img src="data:image/jpeg;base64,{base64.b64encode(open(sub_path, "rb").read()).decode()}" alt="Image" width="600" height="55" style="margin-right: 10px;"></h2>', unsafe_allow_html=True)
# 검색창 만들기
company_list = data['name']

selected_companies = st.multiselect('어떤 종목이 궁금하신가요?', data['name'])
company_button = st.button("검색")

############################################################
col1, col2 = st.columns([4,3])
com = '삼성전자'
stocks = fdr.StockListing('KRX')
item_name = com
code = stocks[stocks['Name'] == item_name]['Code'].to_string(index=False).strip()
end_date = datetime.today()
start_date = datetime(2023, 5, 1)
stock_data = fdr.DataReader(code, start=start_date, end=end_date)

############################################################
if company_button:
    st.markdown("------")
    selected_company = st.radio("##### 검색 결과:", selected_companies)
    st.markdown("------")
    # 선택한 기업에 따라 새로운 페이지 보여주기
    if selected_company:
        
        st.write(f"### {selected_company} 주가 차트 📊")
        stocks = fdr.StockListing('KRX')
        item_name = selected_company
        code = stocks[stocks['Name'] == item_name]['Code'].to_string(index=False).strip()
        end_date = datetime.today()
        start_date = datetime(2023, 5, 1)
        stock_data = fdr.DataReader(code, start=start_date, end=end_date)
        
        ### candle chart 도전!!!
        fig = go.Figure(data=[go.Candlestick(
            x=stock_data.index,
            open=stock_data['Open'],
            high=stock_data['High'],
            low=stock_data['Low'],
            close=stock_data['Close']
        )])     
        fig.update_layout(
            xaxis_title="날짜",
            yaxis_title="주가",
            xaxis_rangeslider_visible=False
        )    
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('--------')
        
        # 거래량 차트
        st.write(f"### {selected_company} 거래량 차트 📉")
        stocks_vol = pd.DataFrame()
        stocks_vol['거래량'] = stock_data['Volume']
        fig = px.line(stocks_vol, x=stocks_vol.index, y='거래량')
        fig.update_layout(xaxis_title="날짜", yaxis_title="거래량")
        st.plotly_chart(fig, use_container_width=True)
        
        # 주식 지표 & 공식
        st.markdown('--------')
        st.write(f"### {selected_company} 주식 지표 📝")
        columns_ori = ['8월1일psr', '8월1일pbr2', '8월1일_s_return', '8월1일_l_return', '순매수강도(20일)', 's_growth', 'l_growth', 'roe', 'deb']
        new_columns = ['PSR', 'PBR', '단기수익률', '장기수익률', '순매수강도', '단기성장률', '장기성장률', 'ROE', '부채비율']
        data_ko = data[columns_ori].copy()
        data_ko.columns = new_columns
        selected_data = data_ko[data['name'] == selected_company]
        st.dataframe(selected_data)

        col1, col2 = st.columns([3,2])
        with col1: 
            st.write(" ")
            st.write(f"#####  ① 지표 해석", unsafe_allow_html=True)
            st.write("<span><strong>PSR:</strong> 기업의 매출액 대비 주가의 상대적인 가치를 나타내는 지표</span>", unsafe_allow_html=True)
            st.write("<span><strong>PBR:</strong> 기업의 순자산 대비 주가의 상대적인 가치를 나타내는 지표</span>", unsafe_allow_html=True)
            st.write("<span><strong>단기수익률:</strong> 주식 가격이 3개월 동안 얼마나 변했는지 나타내는 지표</span>", unsafe_allow_html=True)
            st.write("<span><strong>장기수익률:</strong> 주식 가격이 12개월 동안 얼마나 변했는지 나타내는 지표</span>", unsafe_allow_html=True)
            st.write("<span><strong>단기성장률:</strong> 기업의 매출액이 12개월 동안 얼마나 변했는지 나타내는 지표</span>", unsafe_allow_html=True)
            st.write("<span><strong>단기성장률:</strong> 기업의 매출액이 36개월 동안 얼마나 변했는지 나타내는 지표</span>", unsafe_allow_html=True)
            st.write("<span><strong>순매수강도:</strong> 매수와 매도 주문의 양을 비교하여 해당 종목의 강도나 흐름을 나타내는 지표</span>", unsafe_allow_html=True)
            st.write("<span><strong>ROE:</strong> 기업이 투자한 돈에 비해 얼마나 많은 이익을 얻을 수 있는지를 측정하는 도구</span>", unsafe_allow_html=True)
            st.write("<span><strong>부채비율:</strong> 기업의 재무건정성을 파악할 수 있는 지표</span>",  unsafe_allow_html=True)


        with col2:
            st.write(" ")
            st.write("##### ② 지표 계산방법")
            image_path1 = 'formula.jpeg'
            st.markdown(f'<h1 style="text-align: center; margin-top: -10px;"><img src="data:image/jpeg;base64,{base64.b64encode(open(image_path1, "rb").read()).decode()}" alt="Image" width="400" height="400" style="vertical-align: middle;"></h1>', unsafe_allow_html=True)
    st.markdown("------")
    
# Top 종목
def get_top_stocks(selected_option, num_top_stocks=5):
    stocks = fdr.StockListing("KRX")
    columns1 = ['Code', 'Name', 'Market', 'Changes', 'ChagesRatio', 'High', 'Volume']
    new_columns = ['종목코드', '기업명', '시장', '전일 대비 등락금액', '전일 대비 등락률', '고가', '거래량']
    stocks_ko = stocks[columns1].copy()
    stocks_ko.columns = new_columns

    if selected_option == '전일 대비 등락률':
        columns = ['종목코드', '기업명', '시장', '전일 대비 등락금액', selected_option]
    else:
        columns = ['종목코드', '기업명', '시장', selected_option]
    
    sorted_stocks = stocks_ko.sort_values(selected_option, ascending=False)
    sorted_stocks = sorted_stocks.loc[:, columns]
    
    return sorted_stocks[:num_top_stocks]


st.markdown('<h3 style="text-align: center;"> 😍 MY 관심종목 : 삼성전자 😍</h3>', unsafe_allow_html=True)
st.write('')
col1, col2 = st.columns(2)
### candle chart 도전!!!
with col1: 
    st.write(f'### 주가 차트')
    fig = go.Figure(data=[go.Candlestick(
        x=stock_data.index,
        open=stock_data['Open'],
        high=stock_data['High'],
        low=stock_data['Low'],
        close=stock_data['Close']
    )])     
    fig.update_layout(
        xaxis_title="날짜",
        yaxis_title="주가",
        xaxis_rangeslider_visible=False,
        
    )    
    st.plotly_chart(fig, use_container_width=True)
    

# 주식 지표 & 공식
with col2: 
    st.write(f"### 주식 지표")
    columns_ori = ['8월1일psr', '8월1일pbr2', '8월1일_s_return', '8월1일_l_return', '순매수강도(20일)', 's_growth', 'l_growth', 'roe', 'deb']
    new_columns = ['PSR', 'PBR', '단기수익률', '장기수익률', '순매수강도', '단기성장률', '장기성장률', 'ROE', '부채비율']
    data_ko = data[columns_ori].copy()
    data_ko.columns = new_columns
    selected_data = data_ko[data['name'] == com]
    selected_data.index = ['수치']
    st.dataframe(selected_data)
    st.write("##### 지표 계산방법")
    image_path1 = 'formula.jpeg'
    st.markdown(f'<h1 style="text-align: center; margin-top: -13px;"><img src="data:image/jpeg;base64,{base64.b64encode(open(image_path1, "rb").read()).decode()}" alt="Image" width="400" height="250" style="vertical-align: middle;"></h1>', unsafe_allow_html=True)
    

st.markdown("------")

with col1:
    def main():
        st.subheader("TOP 종목")
        
        selected_option = st.selectbox('선택 옵션', ['고가', '거래량', '전일 대비 등락률'])
        num_top_stocks = st.slider('상위 종목 수', 1, 10, 5)
        
        
        if selected_option == '고가':
            title = "고가 상위 종목"
        elif selected_option == '거래량':
            title = "거래량 상위 종목"
        else:
            title = "전일 대비 등락률 상위 종목"
        
        st.write(f"### {title}")
        
        top_stocks = get_top_stocks(selected_option, num_top_stocks)
        st.write(top_stocks)

    if __name__ == "__main__":
        main()

# 오늘의 증시
with col2:
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.subheader("오늘의 증시")
    end_date = datetime.today()
    start_date = datetime(2023, 5, 1)
    kospi = fdr.DataReader("KS11", start_date, end_date)
    kosdaq = fdr.DataReader("KQ11", start_date, end_date)
    selected = st.selectbox("원하는 시장을 선택하세요", ['KOSPI', 'KOSDAQ'])
    
    if selected == 'KOSPI':
        st.write(" ##### 코스피 (KOSPI)")
        y = kospi['Open']
        chart = alt.Chart(y.reset_index()).mark_line().encode(
            x='Date:T',
            y=alt.Y('Open:Q', scale=alt.Scale(domain=[2350, 2750])),
            tooltip=['Date:T', 'Open:Q']
        ).properties(
            width=800,
            height=350
        ).configure_axis(
            labelFontSize=12,
            titleFontSize=14
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.write(" ##### 코스닥 (KOSDAQ)")
        y1 = kosdaq['Open']
        chart = alt.Chart(y1.reset_index()).mark_line().encode(
            x='Date:T',
            y=alt.Y('Open:Q', scale=alt.Scale(domain=[750, 1000])),
            tooltip=['Date:T', 'Open:Q']
        ).properties(
            width=800,
            height=350
        ).configure_axis(
            labelFontSize=12,
            titleFontSize=14
        )
        st.altair_chart(chart, use_container_width=True)
