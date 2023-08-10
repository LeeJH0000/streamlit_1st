import pandas as pd
import numpy as np
import streamlit as st
import base64
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
# <struct>
# 한글 폰트 사용
# from matplotlib import font_manager, rc
# font_path = "C:\Windows\Fonts/gulim.ttc" 
# font_name = font_manager.FontProperties(fname=font_path).get_name()
# rc('font', family=font_name)

# 함수 호출 및 결과 확인

name = st.text_input('관심있는 종목명을 입력하세요.')

if st.button('검색'):
    st.markdown("------")
    if name:
        
        # 전처리 함수 
        def preprocess_data(file_name):
            # 데이터 불러오기
            origin_df = pd.read_csv(file_name, encoding='cp949')

            # 거래중지 기업 삭제
            origin_df = origin_df[origin_df['순매수강도(20일)'] != '거래중지']

            # 순매수강도 데이터 타입 변환
            origin_df['순매수강도(20일)'] = origin_df['순매수강도(20일)'].astype(float)

            # 종목 코드 앞에 붙은 'a' 삭제
            origin_df['tickers'] = origin_df['tickers'].str[1:]

            return origin_df

        # 함수 호출 및 결과 확인
        origin_df = preprocess_data('801_df_done.csv')

        # 원하는 종목에 대한 군집 생성
        def filtering_data(origin_df, target_ticker):
            # tickers가 target_ticker인 행 선택
            selected_rows = origin_df[origin_df['tickers'] == target_ticker]

            # target_group에 cluster 컬럼 값 할당
            target_group = selected_rows['cluster8월1일_6차와같은방식'].values

            # clustered_df 생성
            clustered_df = origin_df[origin_df['cluster8월1일_6차와같은방식'].isin(target_group)]

            # '수익률상위권2'이 1인 행만 추출하여 새로운 데이터프레임 생성
            # 수익률이 0 초과인 종목
            filtering_df = clustered_df[clustered_df['수익률상위권2'] == 1].copy()

            return filtering_df


        name_code = origin_df[origin_df['name'] == name]['tickers'].values
        name_code = name_code[0]
        filtered_df = filtering_data(origin_df, name_code)
            
        # 점수 계산 함수
        ### psr 스코어
        # psr_score 함수 정의
        def calculate_psr_score(psr_value):
            if 0 < psr_value <= 0.352429:
                return 5
            elif 0.352429 < psr_value <= 0.922929:
                return 4
            elif 0.922929 < psr_value <= 2.419488:
                return 3
            elif 2.419488 < psr_value <= 5.5200765:
                return 2
            elif 5.5200765 < psr_value:
                return 1
            elif psr_value == 0:
                return 0
            else:
                return None


        ### pbr 스코어
        # pbr_score 함수 정의
        def calculate_pbr_score(pbr_value):
            if 0 < pbr_value <= 0.539359:
                return 5
            elif 0.539359 < pbr_value <= 1.060385:
                return 4
            elif 1.060385 < pbr_value <= 2.144504:
                return 3
            elif 2.144504 < pbr_value <= 4.5522215:
                return 2
            elif 4.5522215 < pbr_value:
                return 1
            elif pbr_value == 0:
                return 0
            else:
                return None

        ### s_growth 스코어
        # s_growth_score 함수 정의
        def calculate_s_growth_score(s_growth_value):
            if 0.693270 < s_growth_value :
                return 5
            elif 0.285380 < s_growth_value <= 0.693270:
                return 4
            elif 0.133778 < s_growth_value <=0.285380:
                return 3
            elif 0.013453 < s_growth_value <= 0.133778:
                return 2
            elif s_growth_value <= 0.013453:
                return 1
            else:
                return None


        ### l_growth 스코어
        # l_growth_score 함수 정의
        def calculate_l_growth_score(l_growth_value):
            if 0.419674 < l_growth_value :
                return 5
            elif 0.178690 < l_growth_value <= 0.419674:
                return 4
            elif 0.094675 < l_growth_value <= 0.178690:
                return 3
            elif 0.018034 < l_growth_value <= 0.094675:
                return 2
            elif l_growth_value <= 0.018034:
                return 1
            else:
                return None


        ### s_growth 스코어
        # s_growth_score 함수 정의
        def calculate_s_growth_score(s_growth_value):
            if 0.693270 < s_growth_value :
                return 5
            elif 0.285380 < s_growth_value <= 0.693270:
                return 4
            elif 0.133778 < s_growth_value <=0.285380:
                return 3
            elif 0.013453 < s_growth_value <= 0.133778:
                return 2
            elif s_growth_value <= 0.013453:
                return 1
            else:
                return None


        ### l_growth 스코어
        # l_growth_score 함수 정의
        def calculate_l_growth_score(l_growth_value):
            if 0.419674 < l_growth_value :
                return 5
            elif 0.178690 < l_growth_value <= 0.419674:
                return 4
            elif 0.094675 < l_growth_value <= 0.178690:
                return 3
            elif 0.018034 < l_growth_value <= 0.094675:
                return 2
            elif l_growth_value <= 0.018034:
                return 1
            else:
                return None
            
        ### s_return 스코어
        # s_return_score 함수 정의
        def calculate_s_return_score(s_return_value):
            if 0.3683035 < s_return_value :
                return 5
            elif 0.065116 < s_return_value <= 0.3683035:
                return 4
            elif -0.049239 < s_return_value <= 0.065116:
                return 3
            elif -0.137009 < s_return_value <= -0.049239:
                return 2
            elif s_return_value <= -0.137009:
                return 1
            else:
                return None


        ### l_return 스코어
        # l_return_score 함수 정의
        def calculate_l_return_score(l_return_value):
            if 0.0692075 < l_return_value :
                return 5
            elif 0.027683 < l_return_value <= 0.0692075:
                return 4
            elif 0.012692 < l_return_value <= 0.027683:
                return 3
            elif 0 < l_return_value <= 0.012692:
                return 2
            elif l_return_value <= 0:
                return 1
            else:
                return None

        ### nbs_fi 스코어
        # nbs_score 함수 정의
        def calculate_nbs_score(nbs_value):
            if 0.146663 < nbs_value:
                return 5
            elif 0.029732 < nbs_value <= 0.146663:
                return 4
            elif -0.005576 < nbs_value <= 0.029732:
                return 3
            elif -0.048222 < nbs_value <= -0.005576:
                return 2
            elif nbs_value <= -0.048222:
                return 1
            else:
                return None


        ### roe 스코어
        # roe_score 함수 정의
        def calculate_roe_score(roe_value):
            if 30.955 < roe_value:
                return 5
            elif 20 < roe_value <= 30.955:
                return 4
            elif 10.75 < roe_value <= 20:
                return 3
            elif 4.46 < roe_value <= 10.75:
                return 2
            elif roe_value <= 4.46:
                return 1
            else:
                return None


        ### deb 스코어
        # deb_score 함수 정의
        def calculate_deb_score(deb_value):
            if  deb_value <= 32.45:
                return 5
            elif 32.45 < deb_value <= 80:
                return 4
            elif 80 < deb_value <= 100:
                return 3
            elif 100 < deb_value <= 200:
                return 2
            elif 200 < deb_value:
                return 1
            else:
                return None


        # 점수 계산 데이터프레임 생성

        # 각 컬럼들의 값 설정
        def calculate_scores(filtered_df):
            columns = ['8월1일psr', '8월1일pbr2', 's_growth', 'l_growth', '8월1일_s_return', '8월1일_l_return',
                    '순매수강도(20일)', 'roe', 'deb']
            functions = [calculate_psr_score, calculate_pbr_score, calculate_s_growth_score,
                        calculate_l_growth_score, calculate_s_return_score,
                        calculate_l_return_score, calculate_nbs_score, calculate_roe_score, calculate_deb_score]

            for col, func in zip(columns, functions):
                new_column_name = f'{col}_score'
                filtered_df[new_column_name] = filtered_df[col].apply(func)

            total_sum_columns = [f'{col}_score' for col in columns]
            filtered_df['total_score'] = filtered_df[total_sum_columns].sum(axis=1) / 10

            filtered_df = filtered_df.sort_values(by='total_score', ascending=False)

            return filtered_df

        # 함수 호출 및 결과 확인
        result_df = calculate_scores(filtered_df)
        result_df.index = range(1,len(result_df)+1)


        # 사용자 검색 종목 순위 정보
        find_df = origin_df[origin_df['name'] == name]
        find_df = calculate_scores(find_df)
        find_df = find_df.rename(columns={'sector':'업종명', 'tickers':'종목코드','name':'종목명',
                                        '3주후 수익률':'3주 수익률','total_score':'종합점수',
                                        '8월1일psr':"PSR", '8월1일pbr2':'PBR', 's_growth':'단기성장률', 'l_growth':'장기성장률', '8월1일_s_return':'단기수익률', '8월1일_l_return':'장기수익률',
                                        '순매수강도(20일)': '순매수강도', 'roe':'ROE', 'deb':'부채비율'})
        
        st.write('### 지표 자세히 살펴보기 📊')
        st.write('')
        st.write(f'##### 내가 선택한 {name} 종목의 재무제표 점수')
        find_df.index = ['점수']
        st.dataframe(find_df[['PSR','PBR','순매수강도','ROE','부채비율','단기성장률','장기성장률','단기수익률','장기수익률','3주 수익률','종합점수']])

        
        # 군집 종목 순위 정보
        finds_df = result_df.loc[:,['sector','tickers','name','3주후 수익률','total_score']]
        finds_df = finds_df.rename(columns={'sector':'업종명', 'tickers':'종목코드','name':'종목명','3주후 수익률':'3주 수익률','total_score':'종합점수'})
        
        
        # 사용자 옵션 선택
        st.markdown("------")
        st.write(f'### {name}과 유사한 이런 종목은 어떠세요? 😊')
        st.write('')
        option = st.selectbox('출력 개수', [10, 50, 100, 'All'])

        # 선택된 옵션에 따라 데이터프레임 출력
        st.write('')
        st.write(f'##### 3주간 수익률이 +이며, {name}과 같은 군집에 속한 종목 📊')
        st.write(f'###### ({name}와 같은 군집에 속한 종목은 총 {len(finds_df)}개 입니다.)')
        col1, col2 = st.columns([3,2])
        with col1: 
            if option == 'All':
                st.write('')
                st.write(finds_df)
            else:
                st.write('')
                st.write(finds_df.head(option))
        with col2:
            image_path = 'score.jpeg'
            st.markdown(f'<h3 style="text-align: center; margin-top: 13px;"><img src="data:image/jpeg;base64,{base64.b64encode(open(image_path, "rb").read()).decode()}" alt="Image" width="400" height="380" style="vertical-align: middle;"></h3>', unsafe_allow_html=True)

       
        st.write('-'*100)
st.markdown('<h3 style="text-align: center;">❗️ 주식 종목 추천 서비스 개요 설명 ❗️</h3>', unsafe_allow_html=True)
st.write('')
st.write('')
st.write('##### 1️⃣ 관심있는 종목 입력 시 입력받은 관심 종목을 기반으로 유사 종목 추천 서비스를 제공합니다.')
st.write('')
st.write('##### 2️⃣ 관심 종목을 입력한 후, 유사한 군집에 속한 종목 중 현재 날짜를 기준으로 수익률이 양수에 속한 종목 중 재무점수가 높은 순서대로 추천합니다.')
st.write('')
st.write('##### 3️⃣ 가치, 성장, 모멘텀, 퀄리티를 대표하는 9가지 지표(PSR, PBR, 단기/장기수익률, 단기/장기성장률, 순매수강도, ROE, 부채비율)를 각각 설정된 점수기준을 기반으로 종합점수(5점 만점)을 부여합니다.')
st.write('-'*100)

        

     

# 함수 종류 및 정의

# 전체 주식 종목에 대한 군집화 결과가 담긴 데이터 전처리 함수
# origin_df = preprocess_data('801_df_done.csv')

# 검색한 종목 명과 같은 군집에 속한 주식 종목 추출 함수
# filtered_df = filtering_data(origin_df, '005930')

# 9개 지표의 IQR 기반 스코어를 계산하는 함수
# calculate_psr_score, calculate_pbr_score, calculate_s_growth_score, calculate_l_growth_score ,
# calculate_s_return_score,calculate_l_return_score,
# calculate_nbs_score,calculate_roe_score,calculate_deb_score

# 각 지표별 점수를 표준화하여 최종 5점 만점 기준의 점수를 계산하고, 최종 데이터프레임 반환 함수
# calculate_scores(filtered_df)

 
