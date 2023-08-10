import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import re
from datetime import date
from krwordrank.word import KRWordRank
from wordcloud import WordCloud

# 요약
from typing import List
from konlpy.tag import Okt
from textrankr import TextRank

# 감성 분석
from argparse import Namespace
from sklearn.metrics import classification_report
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

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
# pip install krwordrank 
# pip install wordcloud 

#pip install textrankr 
#pip install konlpy
#pip install transformers


# <키워드 추출>
# Mecab 전처리 파일 사용 이후 
# 문장부호 및 불용어 제거 전처리

# <요약 및 감성 분석>
# 원문 데이터 전처리 


# <키워드 추출 및 워드클라우드 시각화 과정>
# 크롤링한 데이터 불러오기
df = pd.read_csv('final_clean_mecab.csv',index_col=0)

# 뉴스 기사를 검색할 날짜 선택
min_date = date(2023, 4, 11)  # 선택 가능한 가장 이른 날짜
max_date = date(2023, 7, 11)  # 선택 가능한 가장 늦은 날짜
default_date = date(2023, 7, 11)
input_date = st.date_input("뉴스 기사를 확인할 날짜를 선택해주세요.", min_value=min_date, max_value= max_date, value= default_date)
# 분장 부호 제거 함수
def remove_non_alphanumeric(text):
    # 한글과 영어를 제외한 모든 문자를 제거하는 정규표현식 패턴
    pattern = r'[^ㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z\s]'

    # 정규표현식을 사용하여 한글과 영어를 제외한 모든 문자를 제거하고 반환
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text

# DataFrame의 'cleaned_mecab' 열에 있는 모든 문자열에 대해 한글과 영어를 제외한 모든 문자를 제거
df['cleaned_mecab'] = df['cleaned_mecab'].apply(remove_non_alphanumeric)

# 글자수가 1개인 단어 제거 함수
def remove_single_character_words(text):
    # 글자수가 한 개인 단어를 제거하는 정규표현식 패턴
    pattern = r'\b\w\b'

    # 정규표현식을 사용하여 글자수가 한 개인 단어를 제거하고 반환
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text
# 띄어쓰기가 1개 이상인 부분을 단일 띄어쓰기로 변경하는 함수

def remove_extra_spaces(text):
    # 띄어쓰기가 1개 이상인 부분을 단일 띄어쓰기로 변경하는 정규표현식 패턴
    pattern = r'\s+'

    # 정규표현식을 사용하여 띄어쓰기가 1개 이상인 부분을 단일 띄어쓰기로 변경하고 반환
    cleaned_text = re.sub(pattern, ' ', text)
    return cleaned_text

def generate_wordcloud(num):
    # 테스트할 문장 설정
    text = df.loc[num, 'cleaned_mecab']

    # Substring graph를 만들기 위한 최소 등장 빈도수와 최대 길이 입력
    min_count = 3   # 단어의 최소 출현 빈도수 (그래프 생성 시)
    max_length = 10 # 단어의 최대 길이
    wordrank_extractor = KRWordRank(min_count=min_count, max_length=max_length)

    beta = 0.85    # PageRank의 decaying factor beta
    max_iter = 10  # 반복횟수 제한
    texts = text
    keywords, rank, graph = wordrank_extractor.extract(texts, beta, max_iter)

   
    # 왼쪽 열에 키워드 추출 결과를 출력
    col1, col2 = st.columns(2)
    keyword_df = pd.DataFrame(columns=['키워드', '비율 (상위 30개까지)'])
    i = 0

    # 키워드 추출 결과가 있을 때만 데이터프레임에 추가
    if keywords:
        with col1:
            st.write("키워드 추출 결과:")
            for word, r in sorted(keywords.items(), key=lambda x: x[1], reverse=True)[:30]:
                r_rounded = round(r, 2)  # 소수점 셋 째 자리에서 반올림
                keyword_df.loc[i] = [word, r_rounded]
                i += 1
            st.write(keyword_df)
        
        # 불용어 처리
        with open('koreanStopwords.txt', 'r', encoding='utf-8') as f:
            stopwords = [line.strip() for line in f]

        passwords = {word: score for word, score in sorted(
            keywords.items(), key=lambda x: -x[1])[:300] if not (word in stopwords)}

        # 키워드 추출 결과가 0개가 아닐 때 워드클라우드 생성
        if passwords:
            # 한글 폰트 사용
            from matplotlib import font_manager, rc
            font_path = "C:\Windows\Fonts/gulim.ttc" 
            font_name = font_manager.FontProperties(fname=font_path).get_name()
            rc('font', family=font_name)

            # 오른쪽 열에 워드클라우드 결과를 출력
            with col2:
                st.write('')
                krwordrank_cloud = WordCloud(
                    font_path=font_path,
                    width=400,
                    height=400,
                    background_color="white"
                )

                krwordrank_cloud.generate_from_frequencies(passwords)
                plt.figure(figsize=(5, 5))
                plt.imshow(krwordrank_cloud, interpolation="bilinear")
                plt.axis("off")
                st.pyplot(plt)
        else:
            with col2:
                st.write("키워드 추출 결과가 없습니다.")
    else:
        with col1:
            st.write("키워드 추출 결과가 없습니다.")


#<텍스트 요약 과정>
def remove_brackets(text):
    pattern = r'\[.*?\]|\(.*?\)|\<.*?\>'
    result = re.sub(pattern, '', text)
    return result

def remove_special_characters(text):
    pattern = r'[^\w\s.]'
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text

def process_sentences(text):
    pattern = r'\.'
    grouped_sentences = re.split(pattern, text)
    sentences = [sentence.strip() + '.' for sentence in grouped_sentences if sentence.strip()]
    return sentences

def remove_last_sentence(sentences):
    if sentences:
        del sentences[-1]
        del sentences[-1]
    return sentences


### 최종 전처리 코드
def processed():
    df = pd.read_csv('final_ss.csv', encoding='utf-8')

    ### 1. [],(),<> 기호와 안에 있는 필요없는 문자 제거하기
    df['main_text'] = df['main_text'].apply(remove_brackets)

    ### 2. 알파벳, 숫자, 밑줄, 공백, 그리고 '.' 문자를 제외한 모든 다른 문자 모두 삭제
    df['main_text'] = df['main_text'].apply(remove_special_characters)

    ### 3. 문장 이중리스트로 담고 '.' 기호 붙이기
    df['processed_text'] = df['main_text'].apply(process_sentences)

    ### 4. 모든 행에 대해 [0][-1]에 위치한 문장 삭제
    df['processed_text'] = df['processed_text'].apply(remove_last_sentence)

    ### 5.main_text의 ' '기호 삭제하기
    df['main_text'] = df['main_text'].apply(lambda x: x.replace("'", ""))

    ### 6. str 형태로 변환
    df['processed_text'] = df['processed_text'].str.join(' ')

    cleaned_data = df['processed_text']
    return cleaned_data



# 요약 코드
def textrank_summary(num, cleaned_data):

  class OktTokenizer:
      def __init__(self):
          self.okt = Okt()

      def __call__(self, text: str) -> List[str]:
          tokens: List[str] = self.okt.phrases(text)
          return tokens


  text = cleaned_data[num]

  mytokenizer = OktTokenizer()  # 인스턴스 생성
  textrank = TextRank(mytokenizer)

  k = 3  # num sentences in the resulting summary

  summaries = textrank.summarize(text, k, verbose=False)  # 요약 결과를 가져옴

  for summary in summaries:
      if summary.startswith(','):
          summary = summary[1:]  # 첫 번째 문자인 ','를 제외한 문자열로 재할당
      st.write(summary+".")

# 감성 분류 모델
def textrank_summary_bert(num, cleaned_data, model, tokenizer):
    class OktTokenizer:
        def __init__(self):
            self.okt = Okt()

        def __call__(self, text: str) -> List[str]:
            tokens: List[str] = self.okt.phrases(text)
            return tokens

    text = cleaned_data[num]

    mytokenizer = OktTokenizer()  # 인스턴스 생성
    textrank = TextRank(mytokenizer)

    k = 5  # num sentences in the resulting summary

    summaries = textrank.summarize(text, k, verbose=False)  # 요약 결과를 가져옴

    for i, summary in enumerate(summaries):
        if summary.startswith(','):
            summary = summary[1:]  # 첫 번째 문자인 ','를 제외한 문자열로 재할당
        summaries[i] = summary
    return summaries


def sentiment_analysis_summary(summaries, model, tokenizer):
    classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    combined_summary = " ".join(summaries)
    result = classifier(combined_summary)

    result[0]["score"] = "{:.3f}".format(result[0]["score"]) 
    if result[0]['label'] == 'positive': 
        result[0]['label'] = '긍정'
        st.write(f'##### ② 감성 분석 결과 : {result[0]["label"]} [정확도 : {result[0]["score"]}]')
        
    elif result[0]['label'] == 'negative': 
        result[0]['label'] = '부정'
        st.write(f'##### ② 감성 분석 결과 : {result[0]["label"]} [정확도 : {result[0]["score"]}]')
    else : 
        result[0]['label'] = '중립'
        st.write(f'##### ② 감성 분석 결과 : {result[0]["label"]} [정확도 : {result[0]["score"]}]')
        
########################################

if st.button('검색'):
    st.markdown("------")
    # 선택 날짜로 데이터 필터링
    # 데이터프레임의 'date' 컬럼 값과 입력한 날짜를 동일한 형식으로 비교
    selected_date_str = input_date.strftime('%Y.%m.%d')

    # 데이터프레임의 'date' 컬럼 값 형식 통일
    df['date'] = df['date'].str.replace(' ', '')
    df['date'] = pd.to_datetime(df['date'], format='%Y.%m.%d')

    # 데이터프레임에서 선택한 날짜와 일치하는 데이터 추출
    filtered_data = df[df['date'] == selected_date_str]
    filtered_data1 = filtered_data.copy()
    filtered_data1 = filtered_data1.rename(columns={'title': '제목','main_text': '내용'})
    st.write(f'### {input_date}  삼성전자 증권 뉴스 📰')
    st.write('')
    st.write(filtered_data1[['제목','내용']])
    st.markdown("------")

    # DataFrame의 'cleaned_mecab' 열에 있는 모든 문자열에 대해 글자수가 한 개인 단어를 제거
    df['cleaned_mecab'] = df['cleaned_mecab'].apply(remove_single_character_words)

    # DataFrame의 'cleaned_mecab' 열에 있는 모든 문자열에 대해 띄어쓰기가 1개 이상인 부분을 단일 띄어쓰기로 변경
    df['cleaned_mecab'] = df['cleaned_mecab'].apply(remove_extra_spaces)

    # 전처리 함수 적용
    df['cleaned_mecab'] = df['cleaned_mecab'].apply(remove_non_alphanumeric)
    df['cleaned_mecab'] = df['cleaned_mecab'].apply(remove_single_character_words)
    df['cleaned_mecab'] = df['cleaned_mecab'].apply(remove_extra_spaces)
    # 모든 행 list of str 형태로 담기
    df['cleaned_mecab'] = df['cleaned_mecab'].apply(lambda x: [x])

    # 키워드 추출 워드 클라우드 생성 함수

    # 모델 객체 생성
    tokenizer = AutoTokenizer.from_pretrained("snunlp/KR-FinBert-SC")
    model = AutoModelForSequenceClassification.from_pretrained("snunlp/KR-FinBert-SC")

    # 선택한 날짜의 뉴스에 관한 기능 최종 실행

    st.write(f"### {input_date}  삼성전자 증권 뉴스 분석 결과 🧐")
    st.write('')
    for index, row in filtered_data.iterrows():
        # 선택한 날자의 뉴스 기사 출력
        formatted_date = row['date'].strftime('%Y-%m-%d')  # 날짜를 원하는 형식으로 포맷팅
        st.write(f"##### ① 제목 :  {row['title']}")
        

        # 감성 분류
        st.write('')
        cleaned_data=processed()
        summaries = textrank_summary_bert(index, cleaned_data, model, tokenizer)
        sentiment_analysis_summary(summaries, model, tokenizer)
        st.write('')

        # 요약
        st.write(f"##### ③ 요약")
        data = processed()
        textrank_summary(index, data)
        
        # 키워드 추출 및 워드 클라우스 시각화
        st.write('')
        st.write(f"##### ④ 키워드 확인")
        generate_wordcloud(index)

        

        # 원문 하이퍼링크
        st.write('')
        st.write(f"##### ⑤ 원문 : [{row['link']}]({row['link']})")
        st.write("-" * 50)





st.markdown('<h3 style="text-align: center;">❗️ 뉴스 정보 요약 서비스 개요 설명 ❗️</h3>', unsafe_allow_html=True)
st.write('')
st.write('')
st.write('##### 1️⃣ 3줄 요약: 원하는 종목의 뉴스 제목 클릭 시 해당 기사의 본문을 3줄로 압축한 요약문을 제공합니다.')
st.write('')
st.write('##### 2️⃣ 핵심 키워드: 원하는 종목의 뉴스 제목 클릭 시 기사 요약과 함께 기사의 핵심 키워드를 보여줍니다.')
st.write('')
st.write('##### 3️⃣ 뉴스 기사 감성 분석: 뉴스 기사에 대해 긍정/부정/중립 세가지의 감정으로 나눈 뒤, 해당 뉴스가 어떤 감정의 반응을 보이는 뉴스인지 알 수 있는 감성점수를 제공합니다.')
st.write('-'*100)
