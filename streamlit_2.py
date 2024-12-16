import pandas as pd
import json
import altair as alt
import plotly.express as px
import streamlit as st

# 페이지 설정: Wide 모드, 타이틀
st.set_page_config(layout="wide", page_title="대한민국 인구 대시보드", page_icon="🌏")

# 데이터셋 불러오기
df = pd.read_csv('DataSet/201412_202312_korea_population_year_preprocessed.csv')

# JSON 파일 불러오기
with open('korea_map.json', encoding="UTF-8") as f:
    korea_geojson = json.load(f)

# dark 테마 적용
alt.themes.enable("dark")

# 카테고리 리스트와 연도 리스트
category_list = list(df.category.unique())
year_list = list(df.year.unique())[::-1]

# 기본 Streamlit 인터페이스 설정
st.title("대한민국 인구 데이터 분석 대시보드")
st.sidebar.title("데이터 필터")

# 서브페이지 선택
page = st.sidebar.radio("페이지 선택", ["메인 페이지", "시계열 분석", "상관관계 분석"])


# 필터 설정
year = st.sidebar.selectbox("연도", year_list)
target = st.sidebar.selectbox("카테고리", category_list)

# 데이터 필터링
df_selected = df.query('year == @year & category == @target')

# ---- 메인 페이지 ----
if page == "메인 페이지":
    st.subheader(f"**{year}년 {target} 분석**")

    # 2개의 컬럼 레이아웃
    col1, col2 = st.columns([1, 2])

    # ---- 왼쪽 컬럼: 증감 비율과 테이블 ----
    with col1:
        # 인구 증감 분석 함수
        def calculate_population_difference(df, year, target):
            # 현재 연도와 이전 연도 데이터 필터링
            selected_year_data = df.query('year == @year & category == @target')[['city', 'code', 'population']].reset_index(drop=True)
            previous_year_data = df.query('year == @year - 1 & category == @target')[['city', 'code', 'population']].reset_index(drop=True)
    
            # 두 데이터프레임 병합 (left join)
            merged_data = pd.merge(
                selected_year_data,
                previous_year_data,
                on=['city', 'code'],
                suffixes=('_current', '_previous'),
                how='left'  # 왼쪽 기준 병합: 현재 연도의 데이터 우선
            )
    
            # population 차이 계산
            merged_data['population_diff'] = merged_data['population_current'].sub(
            merged_data['population_previous'], fill_value=0
            )
            merged_data['status'] = merged_data['population_diff'].apply(
            lambda x: '증가' if x > 0 else ('감소' if x < 0 else '유지')
            )
    
            return merged_data[['city', 'population_diff', 'status']].sort_values(by='population_diff', ascending=True)


        # 증감 분석
        df_popdiff_sorted = calculate_population_difference(df, year, target)

        # 상태별 비율 계산
        num_increase = len(df_popdiff_sorted[df_popdiff_sorted['status'] == '증가'])
        num_stable = len(df_popdiff_sorted[df_popdiff_sorted['status'] == '유지'])
        num_decrease = len(df_popdiff_sorted[df_popdiff_sorted['status'] == '감소'])
        total = len(df_popdiff_sorted)

        increase_ratio = round((num_increase / total) * 100, 1)
        stable_ratio = round((num_stable / total) * 100, 1)
        decrease_ratio = round((num_decrease / total) * 100, 1)

        # 상태별 색상 적용 함수
        def highlight_status(val):
            if val == '증가':
                return 'background-color: #27AE60; color: white;'
            elif val == '감소':
                return 'background-color: #E74C3C; color: white;'
            elif val == '유지':
                return 'background-color: #F39C12; color: white;'

        # 데이터프레임 스타일 적용 및 표시
        styled_df = df_popdiff_sorted.style.applymap(highlight_status, subset=['status'])
        st.subheader("증감 분석 데이터")
        st.dataframe(styled_df)

        # 도넛 차트 생성 함수
        def make_donut(input_response, input_text, input_color):
            if input_color == 'green':
                chart_color = ['#2ECC71', '#145A32']  # 증가 (초록색)
            elif input_color == 'orange':
                chart_color = ['#F39C12', '#7E5109']  # 유지 (주황색)
            elif input_color == 'red':
                chart_color = ['#E74C3C', '#922B21']  # 감소 (빨간색)
            else:
                chart_color = ['#A0A0A0', '#585858']  # 기본 회색

            source = pd.DataFrame({
                "Topic": ['', input_text],
                "% value": [100 - input_response, input_response]
            })

            plot = alt.Chart(source).mark_arc(innerRadius=45, cornerRadius=25).encode(
                theta="% value",
                color=alt.Color("Topic:N", scale=alt.Scale(domain=[input_text, ''], range=chart_color), legend=None),
            ).properties(width=150, height=150)

            text = alt.Chart(pd.DataFrame({'value': [f"{input_response}%"]})).mark_text(
                align='center', baseline='middle', color="white", fontSize=16
            ).encode(text='value:N')

            return plot + text


        # 도넛 차트 및 비율
        st.subheader("변동 비율")
        st.altair_chart(make_donut(increase_ratio, "증가", "green"), use_container_width=True)
        st.write(f"**증가 비율:** {increase_ratio}% ({num_increase}개 시도)")

        st.altair_chart(make_donut(stable_ratio, "유지", "orange"), use_container_width=True)
        st.write(f"**유지 비율:** {stable_ratio}% ({num_stable}개 시도)")

        st.altair_chart(make_donut(decrease_ratio, "감소", "red"), use_container_width=True)
        st.write(f"**감소 비율:** {decrease_ratio}% ({num_decrease}개 시도)")

    # ---- 오른쪽 컬럼: 지도와 히트맵 ----
    with col2:
        st.subheader("지도 시각화")
        choropleth = px.choropleth_mapbox(
            df_selected,
            geojson=korea_geojson,
            locations='code',
            featureidkey='properties.CTPRVN_CD',
            mapbox_style='carto-positron',
            zoom=5,
            center={'lat': 36.5, 'lon': 126.98},
            color='population',
            color_continuous_scale='matter',
            labels={'population': 'Population'}
        )
        st.plotly_chart(choropleth, use_container_width=True)

        st.subheader("히트맵")
        heatmap = alt.Chart(df).transform_aggregate(
            max_population='max(population)',
            groupby=['year', 'city']
        ).mark_rect().encode(
            y=alt.Y('year:O', title="Year"),
            x=alt.X('city:O', title="City"),
            color=alt.Color('max_population:Q', scale=alt.Scale(scheme="blueorange")),
            tooltip=[alt.Tooltip('year:O'), alt.Tooltip('max_population:Q')]
        ).properties(width=600, height=300)
        st.altair_chart(heatmap, use_container_width=True)

        # 데이터 분석 결과 텍스트
        st.write("""
        **데이터 분석 결과**  
        - 시간이 지날수록 점점 평균연령이 높아짐.
        - 성별 간의 평균 연령 차이가 시간이 지남에 따라 **일정하게 유지되는 경향**이 관찰됨.
        """)

# ---- 시계열 분석 페이지 ----
elif page == "시계열 분석":
    st.subheader("시계열 분석: 평균 연령, 총인구수, 기타 카테고리")

    # 컬럼 레이아웃 설정
    col1, col2 = st.columns(2)

    # ---- 1번 컬럼: 라인 그래프 ----
    with col1:
        st.subheader("카테고리별 시계열 라인 그래프")

        # 평균 연령, 총인구수 등 모든 카테고리의 평균 데이터 필터링
        categories = df['category'].unique()
        line_data = df.groupby(['year', 'category'])['population'].mean().reset_index()

        # Altair 멀티라인 그래프 생성
        line_chart = alt.Chart(line_data).mark_line(point=True).encode(
            x=alt.X('year:O', title='연도'),
            y=alt.Y('population:Q', title='Population (평균)', scale=alt.Scale(domain=[35, line_data['population'].max()])),
            color=alt.Color('category:N', legend=alt.Legend(title="카테고리")),
            tooltip=['year:O', 'category:N', 'population:Q']
        ).properties(width=700, height=400, title="카테고리별 평균 연령 시계열 변화")

        # 라인 그래프 표시
        st.altair_chart(line_chart, use_container_width=True)

         # 성별을 구분하기 위해 category 컬럼에서 '남자', '여자' 추출
        df['sex'] = df['category'].apply(lambda x: '남자' if '남자' in x else ('여자' if '여자' in x else '기타'))

        # 평균 연령 데이터 필터링
        avg_age_data = df[df['category'].str.contains('평균연령')].groupby(['year', 'sex'])['population'].mean().reset_index()

        # 데이터프레임 생성 및 표시
        st.markdown(f"### 성별 데이터프레임")
        avg_age_table = avg_age_data.pivot(index="year", columns="sex", values="population").reset_index()
        st.dataframe(avg_age_table.style.format("{:.2f}"), height=400)
        
        # 데이터 분석 결과 텍스트
        st.write("""
        **데이터 분석 결과**  
        - 여자 평균 연령은 남자 평균 연령보다 대체로 높게 형성함. 
        - 성별 간의 평균 연령 차이가 시간이 지남에 따라 **일정하게 유지되는 경향**이 관찰됨.
        """)

    # ---- 2번 컬럼: 데이터프레임 ----
    with col2:
        st.subheader("카테고리별 데이터프레임")

        for category in categories:
            # 카테고리별 평균 데이터 필터링
            category_data = df[df['category'] == category].groupby(['year'])['population'].mean().reset_index()

            # 데이터프레임 타이틀과 표시
            st.markdown(f"### {category}")
            st.dataframe(category_data)

            # 가로선 구분
            st.markdown("---")

# ---- 3번째 서브페이지: 남자와 여자의 평균 연령 상관관계 ----
elif page == "상관관계 분석":
    st.subheader("남자와 여자의 평균 연령 상관관계 분석")

    # 데이터 필터링: 남자와 여자 평균연령만 선택
    df_sex_avg_age = df[df['category'].isin(['남자 평균연령', '여자 평균연령'])]

    # 남자 평균연령과 여자 평균연령 데이터 추출
    df_sex_avg_age_pivot = df_sex_avg_age.pivot_table(
        index='year',
        columns='category',
        values='population',
        aggfunc='mean'
    ).reset_index()

    df_sex_avg_age_pivot.columns = ['Year', '남자 평균연령', '여자 평균연령']

    # 상관관계 분석
    correlation = df_sex_avg_age_pivot[['남자 평균연령', '여자 평균연령']].corr().iloc[0, 1]

    # 남자와 여자 평균연령의 최소/최대 값 확인
    x_min, x_max = df_sex_avg_age_pivot['남자 평균연령'].min() - 1, df_sex_avg_age_pivot['남자 평균연령'].max() + 1
    y_min, y_max = df_sex_avg_age_pivot['여자 평균연령'].min() - 1, df_sex_avg_age_pivot['여자 평균연령'].max() + 1

    # ---- 선형 회귀 그래프 (회귀선 추가) ----
    st.subheader("선형 회귀 그래프: 남자 vs 여자 평균 연령")

    # 회귀 분석을 위한 Altair 그래프 생성
    regression_chart = alt.Chart(df_sex_avg_age_pivot).mark_point(size=100).encode(
        x=alt.X('남자 평균연령:Q', title="남자 평균연령", scale=alt.Scale(domain=[x_min, x_max])),
        y=alt.Y('여자 평균연령:Q', title="여자 평균연령", scale=alt.Scale(domain=[y_min, y_max])),
        tooltip=['Year', '남자 평균연령', '여자 평균연령'],
        color=alt.value('#2E86C1')
    ).properties(width=700, height=500, title="남자와 여자 평균연령 간 상관관계")

    # 회귀선 추가 (선형 회귀)
    regression_line = alt.Chart(df_sex_avg_age_pivot).mark_line(color="red").encode(
        x='남자 평균연령:Q',
        y='여자 평균연령:Q'
    ).transform_regression('남자 평균연령', '여자 평균연령').properties(width=700, height=500)

    st.altair_chart(regression_chart + regression_line, use_container_width=True)

    # 상관관계 결과 출력
    st.markdown(f"### 상관계수: **{correlation:.2f}**")
    if correlation > 0.8:
        st.write("남자와 여자의 평균 연령은 강한 양의 상관관계.")
    elif correlation > 0.5:
        st.write("남자와 여자의 평균 연령은 중간 정도의 양의 상관관계.")
    else:
        st.write("남자와 여자의 평균 연령은 약한 상관관계.")

    # ---- 데이터프레임 ----
    st.subheader("연도별 남자와 여자 평균 연령")
    st.dataframe(df_sex_avg_age_pivot.style.format("{:.2f}"), height=400)

    # 데이터 분석 결과 요약
    st.markdown("""
    **데이터 분석 요약**
    - 남자와 여자의 평균 연령 간 상관관계 확인, 상관계수가 1로 강한 상관관계 확인할 수 있다.
    - 선형 회귀선과 산점도를 통해 두 변수 간의 관계를 직관적으로 확인할 수 있다.
    """)

    # ---- 행정구역별 남자와 여자의 평균 연령 상관관계 ----
    st.subheader("행정구역별 남자와 여자의 평균 연령 상관관계 분석")

    # 행정구역별 남자 평균연령과 여자 평균연령 데이터 추출
    df_sex_avg_age_region = df_sex_avg_age.pivot_table(
        index=['year', 'city'],
        columns='category',
        values='population',
        aggfunc='mean'
    ).reset_index()

    df_sex_avg_age_region.columns = ['Year', 'City', '남자 평균연령', '여자 평균연령']

    # 각 행정구역별 상관관계 계산
    correlation_per_region = df_sex_avg_age_region.groupby('City').apply(
        lambda x: x[['남자 평균연령', '여자 평균연령']].corr().iloc[0, 1]
    ).reset_index(name='상관계수')

    # ---- 행정구역별 상관관계 산점도 ----
    st.subheader("행정구역별 상관관계 산점도")
    fig = px.scatter(correlation_per_region, 
                     x='City', 
                     y='상관계수', 
                     color='상관계수', 
                     color_continuous_scale='Viridis', 
                     title='행정구역별 남자와 여자의 평균 연령 상관관계')
    
    st.plotly_chart(fig, use_container_width=True)


    # ---- 행정구역별 상관계수 막대 그래프 ----
    st.subheader("행정구역별 상관계수 막대 그래프")
    bar_chart = alt.Chart(correlation_per_region).mark_bar().encode(
        x='City:O',
        y='상관계수:Q',
        color='상관계수:Q',
        tooltip=['City:N', '상관계수:Q']
    ).properties(width=700, height=500, title="행정구역별 남자와 여자의 평균 연령 상관관계 막대 그래프")

    st.altair_chart(bar_chart, use_container_width=True)

    # 데이터 분석 결과 요약
    st.markdown("""
    **행정구역별 데이터 분석 요약**
    - 각 행정구역별 남자와 여자의 상관관계 확인.
    - 경상남도는 남녀 평균연령의 상관관계가 높은 것으로 확인.
    - 세종이 가장 낮고 서울,부산,제주도는 남녀 평균연령의 상관관계가 낮은 편으로 확인
    """)

    # 가로선 구분
    st.markdown("---")




