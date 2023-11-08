import pandas as pd

def star(x):
    if x == 3:
        x = 2 # 애매모호??
    elif x < 3:
        x = 0 # 부정
    elif x > 3:
        x = 1 # 긍정

    return x


CATE_TO_NUM = {
    '배송':0,
    'UX/UI 편의성':1,
    '고객센터':2,
    '상품 구색':3,
    '앱 오류':4,
    '가격&프로모션':5,
    '상품 품질':6,
    '정품 안전성':7,
    '만족도&기타':8,
    '상품 설명':9
}

# CATE_TO_NUM = {
#     '배송':0,
#     'UX/UI 편의성':1,
#     '고객센터':2,
#     '상품 구색':3,
#     '앱 오류':4,
#     '가격&프로모션':5,
#     '상품 품질':6,
#     '정품 안전성':6,
#     '만족도&기타':7,
#     '상품 설명':6
# }



def get_reviews(senti_data_dir):
    # senti_data_dir = '/content/drive/Othercomputers/내 컴퓨터/GitHub/자연어 처리/명품 커머스 플렛폼 분석/multi-label_classification/data/'

    balaan_g = pd.read_csv(senti_data_dir + 'balaan_google.csv', index_col=0)
    balaan_a = pd.read_csv(senti_data_dir + 'balaan_apple.csv')
    mustit_g = pd.read_csv(senti_data_dir + 'mustit_google.csv', index_col=0)
    mustit_a = pd.read_csv(senti_data_dir + 'mustit_apple.csv')
    trenbe_g = pd.read_csv(senti_data_dir + 'trenbe_google.csv', index_col=0)
    trenbe_a = pd.read_csv(senti_data_dir + 'trenbe_apple.csv')
    
    # 브렌드 추가
    balaan_g['brand'] = 'balaan'
    balaan_a['brand'] = 'balaan'
    mustit_g['brand'] = 'mustit'
    mustit_a['brand'] = 'mustit'
    trenbe_g['brand'] = 'trenbe'
    trenbe_a['brand'] = 'trenbe'

    # 날짜 형식 통일(년-월-일)
    balaan_a['DATE'] = pd.to_datetime(balaan_a['DATE']).dt.strftime('%F')
    mustit_a['DATE'] = pd.to_datetime(mustit_a['DATE']).dt.strftime('%F')
    trenbe_a['DATE'] = pd.to_datetime(trenbe_a['DATE']).dt.strftime('%F')

    balaan_g['date'] = balaan_g['date'].astype(str)
    mustit_g['date'] = mustit_g['date'].astype(str)
    trenbe_g['date'] = trenbe_g['date'].astype(str)
    balaan_g['date'] = pd.to_datetime(balaan_g['date']).dt.strftime('%F')
    mustit_g['date'] = pd.to_datetime(mustit_g['date']).dt.strftime('%F')
    trenbe_g['date'] = pd.to_datetime(trenbe_g['date']).dt.strftime('%F')



    # 컬럼 통일
    balaan_a.columns = ['user', 'date', 'star', 'like', 'title', 'review', 'brand']
    balaan_a = balaan_a[['brand', 'user', 'date', 'review', 'star']]

    mustit_a.columns = ['user', 'date', 'star', 'like', 'title', 'review', 'brand']
    mustit_a = mustit_a[['brand', 'user', 'date', 'review', 'star']]

    trenbe_a.columns = ['user', 'date', 'star', 'like', 'title', 'review', 'brand']
    trenbe_a = trenbe_a[['brand', 'user', 'date', 'review', 'star']]

    balaan_g.columns = ['date', 'dateYear', 'dateMonth', 'dateDay', 'star', 'user', 'review', 'brand']
    balaan_g = balaan_g[['brand', 'user', 'date', 'review', 'star']]

    mustit_g.columns = ['date', 'dateYear', 'dateMonth', 'dateDay', 'star', 'user', 'review', 'brand']
    mustit_g = mustit_g[['brand', 'user', 'date', 'review', 'star']]

    trenbe_g.columns = ['date', 'dateYear', 'dateMonth', 'dateDay', 'star', 'user', 'review', 'brand']
    trenbe_g = trenbe_g[['brand', 'user', 'date', 'review', 'star']]

    balaan_a['label'] = balaan_a['star'].apply(lambda x: star(x))
    mustit_a['label'] = mustit_a['star'].apply(lambda x: star(x))
    trenbe_a['label'] = trenbe_a['star'].apply(lambda x: star(x))
    balaan_g['label'] = balaan_g['star'].apply(lambda x: star(x))
    mustit_g['label'] = mustit_g['star'].apply(lambda x: star(x))
    trenbe_g['label'] = trenbe_g['star'].apply(lambda x: star(x))

    data = pd.concat([balaan_a, mustit_a, trenbe_a, balaan_g, mustit_g, trenbe_g])

    data['review'] = data['review'].str.replace('[^ㄱ-ㅎㅏ-ㅣ가-힣 ]', '')

    
    return data


def get_preprocessing_reviews(data_dir):
    data_1 = pd.read_excel(data_dir + 'reviews_4999.xlsx', index_col=0)
    data_2 = pd.read_excel(data_dir + 'reviews_7499.xlsx', index_col=0)
    data_3 = pd.read_excel(data_dir + 'reviews_9999.xlsx', index_col=0)
    data_4 = pd.read_excel(data_dir + 'reviews_12499.xlsx', index_col=0)
    data_5 = pd.read_excel(data_dir + 'reviews_14845.xlsx', index_col=0)
    data_6 = pd.read_excel(data_dir + 'reviews_2499.xlsx', index_col=0)

    data = pd.concat([data_1, data_2, data_3, data_4, data_5, data_6], ignore_index=True)
    data.dropna(how='any', inplace=True)


    # 날짜 형식 통일(년-월-일)
    data['date'] = data['date'].astype(int)
    data['date'] = data['date'].astype(str)
    data['date'] = pd.to_datetime(data['date']).dt.strftime('%F')


    data['cate_label'] = data['category'].apply(lambda x: CATE_TO_NUM[x])

    senti_data = []
    for re, la in zip(data['review'], data['label']):
        tmp = []
        tmp.append(re)
        tmp.append(la)
        senti_data.append(tmp)

    cate_data = []
    for re, la in zip(data['review'], data['cate_label']):
        tmp = []
        tmp.append(re)
        tmp.append(la)
        cate_data.append(tmp)

    
    return senti_data, cate_data, data
