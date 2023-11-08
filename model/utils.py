import os
import numpy as np
import torch
from transformers import BertModel
from model import BERTClassifier, c_BERTClassifier


bertmodel = BertModel.from_pretrained('skt/kobert-base-v1', return_dict=False)

def save_model(model_state, model_name):
    os.makedirs('./trained_model', exist_ok=True)
    torch.save(model_state, os.path.join('./trained_model', model_name))

def load_model(ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device)
    model = BERTClassifier(bertmodel,  dr_rate = 0.2).to(device)
    model.load_state_dict(checkpoint)
    print('model state: ', model.load_state_dict(checkpoint))
    model.eval()
    model.to(device)

    return model

def c_load_model(ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device)
    model = c_BERTClassifier(bertmodel,  dr_rate = 0.2).to(device)
    model.load_state_dict(checkpoint)
    print('model state: ', model.load_state_dict(checkpoint))
    model.eval()
    model.to(device)

    return model

def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc

def softmax(x):
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x)
    y = exp_x / sum_exp_x
    return y



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

NUM_TO_CATE = {
    0:'배송',
    1:'UX/UI 편의성',
    2:'고객센터',
    3:'상품 구색',
    4:'앱 오류',
    5:'가격&프로모션',
    6:'상품 품질',
    7:'정품 안전성',
    8:'만족도&기타',
    9:'상품 설명'
}