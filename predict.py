import imp
import os
import math
import torch
import argparse
from sklearn import metrics
from utils import get_data_predict, getdata
from utils import get_data
from DSAKT import DSAKT, Encoder, Decoder
from SAKT import SAKT
import pickle

def predict(window_size:int, model_path:str, data_path:str):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu");
    pre_data,E = get_data_predict(data_path,window_size)
    N_val = pre_data.shape[1]
    count = 0
    unit_list_val = []
    for i in range(pre_data.shape[1]):
        for j in range(pre_data.shape[2]):
            if pre_data[0][i][j] !=0:
                count = count +1
        unit_list_val.append(count)
        count = 0


    model = torch.load(model_path);
    assert model.window_size == window_size;
    model.to(device);
    model.eval();

    with torch.no_grad():
        predict = model(pre_data[0].to(device), pre_data[1].to(device)).squeeze(-1).to("cpu");
        correctness = pre_data[2];
        
        pred = [];
        cort = [];
        for i in range(N_val):
            pred.extend(predict[i][0:unit_list_val[i]].cpu().numpy().tolist());
            cort.extend(correctness[i][0:unit_list_val[i]].numpy().tolist());
                
        pred = torch.Tensor(pred) > 0.5;
        cort = torch.Tensor(cort) == 1;
        acc = torch.eq(pred, cort).sum() / len(pred);
        
        pred = [];
        cort = [];
        for i in range(N_val):
            pred.extend(predict[i][unit_list_val[i]-1:unit_list_val[i]].cpu().numpy().tolist());
            cort.extend(correctness[i][unit_list_val[i]-1:unit_list_val[i]].numpy().tolist());
                
        rmse = math.sqrt(metrics.mean_squared_error(cort, pred));
        fpr, tpr, thresholds = metrics.roc_curve(cort, pred, pos_label=1);
        auc = metrics.auc(fpr, tpr);
        
        
        
        print('val_auc: %.3f mse: %.3f acc: %.3f' %(auc, rmse, acc));
        with open ('/content/predicts.pickle','wb') as file:
           pickle.dump({'preds':predict,'correct':correctness},file)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser();
    parser.add_argument("-ws", "--window_size", required=True);
    parser.add_argument("-d", "--data_path", required=True);
    parser.add_argument("-m", "--model_path", required=True);
    args = parser.parse_args();
    
    assert os.path.exists(args.data_path);
    assert os.path.exists(args.model_path);
    
    predict(int(args.window_size), args.model_path, args.data_path);
    