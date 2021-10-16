import os, random, torch, pickle
import numpy as np
import glob as gb
import torch.utils.data as Data
import torch.nn.functional as F
from torch import nn
from sklearn import metrics
from imblearn.under_sampling import RandomUnderSampler

seed = 10086
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed) 
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class GlobalMaxPool1d(nn.Module):
    """global max pooling"""
    def __init__(self):
        super(GlobalMaxPool1d, self).__init__()
    def forward(self, x):
         # x shape: (batch_size, channel, seq_len)
         # return shape: (batch_size, channel, 1)
        return F.max_pool1d(x, kernel_size=x.shape[2])

class TextCNN(nn.Module):
    """conv->relu->pool->dropout->linear->sigmoid"""
    def __init__(self, dropout_rate, embed_size, kernel_sizes, channel_nums):
        super(TextCNN, self).__init__()
        
        self.pool = GlobalMaxPool1d()
        self.dropout = nn.Dropout(dropout_rate)
        self.convs = nn.ModuleList()  
        for c, k in zip(channel_nums, kernel_sizes):
            self.convs.append(nn.Conv1d(in_channels = embed_size, 
                                        out_channels = c, 
                                        kernel_size = k))
        self.decoder = nn.Sequential(
            nn.Linear(sum(channel_nums), 1),
            nn.Sigmoid())
        
    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        encoding = torch.cat([self.pool(F.relu(conv(inputs))).squeeze(-1) for conv in self.convs], dim=1)
        outputs = self.decoder(self.dropout(encoding))
        return outputs


def load_feature(feature_type):
    feature_file = './data/features/' + feature_type + '.pkl'
    print(f'features get from {feature_file}.\n')
    with open(feature_file, 'rb') as f:
        data = pickle.load(f)
    X_train, X_test, y_train, y_test = data['X_train'],data['X_test'],data['y_train'],data['y_test']
    return X_train, X_test, y_train, y_test
 
def compute_metrics(all_trues, all_scores, threshold=0.5):
    all_preds = (all_scores >= threshold)

    acc = metrics.accuracy_score(all_trues, all_preds)
    pre = metrics.precision_score(all_trues, all_preds)
    rec = metrics.recall_score(all_trues, all_preds)
    f1 = metrics.f1_score(all_trues, all_preds)
    mcc = metrics.matthews_corrcoef(all_trues, all_preds)
    fpr, tpr, _ = metrics.roc_curve(all_trues, all_scores)
    AUC = metrics.auc(fpr, tpr)
    p, r, _ = metrics.precision_recall_curve(all_trues, all_scores)
    AUPR = metrics.auc(r, p)
    
    return acc, f1, pre, rec, mcc, AUC, AUPR

def train_epoch(model, train_iter, optimizer, loss):
    # Model on train mode
    model.train()
    
    all_trues = []
    all_scores = []
    losses, sample_num = 0.0, 0
    for batch_idx, (X , y) in enumerate(train_iter):
        sample_num += y.size(0)
        
        # Create vaiables
        with torch.no_grad():   
            X_var = torch.autograd.Variable(X.float())
            y_var = torch.autograd.Variable(y.float())
            
        # compute output
        output = model(X_var).view(-1)
        
        # calculate and record loss
        loss_batch = loss(output, y_var)
        losses += loss_batch.item()

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss_batch.backward()
        optimizer.step()
        
        all_trues.append(y_var.data.cpu().numpy())
        all_scores.append(output.data.cpu().numpy())

    all_trues = np.concatenate(all_trues, axis=0)
    all_scores = np.concatenate(all_scores, axis=0)
    acc, f1, pre, rec, mcc, AUC, AUPR = compute_metrics(all_trues, all_scores)
        
    return losses/sample_num, acc, f1, pre, rec, mcc, AUC, AUPR

def eval_epoch(model, eval_iter, loss):
    # Model on eval mode
    model.eval()

    all_trues = []
    all_scores = []
    losses, sample_num = 0.0, 0
    for batch_idx, (X , y) in enumerate(eval_iter):
        sample_num += y.size(0)
        
        # Create vaiables
        with torch.no_grad():
            X_var = torch.autograd.Variable(X.float())
            y_var = torch.autograd.Variable(y.float())

        # compute output
        output = model(X_var).view(-1)
        
        # compute loss and record loss
        loss_batch = loss(output, y_var)
        losses += loss_batch.item()

        all_trues.append(y_var.data.cpu().numpy())
        all_scores.append(output.data.cpu().numpy())

    all_trues = np.concatenate(all_trues, axis=0)
    all_scores = np.concatenate(all_scores, axis=0)

    acc, f1, pre, rec, mcc, AUC, AUPR = compute_metrics(all_trues, all_scores)
    
    return losses/sample_num, acc, f1, pre, rec, mcc, AUC, AUPR

def train(model, model_n, X_resampled, y_resampled, X_test, y_test, save, result_file, epoch_num, batch_size, lr):
    
    # Data loaders
    train_iter = Data.DataLoader(Data.TensorDataset(X_resampled, y_resampled), batch_size)
    test_iter = Data.DataLoader(Data.TensorDataset(X_test, y_test), batch_size)
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    loss = nn.BCELoss()
    
    # Train model
    patience = 25
    new_auc, count = .0, 0
    for epoch in range(epoch_num):
        train_loss, train_acc, train_f1, train_pre, train_rec, train_mcc, train_auc, train_aupr = train_epoch(
            model=model,
            train_iter=train_iter,
            optimizer=optimizer,
            loss=loss,
        )

        test_loss, acc, f1, pre, rec, mcc, auc, aupr = eval_epoch(
            model=model,
            eval_iter=test_iter,
            loss=loss,
        )

        res = '\t'.join([
                '\nEpoch [%d/%d]' % (epoch + 1, epoch_num),
                '\nTraining set',
                'loss:%0.5f' % train_loss,
                'accuracy:%0.6f' % train_acc,
                'f-score:%0.6f' % train_f1,
                'precision:%0.6f' % train_pre,
                'recall:%0.6f' % train_rec,
                'mcc:%0.6f' % train_mcc,
                'auc:%0.6f' % train_auc,
                'aupr:%0.6f' % train_aupr,
                '\nTesting set',
                'loss:%0.5f' % test_loss,
                'accuracy:%0.6f' % acc,
                'f-score:%0.6f' % f1,
                'precision:%0.6f' % pre,
                'recall:%0.6f' % rec,
                'mcc:%0.6f' % mcc,
                'auc:%0.6f' % auc,
                'aupr:%0.6f' % aupr,
            ])
        print(res)

        # Sava the model 
        if epoch>1 and train_auc > new_auc:
            count = 0
            new_auc = train_auc
            print("!!!new model was saved, testing set AUC:{:.6f}".format(auc))
            torch.save(model.state_dict(), os.path.join(save, 'model_{:0>2d}.pkl'.format(model_n)))   
        else:
            count += 1
            if count >= patience:
                return None
            
        # Start log
        with open(os.path.join(save, result_file), 'a') as f:
            if model_n == 1 and epoch == 0:
                f.truncate(0)
                f.write('model, epoch, train loss, train accuracy, train auc, test loss, accuracy, f-score, precision, recall, mcc, auc, aupr\n')
            # Log results
            if (epoch%10)-1 == 0 or count == 0:
                f.write('%d, %d, %0.6f, %0.6f, %0.6f, %0.6f, %0.6f, %0.6f, %0.6f, %0.6f, %0.6f, %0.6f, %0.6f\n' % (
                        model_n, (epoch + 1), train_loss, train_acc, train_auc, test_loss, acc, f1, pre, rec, mcc, auc, aupr))
         
         
def train_models(X_train, y_train, X_test, y_test, Model, k, epoch_num, batch_size, lr, save, result_file, train_device):
    # train_device = device
    n_model = 0
    
    # shuffle train data
    train_index = [i for i in range(len(y_train))]
    random.shuffle(train_index)
    X_train, y_train = X_train[train_index], y_train[train_index]
    
    shapes = X_train.shape
    while True:
        n_model += 1
        print(f'\n==================== Model {n_model}/{k} ====================')
        
        # use RandomUnderSampler to sample
        random_state = random.randint(1000,9999)
        rus = RandomUnderSampler(random_state=random_state)
        if len(shapes)==3:
            X_train = X_train.reshape(shapes[0], shapes[1]*shapes[2])
            X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
            X_resampled = X_resampled.reshape(-1, shapes[1], shapes[2])
            X_train = X_train.reshape(shapes[0], shapes[1], shapes[2])
        else:
            X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
        
        X_resampled, y_resampled, X_test, y_test = torch.tensor(X_resampled, device=device), torch.tensor(y_resampled, device=device), torch.tensor(X_test, device=device), torch.tensor(y_test, device=device)
        print(X_resampled.shape, y_resampled.shape, X_test.shape, y_test.shape)
        
        # initial weight 
        model = Model
        for m in model.modules():     
            if isinstance(m, (nn.Linear, nn.Conv1d)): 
                nn.init.xavier_normal_(m.weight.data)
                
        # train 
        train(model, n_model, X_resampled, y_resampled, X_test, y_test, save, result_file, epoch_num, batch_size, lr)
            
        if n_model>=k:
            break
    print(f'Get {k} base models.')
    

def ensemble_test(X_test, y_test, model, path_dir, result_file, models_num, threshold):
    X_test = torch.tensor(X_test).to(device)
    y_score_sum = [.0]*len(y_test)
    model_n = 0
    print('Load models in' + path_dir)
    for f in gb.glob(path_dir + '/*pkl'):
        model_n += 1
        model.load_state_dict(torch.load(f, map_location=device))
        model.eval()
        y_score = model(X_test).view(-1).cpu().data.numpy()
        y_score_sum += y_score
        
        acc, f1, pre, rec, mcc, AUC, AUPR = compute_metrics(y_test, y_score, threshold)
        
        # Start log
        with open(os.path.join(path_dir, result_file), 'a') as f:
            if model_n == 1:
                f.truncate(0)
                f.write('model_n, accuracy, f-score, precision, recall, mcc, auc, aupr\n')
            # Log results
            f.write('%d, %0.6f, %0.6f, %0.6f, %0.6f, %0.6f, %0.6f, %0.6f\n' % (model_n, acc, f1, pre, rec, mcc, AUC, AUPR))

    if model_n == models_num:
        y_score_avg = y_score_sum/models_num
        esb_acc, esb_f1, esb_pre, esb_rec, esb_mcc, esb_AUC, esb_AUPR = compute_metrics(y_test, y_score_avg, threshold)
        print(f'\nGot {models_num} models! Ensembled models loaded done!')
        res = '\t'.join([
                    'Evaluation results:\n'
                    'accuracy:%0.3f' % esb_acc,
                    'f-score:%0.3f' % esb_f1,
                    'precision:%0.3f' % esb_pre,
                    'recall:%0.3f' % esb_rec,
                    'mcc:%0.3f' % esb_mcc,
                    'auc:%0.3f' % esb_AUC,
                    'aupr:%0.3f' % esb_AUPR,
                ])
        print(res)
        
        # Log results
        with open(os.path.join(path_dir, result_file), 'a') as f:
            f.write('ensemble, %0.6f, %0.6f, %0.6f, %0.6f, %0.6f, %0.6f, %0.6f\n' % (esb_acc, esb_f1, esb_pre, esb_rec, esb_mcc, esb_AUC, AUPR))
            
    else:
        print(f'\nGot {model_n} models! Ensembled models loaded fail!')
        return 
   

if __name__=='__main__':
    
    # load data
    feature_type = 'pssm'
    X_train, X_test, y_train, y_test = load_feature(feature_type=feature_type)
    print('Data has generated.')
    
    # model 
    model_type = "TextCNN"
    embed_size = 20 if feature_type=='onehot' or feature_type=='pssm' else 100
    dropout_rate, kernel_sizes, channel_nums = 0.0, [5,9,13], [128,128,128]
    model = TextCNN(dropout_rate, embed_size, kernel_sizes, channel_nums).to(device)
    
    # param 
    threshold, models_num, epoch_num, batch_size, lr = 0.5, 17, 1000, 128, 0.001 
    path_dir = './saved_models'
    record_file ='record.csv'
    result_file ='results.csv'
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
    print('Models save in ' + path_dir)
    
    # train
    train_models(X_train, y_train, X_test, y_test, model, models_num, epoch_num, batch_size, lr, path_dir, record_file, device)
    
    # ensemble test
    ensemble_test(X_test, y_test, model, path_dir, result_file, models_num, threshold)
    
