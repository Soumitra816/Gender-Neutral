import numpy as np
import argparse, time, pickle
import torch
import numpy
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report
from model import RMS_Fourier, MaskedNLLLoss,MaskedNLLLoss_hu
from dataloader import *
from functions import Conv
from model import RMS_Fourier, MaskedNLLLoss,Trans
import config
from sklearn.model_selection import train_test_split



def collate_batch(batch):
    text,label,labels,umask = list(),list(),list(),list()
    for t,l,mask in batch:
        text.append(t)
    
        label.append(l)
        
        labels.append(labels)
        umask.append(mask)
    
    return pad_sequence(text, padding_value=2.0), pad_sequence(acus, padding_value=2.0),\
    pad_sequence(vid, padding_value=2.0),pad_sequence(label,True),pad_sequence(umask, padding_value=2),pad_sequence(labels,True)

def get_IEMOCAP_loaders(path1,path2, batch_size=32,  num_workers=0, pin_mlabelsry=False):
    trainset = load_dataset(path1)
    #validset = load_dataset(path2)
    testset  = load_dataset(path2)
    print(len(trainset))
    lengths = [int(len(trainset)*0.8), int(len(trainset)*0.2)]
    trainset, validset = torch.utils.data.random_split(list(trainset), lengths)

   
    train_loader = DataLoader(list(trainset),
                              batch_size=batch_size,
                              collate_fn=collate_batch,
                              num_workers=num_workers,
                              pin_mlabelsry=pin_mlabelsry)
    valid_loader = DataLoader(list(validset),
                              batch_size=batch_size,
                              collate_fn=collate_batch,
                              num_workers=num_workers,
                              pin_mlabelsry=pin_mlabelsry)
    test_loader = DataLoader(list(testset),
                             batch_size=batch_size,
                             collate_fn=collate_batch,
                             num_workers=num_workers,
                             pin_mlabelsry=pin_mlabelsry)

    return train_loader,valid_loader,test_loader

def train_or_eval_model(model, loss_function, dataloader, epoch, optimizer=None, train=False):
    losses = []
    preds_labels = []
    preds_sent = []
    labels = []
    _labels=[]
    labels_labels=[]
    masks = []
    se_mat=torch.rand(10, 2524).cuda()
    alphas, alphas_f, alphas_b, vids = [], [], [], []
    count=0
    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()

    for i, data in enumerate(dataloader):
        
        if train:
            optimizer.zero_grad() 
        textf, label,umask = data[0].cuda(),data[1].cuda(),data[2].permute(1,0).cuda()
        
        
        log_prob_label, alpha, alpha_f, alpha_b,hidden = model(textf, umask)
        log=torch.cat((log_prob_label,hidden), dim=-1)
        print("I am hidden",log.shape)
       
        
      

        lp_sen = log_prob_.transpose(0, 1).contiguous().view(-1, log_prob_.size()[2])
        # lp_hu = hu.transpose(0, 1).contiguous().view(-1, log_prob_.size()[2])
        lp_labels = log_prob_labels.transpose(0, 1).contiguous().view(-1, log_prob_labels.size()[2])
        # print(lp_)
        

        
        

        labels_ = log_prob_.view(-1,3)
        
        labels_labels= log_prob_labels.view(-1,7)
        
        
    
        lp_ = log_prob.transpose(0, 1).contiguous().view(-1, log_prob.size()[2])
        # print(lp_)
        

        
        labels_ = hu.view(-1)
       
        
    
        
        #loss_hu = loss_hu(lp_, labels_)

        # loss = loss_function(lp_sen,lp_labels, labels_,labels_labels, umask)

        # lp_se = torch.argmax(lp_sen, 1) 
        # lp_em = torch.argmax(lp_labels, 1)
        # preds.append(lp_.data.cpu().numpy())
        
        # #labels.append(labels_.data.cpu().numpy())
       
        # masks.append(umask.reshape(-1).cpu().numpy())

        # losses.append(loss.item()*masks[-1].sum())
        # if train:
        #     loss.backward()
        #     if args.tensorboard:
        #         for param in model.named_parameters():
        #             writer.add_histogram(param[0], param[1].grad, epoch)
        #     optimizer.step()
        # else:
        #     alphas += alpha
        #     alphas_f += alpha_f
        #     alphas_b += alpha_b
        #     vids += data[-1]

        # if preds!=[]:
        #     preds_hu  = np.concatenate(preds)
        #     labels = np.concatenate(labels)
            
        # else:
        #     return float('nan'), float('nan'), [], [], [], float('nan'), []

        # avg_loss_hu = round(np.sum(losses)/np.sum(masks), 4)
        # preds_hu = np.array(preds) >0.4
        
        
        
        #print(lp_em)
        #se_mat
        

        # preds_hu.append(lp_se.data.cpu().numpy())
        preds_sent.append(lp_se.data.cpu().numpy())
        preds_labels.append(lp_em.data.cpu().numpy())
        labels_sent = torch.argmax(labels_, 1) 
        labels_em = torch.argmax(labels_labels, 1)
        
        _labels.append(labels_sent.data.cpu().numpy())
        labels_labels.append(labels_em.data.cpu().numpy())
       
        masks.append(umask.reshape(-1).cpu().numpy())
     

        losses.append(loss.item()*masks[-1].sum())
        if train:
            loss.backward()
            if args.tensorboard:
                for param in model.named_parameters():
                    writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step()
        else:
            alphas += alpha
            alphas_f += alpha_f
            alphas_b += alpha_b
            vids += data[-1]

    if preds_labels!=[]:
        preds_labels  = np.concatenate(preds_label)
        labels_labels = np.concatenate(labels)
        
        

        masks  = np.concatenate(masks)
    
    else:
        return float('nan'), float('nan'), [], [], [], float('nan'), []

    avg_loss_hu = round(np.sum(losses)/np.sum(masks), 4)
    avg_loss_hu = round(np.sum(losses)/np.sum(masks), 4)
    preds_hu = np.array(preds) >0.4
   
    avg_accuracy_hu = round(accuracy_score(labels, preds, sample_weight=masks)*100, 2)
    avg_fscore_hu = round(f1_score(labels, preds, sample_weight=masks, average='weighted')*100, 2)
    

   
 
    avg_accuracy_labels = round(accuracy_score(labels, preds_label, sample_weight=masks)*100, 2)

    
    avg_fscore_labels = round(f1_score(labels_labels, preds_labels, sample_weight=masks, average='weighted')*100, 2)
   
    avg_accuracy_sent = round(accuracy_score(_labels, preds_sent, sample_weight=masks)*100, 2)
  
    avg_fscore_sent = round(f1_score(_labels, preds_sent, sample_weight=masks, average='weighted')*100, 2)
    
    return avg_fscore_hu,avg_accuracy_hu,avg_loss, avg_accuracy_labels,avg_accuracy_sent, labels_labels, preds_labels,_labels,preds_sent, masks, avg_fscore_labels,avg_fscore_sent, [alphas, alphas_f, alphas_b, vids]

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')
    parser.add_argument('--l2', type=float, default=0.0001, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--dropout', type=float, default=0.06, metavar='dropout', help='dropout rate')
    parser.add_argument('--batch-size', type=int, default=64, metavar='BS', help='batch size')
    parser.add_argument('--epochs', type=int, default=1211160, metavar='E', help='number of epochs')
    parser.add_argument('--class-weight', action='store_true', default=False, help='use class weight')
    parser.add_argument('--attention', action='store_true', default=False, help='use attention on top of lstm')
    parser.add_argument('--tensorboard', action='store_true', default=False, help='Enables tensorboard log')
    args = parser.parse_args()

    print(args)

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    if args.tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter()

    batch_size = args.batch_size
    cuda       = args.cuda
    n_epochs   = args.epochs
    
    n_classes  = 2
    D_e = 300
    D_h = 100
    # h_m=

    # model = LSTMModel(D_m, D_e, D_h,
    #                   n_classes=n_classes,
    #                   dropout=args.dropout,
    #                   attention=args.attention)
    model = RMS_Fourier(D_m,D_e,D_h,attention=args.attention)
    model1=Trans(D_e+D_h)
    
    if cuda:
        model.cuda()
        model1.cuda()
        
    loss_weights = torch.FloatTensor([ 1.0,  1.0])
    
    if args.class_weight:
        loss_function  = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)
        loss_hu=MaskedNLLLoss_hu(loss_weights.cuda() if cuda else loss_weights)
    else:
        loss_function = MaskedNLLLoss()
        loss_hu=MaskedNLLLoss_hu()
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.l2)

    train_loader,valid_loader,test_loader = get_loaders('train_output.pkl','test_output.pkl',batch_size=batch_size)
  

    best_loss, best_labels,  best_mask = None, None, None

    for e in range(n_epochs):
        start_time = time.time()
        print("hi there")
        train_loss, train_acc_labels,train_acc_sent, _, _, _, _,_,train_fscore_labels,train_fscore_sen, _ = train_or_eval_model(model, loss_function,
                                               train_loader, e, optimizer, True)
        valid_loss, valid_acc_labels,valid_acc_sent, _, _, _, _,_,val_fscore_labels,val_fscore_sen, _ = train_or_eval_model(model, loss_function, valid_loader, e)
        test_loss, test_acc, test_label, test_pred, test_mask, test_fscore, attentions = train_or_eval_model(model, loss_function, test_loader, e)

        if best_loss == None or best_loss > test_loss:
             best_loss, best_label, best_pred, best_mask, best_attn =\
                     test_loss, test_label, test_pred, test_mask, attentions

        if args.tensorboard:
            writer.add_scalar('test: accuracy/loss', test_acc/test_loss, e)
            writer.add_scalar('train: accuracy/loss', train_acc/train_loss, e)
        print('epoch {} train_loss {} train_acc {} train_fscore {} valid_loss {} valid_acc {} val_fscore {} test_loss {} test_acc {} test_fscore {} time {}'.\
                 format(e+1, train_loss, train_acc_sent, train_fscore_sen, valid_loss, valid_acc_sent, val_fscore_sen,\
                         train_acc_labels, train_fscore_labels, valid_loss, valid_acc_labels, val_fscore_labels,test_loss, test_acc, test_fscore, round(time.time()-start_time, 2)))
    if args.tensorboard:
        writer.close()

    print('Test performance..')
    from sklearn.metrics import f1_score
    l=f1_score(best_label, best_pred, average='micro')
    print("i am everything",l)
    print("best label",best_label)
    print("best pred",best_pred)
    #print('Loss {} F1-score {}'.format(best_loss,round(f1_score(best_label, best_pred, sample_weight=best_mask, average='weighted')*100, 2)))
    print(classification_report(best_label, best_pred, sample_weight=best_mask, digits=4))
    print(confusion_matrix(best_label, best_pred, sample_weight=best_mask))

  

