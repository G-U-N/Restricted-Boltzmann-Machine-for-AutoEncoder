import torchvision
from matplotlib import pyplot as plt
from utils import parse_option, set_gpu, PrepareFunc, set_seeds, cal_debias_al_acc, debias_dataloader2tensor
import sys
import datetime
import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
import torch.nn.functional as F
warnings.filterwarnings("ignore")

YOUR_STUDENT_ID = "YOUR_STUDENT_ID"

def train_rbm(model, train_loader, rbm_models, criterion, args):
    print("Begin training..")
    for epoch in range(args.rbm_epoch):
        epoch_loss = 0
        for idx, (x, _) in enumerate(train_loader):
            x = x.view(x.shape[0], -1).to(torch.device('cuda'))
            model.contrastive_divergence(rbm_models.v2h(x), args.lr_rbm)
            loss = criterion (rbm_models.h2v(model.v2h2v(rbm_models.v2h(x))),x)
            epoch_loss += loss.item()
        print(f'Epoch {epoch} Loss: {epoch_loss:.4f}.')
    print("Completed.")

def validate_loaded_ae(ae_model, rbm_models, train_loader):
    for idx, (x, _) in enumerate(train_loader):
        x = x.view(x.shape[0], -1).to(torch.device('cuda'))
        print(torch.norm(rbm_models.v2h2v(x) - ae_model(x)))

def train_ae(model, train_loader, criterion, optimizer, args):
    for epoch in range(args.max_epoch):
        epoch_loss = 0
        for idx, (x, _) in enumerate(train_loader):
            x = x.view(x.shape[0], -1).to(torch.device('cuda'))

            loss = criterion(model(x),x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        print(f'Epoch {epoch} Loss: {epoch_loss/len(train_loader):.4f}.')

def val_ae(model, test_loader, prefix=None, is_raw=False,pca=None):
    from sklearn.linear_model import LogisticRegression
    hidden, label = [], []
    for idx, (x, y) in enumerate(test_loader):
        if x.shape[1] == 3:
            x = x.mean(dim=1, keepdim=True)
        if isinstance(y, list):
            y = y[0]
        x = x.view(x.shape[0], -1).to(torch.device('cuda'))
        if is_raw:
            hidden.append(x.detach().cpu())
        else:
            hidden.append(model.encoder(x).detach().cpu())
        label.append(y.cpu())
    hidden_np = torch.cat(hidden).numpy()
    if  pca is not None:
        hidden_np=pca.transform(hidden_np)
    hidden_np = StandardScaler().fit_transform(hidden_np)
    label_np = torch.cat(label).numpy()
    clf = LogisticRegression(max_iter=300)
    clf.fit(hidden_np, label_np)
    test_acc = clf.score(hidden_np, label_np)
    print(f'Test Accuracy: {test_acc}.')
    if prefix is not None:
        with open(f'./bonus/{YOUR_STUDENT_ID}.csv', 'a') as f:
            f.write(f'{prefix}: {test_acc}\n')

def tsne_ae(model, cur_loader, file_name='', is_raw=False,pca=None):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from sklearn import manifold
    Axes3D

    sampled_num = 10 * 200
    hidden, label = [], []
    for idx, (x, y) in enumerate(cur_loader):
        if x.shape[1] == 3:
            x = x.mean(dim=1, keepdim=True)
        if isinstance(y, list):
            y = y[0]
        x = x.view(x.shape[0], -1).to(torch.device('cuda'))
        if is_raw:
            hidden.append(x.detach().cpu())
        else:
            hidden.append(model.encoder(x).detach().cpu())
        label.append(y.cpu())
    hidden_np = torch.cat(hidden).numpy()
    if pca is not None:
        hidden_np=pca.transform(hidden_np)
    label_np = torch.cat(label).numpy()
    sampled_idx = np.random.choice(hidden_np.shape[0], sampled_num, replace=False)
    X, y = hidden_np[sampled_idx], label_np[sampled_idx]
    t_SNE_method = manifold.TSNE(n_components=2, init='pca', random_state=929)
    trans_X = t_SNE_method.fit_transform(X)
    plt.scatter(trans_X[:, 0], trans_X[:, 1], s=15, c=y, alpha=.4)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f'bonus/t-sne-{file_name}.png')
    plt.clf()

def val_extract_bias_conflicting(model, train_loader):
    model.eval()
    hidden, label, label_bias, query_label = [], [], [], []
    for _, (x, (y, y_bias)) in enumerate(train_loader):
        
        cur_idx = (y == 1)
        if cur_idx.sum() == 0:
            continue
        x, y, y_bias = x[cur_idx], y[cur_idx], y_bias[cur_idx]
        hidden.append(x.detach().cpu())
        label.append(y.cpu())
        label_bias.append(y_bias.cpu())
        query_label.append((y != y_bias).long().cpu())
    hiddens,labels,label_biases,query_labels=torch.cat(hidden), torch.cat(label), torch.cat(label_bias), torch.cat(query_label)
    hiddens=hiddens.to(torch.device("cuda"))
    labels=labels.to(torch.device("cuda"))
    losses=F.cross_entropy(model(hiddens),labels,reduce=False)
    losses=losses.detach().cpu().numpy().tolist()
    bias_idx=np.argsort(losses).tolist()[::-1][:1000]
    query_idx=bias_idx
    labels=labels.detach().cpu()
    print(f'Test Accuracy: {cal_debias_al_acc(query_idx, labels, label_biases)}.')

def supervised_train(model,train_loader,optimizer,args):
    for epoch in range(args.finetune_epoch):
        epoch_loss = 0.
        total=0.
        accy=0.
        for idx, (x, y) in enumerate(train_loader):
            x = x.to(torch.device('cuda'))
            y = y[0].to(torch.device('cuda'))
            logit=model(x)
            loss = F.cross_entropy(logit,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            with torch.no_grad():
                total+=len(x)
                _,pred=torch.max(logit,dim=1)
                accy+=torch.sum(pred==y)
        print(f'Epoch {epoch} Loss: {epoch_loss/len(train_loader):.4f}. Accy: {accy/total:.4f}')    

if __name__ == '__main__':
    set_seeds(929, 929, 929, 929)
    is_colab = 'google.colab' in sys.modules
    args = parse_option()


    if args.time_str == '':
        args.time_str = datetime.datetime.now().strftime('%m%d-%H-%M-%S-%f')[:-3]
    if not is_colab:
        set_gpu(args.gpu)
    pprint(vars(args))

    prepare_handle = PrepareFunc(args)
    train_loader, test_loader = prepare_handle.prepare_dataloader(args.dataset)


    '''
    test1
    '''
    val_ae(None, test_loader, prefix='raw',is_raw=True)
    tsne_ae(None, test_loader, 'raw',is_raw=True)
    
    
    '''
    test2
    '''
    hidden, label = [], []
    for idx, (x, y) in enumerate(test_loader):
        if x.shape[1] == 3:
            x = x.mean(dim=1, keepdim=True)
        if isinstance(y, list):
            y = y[0]
        x = x.view(x.shape[0], -1)
        hidden.append(x.detach().cpu())
    hidden_np = torch.cat(hidden).numpy()
    pca=PCA(n_components=30).fit(hidden_np)
    val_ae(None, test_loader, prefix='raw-pca',is_raw=True,pca=pca)
    tsne_ae(None, test_loader, 'raw-pca',is_raw=True,pca=pca)
    
    '''
    test3
    '''

    ae_dims = [784, 2000, 1000, 500, 30]
    rbm_models = prepare_handle.prepare_model('rbm_handle')
    criterion = prepare_handle.prepare_loss_fn()
    if args.do_train_rbm:
        for in_features, out_features in zip(ae_dims[:-1], ae_dims[1:]):
            cur_model = prepare_handle.prepare_model('rbm', [in_features, out_features])
            train_rbm(cur_model, train_loader, rbm_models, criterion, args)
            rbm_models.append(cur_model)
    ae_model = prepare_handle.prepare_model('ae', ae_dims)
    prepare_handle.load_rbm_pretrained_models(ae_model, rbm_models)
    optimizer = prepare_handle.prepare_optimizer(ae_model)
    train_ae(ae_model, train_loader, criterion, optimizer, args)
    val_ae(ae_model, test_loader, prefix="ae")
    tsne_ae(ae_model, test_loader, 'mnist-test-autoencoder')    
        
    '''
    test4
    '''
    args.do_train_rbm=True
    ae_dims = [784, 2000, 1000, 500, 30]
    rbm_models = prepare_handle.prepare_model('rbm_handle')
    criterion = prepare_handle.prepare_loss_fn()
    if args.do_train_rbm:
        for in_features, out_features in zip(ae_dims[:-1], ae_dims[1:]):
            cur_model = prepare_handle.prepare_model('rbm', [in_features, out_features])
            train_rbm(cur_model, train_loader, rbm_models, criterion, args)
            rbm_models.append(cur_model)

    ae_model = prepare_handle.prepare_model('ae', ae_dims)

    prepare_handle.load_rbm_pretrained_models(ae_model, rbm_models)
    optimizer = prepare_handle.prepare_optimizer(ae_model)
    train_ae(ae_model, train_loader, criterion, optimizer, args)
    torch.save(ae_model.state_dict(),"ae_param.pth")

    val_ae(ae_model, test_loader, prefix="rbm")
    tsne_ae(ae_model, test_loader, 'mnist-test-bolzmann')
    
    
    
    '''
    test 5:
    '''
    if args.bonus:
        ae_dims = [784, 2000, 1000, 500, 30]
        ae_model = prepare_handle.prepare_model('ae', ae_dims)
        ae_model.load_state_dict(torch.load("ae_param.pth"))
        debias_train_loader, debias_test_loader = prepare_handle.prepare_dataloader('ColoredMNIST')
        
        val_ae(ae_model, debias_test_loader,prefix='color-raw',is_raw=True)
        tsne_ae(ae_model, debias_test_loader, 'color-raw',is_raw=True)
        val_ae(ae_model, debias_test_loader,prefix='color')
        tsne_ae(ae_model, debias_test_loader, 'color')
        
        model=torchvision.models.resnet18(pretrained=True)
        model.fc=torch.nn.Linear(512,10)
        model=model.to(torch.device("cuda"))
        for p in model.parameters():
            p.requires_grad=False
        for p in model.fc.parameters():
            p.requires_grad=True
        optimizer = prepare_handle.prepare_optimizer(model)
        print(optimizer)
        supervised_train(model,debias_train_loader,optimizer,args)
        with torch.no_grad():
            val_extract_bias_conflicting(model, debias_train_loader)

