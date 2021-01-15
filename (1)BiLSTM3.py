# import tensorflow as tf
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter

model_path = "model_BiLSTM3"
batch_size = 64

writer = SummaryWriter(f'./{model_path}/log')

string_id_x = {}
string_id_y = {}
def read_data(path):
    with open(path, "rb") as f:
        data = f.read().decode("utf-8")
    train_data = data.split("\n\n")  # 双行切分
    train_data = [token.split("\n") for token in train_data]    #逐个句子切分
    train_data = [[j.split() for j in i] for i in train_data]   #[   [ ['中','B-LOC'],['国','I-LOC'],xxxx ]                ]
    train_data.pop()    #弹出最后一个回车
    train_x = [[token[0] for token in sentence] for sentence in train_data] #[  ['中','国','x',xxx],['我','们',xxx]       ]
    train_y = [[token[1] for token in sentence] for sentence in train_data]
    return train_x,train_y,train_data


train_x,train_y,train_data =  read_data('data/example.train')
val_x,val_y,val_data = read_data('data/example.dev')

all_data_x = train_x+val_x
all_data_x = all_data_x+[['BIN','EOS']]
all_data_y = train_y+val_y+[['O','O']]

x_num,y_num = 1,1
for index in range(len(all_data_x)):
    for i in range(len(all_data_x[index])):
        char = all_data_x[index][i]
        if char not in string_id_x:
            string_id_x[char] = x_num
            x_num+=1
        label_char = all_data_y[index][i]
        if label_char not in string_id_y:
            string_id_y[label_char] = y_num
            y_num+=1

new_dict={v:k for k,v in string_id_y.items()}

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TextLoader(torch.utils.data.Dataset):
    def __init__(self,data):
        self.train_data = data
    def x_y(self,index):
        data_line = self.train_data[index]
        train_x_ = [ string_id_x.get('BIN') ] +[ string_id_x.get(token[0]) for token in data_line] + [string_id_x.get('EOS') ]
        train_y_ = [string_id_y.get('O')]+[ string_id_y.get(token[1]) for token in data_line]+[string_id_y.get('O')]
        return (torch.IntTensor(train_x_),torch.IntTensor(train_y_))
    def __getitem__(self, index):
        return self.x_y(index)
    def __len__(self):
        return  len(self.train_data)

class TextCollate():
    def __init__(self):
        pass
    def __call__(self, batch):
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]
        text_padded = torch.LongTensor(len(batch), max_input_len)
        label_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        label_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text
            label = batch[ids_sorted_decreasing[i]][1]
            label_padded[i, :label.size(0)] = label
        return text_padded,label_padded,input_lengths


def to_gpu(x):
    x = x.contiguous()
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)

def parse_batch(batch):
    text_padded, label_padded,input_lengths = batch
    text_padded = to_gpu(text_padded).long()
    label_padded = to_gpu(label_padded).long()
    return text_padded,label_padded,input_lengths


class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))
    def forward(self, x):
        return self.linear_layer(x)

class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class NERModel(torch.nn.Module):
    def __init__(self):
        super(NERModel,self).__init__()
        self.embedding = nn.Embedding( len(string_id_x)+1 ,128)

        self.BiLSTM = nn.LSTM(128,64,num_layers=3,batch_first=True,bidirectional=True)
        Linears = []
        for input_dim,output_dim in zip([128,64],[64, 32 ]):
            linear_layer = nn.Sequential(
                LinearNorm(
                    input_dim,output_dim,bias=True,w_init_gain='relu'
                )
            )
            Linears.append(linear_layer)
        self.Linears = nn.ModuleList(Linears)
        self.last = nn.Sequential(
                LinearNorm(
                    32,y_num,bias=True,w_init_gain='relu'
                )
            )
    def forward(self, x,input_lengths):
        x = self.embedding(x) #[batch_size,max_len,]
        # pytorch tensor are not reversible, hence the conversion
        # lstm:input [batch_size,len,dim]
        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True)

        self.BiLSTM.flatten_parameters()
        outputs, _ = self.BiLSTM(x)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True)

        for Linear in self.Linears:
            outputs = F.relu(Linear(outputs))
        outputs = self.last(outputs)
        return outputs
    def inference(self,x):
        x = self.embedding(x)
        # pytorch tensor are not reversible, hence the conversion
        # lstm:input [batch_size,len,dim]
        self.BiLSTM.flatten_parameters()
        outputs, _ = self.BiLSTM(x)

        for Linear in self.Linears:
            outputs = F.relu(Linear(outputs))
        outputs = self.last(outputs)
        return outputs

net = NERModel().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=0.005)
criterion = torch.nn.CrossEntropyLoss()

train_data_wait_load = TextLoader(train_data)
val_data_wait_load = TextLoader(val_data)
collate = TextCollate()
train_loader = DataLoader(train_data_wait_load,batch_size=batch_size, shuffle=True,collate_fn=collate)


def val(time):
    net.eval()
    loss_all = 0
    with torch.no_grad():
        val_loader = DataLoader(val_data_wait_load, batch_size=batch_size, collate_fn=collate)
        accuracy_all = 0
        char_num = 0
        for i, batch in enumerate(val_loader):
            x, y, input_lengths = parse_batch(batch)
            y_predit = net(x, input_lengths)
            prediction = torch.max( F.softmax(y_predit,2),2 )[1]
            y_predit = y_predit.view(-1, y_num)
            prediction = prediction.view(-1)
            y = y.view(-1)
            char_num = char_num + y.shape[0]
            accuracy_all = accuracy_all+int(sum(prediction==y))
            loss_all += criterion(y_predit, y)
        print(accuracy_all,char_num,accuracy_all/char_num)
        ave_loss = loss_all / ( i + 1 )
        writer.add_scalar('val_loss',ave_loss,time)
        writer.add_scalar('accuracy', accuracy_all/char_num, time)
    net.train()

is_paint = False

def train():
    global  is_paint
    for t in range(100):
        if t%10==0:
            val(t)
            torch.save(net.state_dict(), f'{model_path}/model_{t}.pth')
        loss_all = 0.0
        loss_epoch = 0
        for index,batch in enumerate(train_loader):
            x,y,input_lengths = parse_batch(batch)
            y_predit = net(x,input_lengths)
            if is_paint == False:
                writer.add_graph(net, (x, input_lengths))
                is_paint = True
            y_predit = y_predit.view(-1,y_num)
            y = y.view(-1)
            loss = criterion(y_predit,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_epoch += loss
            loss_all+=loss
            if index%100==0:
                print(f'loss: {loss_all/100} ')
                loss_all = 0
        print(f'epoch_time:{t} loss_epoch: {loss_epoch/(index + 1 )}')
        writer.add_scalar('loss',loss_epoch/(index+1),t)

def inference():
    net.load_state_dict(torch.load(f'{model_path}/model_90.pth'))
    while True:
        text = input("句子")
        text_after = []
        temp = []
        for word in text:
            try:
                temp.append(string_id_x[word])
                text_after.append(word)
            except:
                pass
        train_x_ = [ string_id_x.get('BIN') ] + temp + [ string_id_x.get('EOS') ]
        train_x_ = torch.IntTensor(train_x_).unsqueeze(0)
        train_x_ = to_gpu(train_x_).long()
        y_predit = net.inference(train_x_)
        prediction = torch.max(F.softmax(y_predit, 2), 2)[1]
        text = list(text)
        text = text
        prediction_label = [new_dict.get(i) for i in prediction[0].cpu().numpy()[1:-1]]
        print(
            [
                text_after[i]+":"+prediction_label[i] for i in range(len(prediction_label))
            ]
        )
        flag = False
        for i in range(len(prediction_label)):
            if prediction_label[i]!='O':
                if flag==False:
                    flag=True
                print(text[i],":",prediction_label[i],end=" ")
            elif prediction_label[i]=='O' and flag==True:
                flag = False
                print(" ")
        print("")


def cal_inference():
    net.load_state_dict(torch.load(f'{model_path}/model_90.pth'))
    from exc_text import read_data_clean
    test_x, test_y, test_data = read_data_clean('data/example.test')
    conlleval = []
    for index in range(len(test_data)):
        print(index / len(test_data))
        data_line = test_data[index]
        string_x = [token[0] for token in data_line]
        string_y = [token[1] for token in data_line]
        train_x_ = [string_id_x.get('BIN')] + [string_id_x[token] for token in string_x] + [string_id_x.get('EOS')]
        train_x_ = torch.IntTensor(train_x_).unsqueeze(0)
        train_x_ = to_gpu(train_x_).long()
        y_predit = net.inference(train_x_)
        prediction = torch.max(F.softmax(y_predit, 2), 2)[1]
        text = list(string_x)
        prediction_label = [new_dict.get(i) for i in prediction[0].cpu().numpy()[1:-1]]
        for i in range(len(data_line)):
            conlleval.append(
                '{} {} {}'.format(
                    string_x[i], string_y[i], prediction_label[i]
                )
            )
    from cal_f1 import get_result
    res = get_result(conlleval)
    print(res)

#cal_inference()
#train()
inference()