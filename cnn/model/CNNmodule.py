import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from dropblock import DropBlock2D

class Block(nn.Module):
    """
    Changed Block to no longer have last self.relu, which was completely unnecessary.
    """
    def __init__(self, n_inputs, n_outputs, n_outputs_final, kernel_size, stride, padding, dilation=1, dropout=0.2):
        super(Block, self).__init__()
        self.conv1 = weight_norm(nn.Conv2d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.acti1 = nn.ReLU()
        self.dropout1 = nn.Dropout2d(dropout)

        self.conv2 = weight_norm(nn.Conv2d(n_outputs, n_outputs_final, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.acti2 = nn.ReLU()
        self.dropout2 = nn.Dropout2d(dropout)
        self.net = nn.Sequential(self.conv1, self.acti1, self.dropout1,
                                 self.conv2, self.acti2, self.dropout2)
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        return out
class BNBlock(nn.Module):
    """
    Changed Block to no longer have last self.relu, which was completely unnecessary.
    """
    def __init__(self, n_inputs, n_outputs, n_outputs_final, kernel_size, stride, padding, dilation=1, activation = 'relu', dropout=0.2):
        super(BNBlock, self).__init__()
        if activation == 'relu':
            activation_nn =  nn.ReLU()
        if dropout >0:
            drop_block1 = DropBlock2D(block_size=3, drop_prob=dropout)
            drop_block2 = DropBlock2D(block_size=3, drop_prob=dropout)

        self.conv1 = weight_norm(nn.Conv2d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation, bias=False))
        self.bn1 = nn.BatchNorm2d(n_outputs)
        self.acti1 = activation_nn#nn.ReLU()


        self.conv2 = weight_norm(nn.Conv2d(n_outputs, n_outputs_final, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation, bias=False))
        self.bn2 = nn.BatchNorm2d(n_outputs_final)
        self.acti2 = activation_nn#nn.ReLU()
        self.net = nn.Sequential(self.conv1, self.bn1, self.acti1,
                                 self.conv2, self.bn2, self.acti2)
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        return out
class ConvNet(nn.Module):
    def __init__(self, picture_dim, target_dim,num_inputs, num_channels, kernel_size, activation = 'relu', dropout=0.2): #, kernel_size=3
        super(ConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels) - 1
        # Create correct number of blocks
        for i in range(num_levels):
            if i > 0:
                #m =
                layers += [nn.MaxPool2d(2, stride=2)]
            in_channels = num_inputs if i == 0 else int(num_channels[i])
            out_channels = num_channels[i]
            out_channels_final = num_channels[i+1]
            layers += [BNBlock(in_channels, out_channels, out_channels_final,kernel_size[i], stride=1,
                                     padding=int((kernel_size[i]-1)/2), activation = 'relu', dropout=0.2)]

        # here i could add so that I could use pass all running results to the final FC layer.
        #dropout_2d_final = nn.Dropout2d(dropout)
        self.network_cnn = nn.Sequential(*layers)#,dropout_2d_final)

        self.fst_fc = int(num_channels[-1]*(picture_dim)/(2**(2*num_levels-2)))

        if activation == 'relu':
            activation_nn =  nn.ReLU()

        fc1 = nn.Linear( self.fst_fc, 256)
        acti_fc = activation_nn
        dropout_fc2 = nn.Dropout(dropout)
        fc2 = nn.Linear(256, target_dim)
        self.network_fc = nn.Sequential(fc1, dropout_fc2, acti_fc, fc2)
        #self.network_fc = nn.Sequential(fc1,acti_fc, fc2)
    def forward(self, x):
        out = self.network_cnn(x)
        out = out.view(-1,self.fst_fc )
        #out = out.view(-1,self.fst_fc )
        out = self.network_fc(out)
        return out#F.log_softmax(out, dim=1)

    def reset_fc(self, dim_to_unroll, dim_target):
        self.fst_fc = dim_to_unroll
        fc1 = nn.Linear( dim_to_unroll, 256)
        fc2 = nn.Linear(256, dim_target)
        self.network_fc = nn.Sequential(fc1,fc2)

def train(model, trainloader, optimizer, criterion, BATCH_SIZE, log_interval, ep):
    """
    Function defined for training.
    """
    steps = 0
    train_loss = 0
    train_loss_detailed = []
    model.train()
    for batch_idx, (data, target) in enumerate(trainloader):

        #if args.cuda: data, target = data.cuda(), target.cuda()
        #data = data.view(-1, input_channels)#, seq_length)
        #if args.permute:
        #    data = data[:, :, permute]
        #data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)

        #print(output.shape)
        loss = criterion(output, target)#F.nll_loss(output, target)
        loss.backward()
        #if args.clip > 0:
        #    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        train_loss += loss.item()/target.shape[0]
        train_loss_detailed.append(train_loss)

        steps += BATCH_SIZE
        if batch_idx > 0 and batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tSteps: {}'.format(
                ep, batch_idx * BATCH_SIZE, len(trainloader.dataset),
                100. * batch_idx / len(trainloader), train_loss/log_interval, steps))
        train_loss = 0


    return train_loss_detailed

def val(model, testloader, criterion, BATCH_SIZE, log_interval, ep):
    """
    Function defined for testing. Note requires a testloader to be declared
     before use.

    """
    model.eval()
    val_loss = 0
    val_loss_detailed = list()
    correct = 0

    with torch.no_grad():
        for batch_idx,(data, target) in enumerate(testloader):
            output = model(data)
            val_loss += criterion(output, target).item()/target.shape[0]
            val_loss_detailed.append(val_loss)
            val_loss=0

            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

            #if batch_idx > 0 and batch_idx % log_interval == 0:
            #
               # test_loss = 0
        #test_loss /= len(testloader.dataset)
        #print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            #test_loss, correct, len(testloader.dataset),
            #100. * correct / float(len(testloader.dataset))))

    return val_loss_detailed, correct

def test_indepth(model, testloader, criterion, BATCH_SIZE, log_interval, ep):
    """
    Function defined for testing. Note requires a testloader to be declared
     before use.

    """
    model.eval()
    output_all = []
    target_all = []
    imall =[]

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in testloader:
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

            output_all.append(output.cpu())#.detach())
            target_all.append(target.cpu())
            imall.append(data)
        test_loss /= len(testloader.dataset)
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            test_loss, correct, len(testloader.dataset),
            100. * correct / float(len(testloader.dataset))))

        return torch.cat(output_all,dim =0), torch.cat(target_all, dim=0), torch.cat(imall,dim=0), test_loss#pred#test_loss, correct / len(test_loader.dataset)
