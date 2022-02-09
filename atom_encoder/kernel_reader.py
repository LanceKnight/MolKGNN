import torch


from torch.nn import Embedding, Module, Linear, Softmax, CrossEntropyLoss
import torch
from torch import optim
from clearml import Task
from tqdm import tqdm
import os


class Encoder(Module):
    def __init__(self, hidden_dim):
        super(Encoder, self).__init__()
        self.encode = Embedding(118, hidden_dim)

    def forward(self, atom_num):
        atom_emb = self.encode(atom_num)
        return atom_emb

    def save_param(self, path):
        torch.save(self.state_dict(), path)

    def load_param(self, path):
        # param = torch.load(path)
        self.encode.load_state_dict(torch.load(path))


class Decoder(Module):
    def __init__(self, hidden_dim):
        super(Decoder, self).__init__()
        self.decode = Linear(hidden_dim, 118)

    def forward(self, atom_emb):
        atom_num_prob = self.decode(atom_emb)
        return atom_num_prob

    def save_param(self, path):
        torch.save(self.state_dict(), path)

    def load_param(self, path):
        self.load_state_dict(torch.load(path))


def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)

    acc = torch.round(acc * 100)

    return acc


def train(criterion, optimizer):
    atom_list = []
    for atom_id in range(118):
        atom_tensor = torch.tensor(atom_id)
        atom_list.append(atom_tensor)
        atom_list.append(atom_tensor)
    atom_tensor = torch.stack(atom_list)

    emb = encoder(atom_tensor)
    atom_num_prob = decoder(emb)
    y_pred_softmax = torch.log_softmax(atom_num_prob, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    loss = criterion(atom_num_prob, atom_tensor)
    acc = multi_acc(atom_num_prob, atom_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item(), acc.item()


def intepret(input):
    output = decoder(input)
    pred_num = torch.max(torch.log_softmax(output, dim=1), dim=1)
    print(f'num:{pred_num}')


if __name__ == '__main__':
    # gnn_type = 'kgnn'
    # task = Task.init(project_name=f"Tests/{gnn_type}",
    #                  task_name="atom_embedding_intepreter",
    #                  tags=[gnn_type, "debug", "dummy"])
    # logger = task.get_logger()

    encoder = Encoder(32)
    encoder.load_param('atom_encoder.pt')
    decoder = Decoder(32)
    if os.path.exists('decoder.pt'):
        decoder.load_param('decoder.pt')

    model = decoder

    # Train Decoder
    # criterion = CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=1 * 10 ** -2)
    # for epoch in tqdm(range(100)):
    #     loss, acc = train(criterion, optimizer)
    #     print(f'loss:{loss} acc:{acc}')
    #     logger.report_scalar(title='loss', series='train', value=loss, iteration=epoch)
    #     logger.report_scalar(title='accuracy', series='train', value=acc, iteration=epoch)
    # decoder.save_param('decoder.pt')

    # input = encoder(torch.tensor([3]))

    path = 'kernels.pt'
    params = torch.load(path)
    deg1_x_center = params['2.x_center'].squeeze(1)
    print(deg1_x_center.shape)
    deg1_atom_center = intepret(deg1_x_center)
    print(deg1_atom_center)
    for param in params:
        print(param)
