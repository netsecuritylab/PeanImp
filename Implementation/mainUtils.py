import torch
import logging
import time
import numpy as np
from tqdm import tqdm
from train_evalCopy import train
import argparse
import pickle
import PEAN_model_copy
import os
import pyshark

logger = logging.getLogger(__name__)

def get_k_fold_data(k, i, X):
    assert k > 1
    fold_size = len(X) // k

    X_train = None
    for j in range(k):
        X_part = X[j * fold_size: (j + 1) * fold_size]
        if j == i:
            X_valid = X_part
        elif X_train is None:
            X_train = X_part
        else:
            X_train = X_train + X_part
    return X_train, X_valid


class Config(object):
    def __init__(self,):
        self.model_name = "PEAN"
        pretrain_path = './Model/pretrain/'
        record_path = './Model/record/'
        log_path = './Model/log/'
        loss_path = './Model/loss/'
        save_path = './Model/save/'
        dirs = [pretrain_path, record_path, log_path, loss_path, save_path]
        for dir in dirs:
            if not os.path.exists(dir):
                os.makedirs(dir)

        self.pretrainModel_json = pretrain_path + 'model_128d_8h_2l/config.json'
        self.pretrainModel_path = pretrain_path + 'model_128d_8h_2l/model_128d_8h_2l.pth'
        self.dataset = 'sni_whs'
        self.train_path = './Implementation/testMainDataset.txt' #'./TrafficData/' + '{}_train.txt'.format(self.dataset)
        self.class_list = [x.strip() for x in open('./TrafficData/class.txt').readlines()]
        self.save_path = save_path
        self.record_path = record_path
        self.loss_path = loss_path
        self.log_path = log_path
        self.vocab_path = './Config/vocab.txt'
        self.n_vocab = 261
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dropout = 0.5
        self.require_improvement = 10000
        self.num_classes = len(self.class_list)
        self.bert_dim = 128  # It must be consistent with the settings in pretrain_config.json
        self.num_layers = 2
        self.middle_fc_size = 2048


def setup_main_params():
    parser = argparse.ArgumentParser(description='Traffic Classification')
    parser.add_argument('--pad_num', type=int, default=10, help='the padding size of packet num')
    parser.add_argument('--pad_len', default=400, type=int, help='the padding size(length) of each packet')
    parser.add_argument('--pad_len_seq', default=10, type=int, help='the padding size of packet length sequence')
    parser.add_argument('--emb', default=128, type=int, help='the emb size of bytes')
    parser.add_argument('--device', default='cuda:0', type=str, help='the training device')
    parser.add_argument('--load', default=False, type=bool, help='whether train on previous model')
    parser.add_argument('--batch', default=64, type=int, help='batch_size')
    parser.add_argument('--feature', default='ensemble', type=str, help='length / raw / ensemble')
    parser.add_argument('--method', default='trf', type=str, help='lstm / trf (Sequential Layer)')
    parser.add_argument('--embway', default='random', type=str, help='random / pretrain (for raw)')
    parser.add_argument('--imploss', default=False, type=bool, help='whether to use improved loss')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--length_emb_size', default=32, type=int, help='len emb size')
    parser.add_argument('--lenhidden', default=128, type=int, help='len hidden size')
    parser.add_argument('--embhidden', default=1024, type=int, help='emb hidden size')
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--trf_heads', default=8, type=int, help='transformers heads number')
    parser.add_argument('--trf_layers', default=2, type=int, help='transformers layers')
    parser.add_argument('--mode', default='train', type=str, help='train/test')
    parser.add_argument('--k', default='10', type=int, help='k fold validation')
    parser.add_argument('--epoch', default='300', type=int, help='epoch')
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    return args

def get_config(args):
    config = Config()
    config.pad_num = args.pad_num
    config.pad_length = args.pad_len
    config.pad_len_seq = args.pad_len_seq
    config.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')  # 设备
    config.mode = args.mode
    config.embedding_size = args.emb
    config.batch_size = args.batch
    config.load = args.load
    config.lenlstmhidden_size = args.lenhidden
    config.emblstmhidden_size = args.embhidden
    config.feature = args.feature
    config.method = args.method
    config.embway = args.embway
    config.length_emb_size = args.length_emb_size
    config.imploss = args.imploss
    config.learning_rate = args.lr
    config.seed = args.seed
    config.trf_heads = args.trf_heads
    config.trf_layers = args.trf_layers
    config.k = args.k
    config.num_epochs = args.epoch
    if args.mode == "test":
        config.load = True
        config.num_epochs = 0

    name = "{}_{}".format(config.feature, config.seed)
    if config.feature != "length":
        name += "_{}_{}_{}_{}_{}_{}".format(config.embway, config.method, config.embedding_size, config.pad_num,
                                            config.pad_length, config.emblstmhidden_size)
    if config.feature == "length" or config.feature == "ensemble":
        name += "_{}_{}".format(config.pad_len_seq, config.lenlstmhidden_size)
    if config.method == "trf":
        if config.trf_heads == 8 and config.trf_layers == 2:
            pass
        else:
            name += "_{}_{}".format(config.trf_heads, config.trf_layers)
    if config.imploss:
        name += "_imploss"
    config.print_path = config.record_path + name + ".txt"  # record console log
    config.loss_path = config.loss_path + name + ".txt"     # record loss
    config.save_path = config.save_path + name + ".ckpt"    # record saved model
    print("\nModel save at: ", config.save_path)
    from transformers import BertTokenizer
    config.tokenizer = BertTokenizer(vocab_file=config.vocab_path, max_seq_length=config.pad_num - 2, max_len=config.pad_num)

    return config

def prepare_data(config, args):
    print("----------------------------\n")

    with open(config.print_path, 'a') as f:
        f.write("----------------------------\n\n")

    msg = "Iput Feature: {}\nRandom Seed: {}\n".format(config.feature, config.seed)
    if args.feature == "raw" or args.feature == "ensemble":
        msg += "Sequential use: {}\n".format(config.method)
        msg += "Embedding way: {}(hidden:{})\n".format(config.embway, config.emblstmhidden_size)
        if config.method == "pretrain":
            msg += "Bert Size: {}\n".format(config.bert_dim)
        else:
            msg += "Embedding Size: {}\n".format(config.embedding_size)
        msg += "Pad_num: {}\n".format(config.pad_num)
        msg += "Pad_len: {}\n".format(config.pad_length)

    if config.feature == "length" or args.feature == "ensemble":
        msg += "Length use: lstm(emb: {}, hidden:{})\n".format(config.length_emb_size, config.lenlstmhidden_size)
        msg += "Pad_len_seq: {}\n".format(config.pad_len_seq)

    if config.method == "trf":
        msg += "trf heads:{}\n".format(config.trf_heads)
        msg += "trf_layers: {}\n".format(config.trf_layers)

    msg += "Use Improved loss: {}\n".format(config.imploss)
    msg += "Learning Rate: {}\n".format(config.learning_rate)
    msg += "Batch Size:{}\n".format(config.batch_size)

    print(msg)
    with open(config.print_path, 'a') as f:
        f.write(msg)
    print("----------------------------\n")
    with open(config.print_path, 'a') as f:
        f.write("----------------------------\n\n")

    print("Loading data...")
    with open(config.print_path, 'a') as f:
        f.write("Loading data...\n")


    train_data = build_dataset(config)

    print("Train_set length: {}".format(len(train_data)))
    with open(config.print_path, 'a') as f:
        f.write("train_set: {}\n".format(len(train_data)))
    return train_data


def get_model(config):
    return PEAN_model_copy.PEAN(config).to(config.device)

UNK, PAD, CLS, SEP = '[UNK]', '[PAD]', '[CLS]','[SEP]'


def readPcapMain(pcap_path, output_txt, labels):
    """ 
    For now only TLS bytes will be saved, while packet length instead
    comes from the sum of the two.
    """
    packets = pyshark.FileCapture(pcap_path)

    logger.info(f"Converting pcap file at {pcap_path} into the specified .txt")
    with open(output_txt, "a", encoding="utf-8") as f:
        len_seq = []
        pkt_bytes = []
        for i, pkt in enumerate(packets):

            pkt_len = 0
            tls_data_len = 0
            tls_bytes = None

            if 'TLS' in pkt:
                try:
                    if hasattr(pkt.tls, 'record_length'):
                        lengths = pkt.tls.record_length.all_fields
                        for l in lengths:
                            # l.showname contiene la stringa "Length: 535"
                            # l.raw_value contiene il valore esadecimale
                            # l.get_default_value() contiene il numero "535"
                            tls_data_len += int(l.get_default_value())
                        pkt_len += tls_data_len
                    tls_bytes = pkt.tcp.payload.replace(':', ' ')[:400] # accessing through tcp is much easier

                    if 'TCP' in pkt:
                        pkt_len += int(pkt.tcp.len)
                except:
                    continue


           #if 'UDP' in pkt:
           #    print(f'{i} UDP: ', int(pkt.udp.length))
            if pkt_len and tls_bytes:
                len_seq.append(pkt_len)
                pkt_bytes.append(tls_bytes)
            else:
                continue

            '''
            The paper uses 10 packets per flow, we save 15 just in case
            '''
            if len(len_seq) == 15:
                f.write('\t'.join(map(str, pkt_bytes)) + '\t')
                f.write(' '.join(map(str, len_seq)) + f'\t{labels[os.path.basename(pcap_path)]}' + '\n') # TODO: label is placeholder
                pkt_bytes = []
                len_seq = []
    packets.close()
    print('Parsing finished!')

def readPcap_folderMain(folder_path, output_txt):
    labels = assign_labels(folder_path)
    for folder in os.listdir(folder_path):
        path = os.path.join(folder_path, folder)
        if not(os.path.isfile(path)):
            for file in os.listdir(path):
                readPcapMain(os.path.join(path, file), output_txt, labels)

def assign_labels(path):
    labels = {}
    for i, directory in enumerate(os.listdir(path)):
        labels[directory] = i
    return labels

def build_dataset(config):
    def load_dataset(path, pad_num = 10, pad_length = 400, pad_len_seq = 10):
        cache_dir = './DataCache/'
        cached_dataset_file = cache_dir + '{}_{}_{}_{}.txt'.format(config.dataset, pad_num, pad_length, pad_len_seq)
        if os.path.exists(cached_dataset_file):
            print("Loading features from cached file {}".format(cached_dataset_file))
            with open(cached_dataset_file, "rb") as handle:
                contents = pickle.load(handle)
                return contents
        else:
            print("Creating training dataset....")
            contents = []
            with open(path, 'r') as f:
                for line in tqdm(f):
                    if not line:
                        continue
                    item = line.split('\t')
                    flow = item[0:-2]  # packets
                    if len(flow) < 2:
                        continue
                    if len(flow) > pad_num:
                        flow = flow[0 : pad_num]
                    length_seq = item[-2].strip().split(' ')
                    length_seq = list(map(int, length_seq))

                    label = item[-1]
                    masks = []
                    seq_lens = []
                    traffic_bytes_idss = []
                    for packet in flow:
                        traffic_bytes = config.tokenizer.tokenize(packet)
                        if len(traffic_bytes) <= pad_length - 2:
                            traffic_bytes = [CLS] + traffic_bytes + [SEP]
                        else:
                            traffic_bytes = [CLS] + traffic_bytes
                            traffic_bytes[pad_length - 1] = SEP


                        seq_len = len(traffic_bytes)
                        mask = []
                        traffic_bytes_ids = config.tokenizer.convert_tokens_to_ids(traffic_bytes)

                        if pad_length:
                            if len(traffic_bytes) < pad_length:
                                mask = [1] * len(traffic_bytes_ids) + [0] * (pad_length - len(traffic_bytes))  # [1,1,...,1,0,0]
                                traffic_bytes_ids += ([0] * (pad_length - len(traffic_bytes)))
                            else:
                                mask = [1] * pad_length
                                traffic_bytes_ids = traffic_bytes_ids[:pad_length]
                                seq_len = pad_length
                        traffic_bytes_idss.append(traffic_bytes_ids)
                        seq_lens.append(seq_len)
                        masks.append(mask)


                        if pad_len_seq:
                            if len(length_seq) < pad_len_seq:
                                length_seq += [0] * (pad_len_seq - len(length_seq))
                            else:
                                length_seq = length_seq[:pad_len_seq]


                    if pad_num: 
                        if len(traffic_bytes_idss) < pad_num:
                            len_tmp = len(traffic_bytes_idss)

                            mask = [0] * pad_length
                           
                            traffic_bytes_ids = [1] + [0] * (pad_length-2) + [2]
                            seq_len = 0
                            for i in range(pad_num - len_tmp):
                                masks.append(mask)
                                traffic_bytes_idss.append(traffic_bytes_ids)
                                seq_lens.append(seq_len)
                        else:
                            traffic_bytes_idss = traffic_bytes_idss[:pad_num]
                            masks = masks[:pad_num]
                            seq_lens = seq_lens[:pad_num]

                    contents.append((traffic_bytes_idss, seq_lens, masks, length_seq, int(label))) 

            print("Saving dataset cached file {}".format(cached_dataset_file))
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            with open(cached_dataset_file, "wb") as handle:
                pickle.dump(contents, handle, protocol=pickle.HIGHEST_PROTOCOL)
            return contents

    train = load_dataset(config.train_path, config.pad_num, config.pad_length, config.pad_len_seq)
    return train


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device, pad_len_seq):
        self.batch_size = batch_size 
        self.batches = batches
        print("len batches and batch size: ", len(batches), self.batch_size)
        self.n_batches = len(batches) // self.batch_size
        self.residue = False
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device
        self.pad_len_seq = pad_len_seq

    def _to_tensor(self, datas):
        # datas: batch_size * contents
        # contents: traffic_bytes_idss, seq_lens, masks, length_seq, int(label)
        traffic_bytes_idss = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        length_seq = torch.LongTensor([_[3] for _ in datas])
        length_seq = torch.reshape(length_seq, (-1,self.pad_len_seq,1)).to(self.device)
        seq_lens = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        masks = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        label = torch.LongTensor([_[4] for _ in datas]).to(self.device)

        return (traffic_bytes_idss, length_seq, seq_lens, masks), label

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches

def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device, config.pad_len_seq)
    return iter

def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

if __name__ == '__main__':
    print("Testing...")
   #readPcapMain('Implementation/pcapDatasets/org.telegram.messenger.pcap/org.telegram.messenger.pcap'
   #             , 'Implementation/mainDatasetTest.txt')
    readPcap_folderMain("Implementation/pcapDatasets", 'Implementation/testMainDataset.txt')