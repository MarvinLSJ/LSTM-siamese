#coding=utf-8

import os
import yaml
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import DataLoader

# utils
from utils import get_embedding, load_embed, save_embed, data_preprocessing
# data
from data import myDS, mytestDS
# model
from model import Siamese_lstm

FLAGS = None

def main(_):

    # Load the configuration file.
    with open(FLAGS.config, 'r') as f:
        config = yaml.load(f)

    print('**********', config['experiment_name'],'**********')

    """ Data Preprocessing """

    if config['data_preprocessing']:
        print 'Pre-processing Original Data ...'
        data_preprocessing()
        print 'Data Pre-processing Done!'

    """ Read Data & Get Embedding """

    train_data = pd.read_csv('input/cleaned_train.csv')
    test_data = pd.read_csv('input/cleaned_test.csv')

    # split dataset
    msk = np.random.rand(len(train_data)) < 0.8
    train = train_data[msk]
    valid = train_data[~msk]
    all_sents = train_data['s1'].tolist() + train_data['s2'].tolist() + test_data['s1'].tolist() + test_data['s2'].tolist()

    # dataset
    trainDS = myDS(train, all_sents)
    validDS = myDS(valid, all_sents)

    print 'Data size:',train_data.shape[0], test_data.shape[0]

    full_embed_path = config['embedding']['full_embedding_path']
    cur_embed_path = config['embedding']['cur_embedding_path']

    if os.path.exists(cur_embed_path) and not config['make_dict']:
        embed_dict = load_embed(cur_embed_path)
        print 'Loaded existing embedding.'
    else:
        print 'Making embedding...'
        embed_dict = get_embedding(trainDS.vocab._id2word, full_embed_path)
        save_embed(embed_dict,cur_embed_path)
        print 'Saved generated embedding.'


    vocab_size = len(embed_dict)
    # initialize nn embedding
    embedding = nn.Embedding(vocab_size, config['model']['embed_size'])
    embed_list = []
    for word in trainDS.vocab._id2word:
        embed_list.append(embed_dict[word])
    weight_matrix = np.array(embed_list)
    # pass weights to nn embedding
    embedding.weight = nn.Parameter(torch.from_numpy(weight_matrix).type(torch.FloatTensor), requires_grad = False)

    """ Model Preparation """

    # embedding
    config['embedding_matrix'] = embedding
    config['vocab_size'] = len(embed_dict)

    # model
    siamese = Siamese_lstm(config)

    # loss func
    loss_weights = Variable(torch.FloatTensor([1, 3]))
    if torch.cuda.is_available():
        loss_weights = loss_weights.cuda()
    criterion = torch.nn.CrossEntropyLoss(loss_weights)

    # optimizer
    learning_rate = config['training']['learning_rate']
    if config['training']['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, siamese.parameters()), lr=learning_rate)
    elif config['training']['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, siamese.parameters()), lr=learning_rate)
    elif config['training']['optimizer'] == 'adadelta':
        optimizer = torch.optim.Adadelta(filter(lambda x: x.requires_grad, siamese.parameters()), lr=learning_rate)
    elif config['training']['optimizer'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(filter(lambda x: x.requires_grad, siamese.parameters()), lr=learning_rate)
    print 'Optimizer:', config['training']['learning_rate']

    # log info
    train_log_string = '%s :: Epoch %i :: Iter %i / %i :: train loss: %0.4f'
    valid_log_string = '%s :: Epoch %i :: valid loss: %0.4f\n'

    # Restore saved model (if one exists).
    ckpt_path = os.path.join(config['ckpt_dir'], config['experiment_name']+'.pt')

    if os.path.exists(ckpt_path):
        print('Loading checkpoint: %s' % ckpt_path)
        ckpt = torch.load(ckpt_path)
        epoch = ckpt['epoch']
        siamese.load_state_dict(ckpt['siamese'])
        optimizer.load_state_dict(ckpt['optimizer'])
    else:
        epoch = 0
        print 'Fresh start!\n'

    """ Train """

    if config['task'] == 'train':

        # save every epoch for visualization
        train_loss_record = []
        valid_loss_record = []
        best_record = 10.0

        # training
        print 'Experiment:{}\n'.format(config['experiment_name'])

        while epoch < config['training']['num_epochs']:

            print 'Start Epoch{} Training...'.format(epoch)

            # loss
            train_loss = []
            train_loss_sum = []
            # dataloader
            train_dataloader = DataLoader(dataset=trainDS, shuffle=True, num_workers=2, batch_size=1)

            for idx, data in enumerate(train_dataloader, 0):

                # get data
                s1, s2, label = data

                # clear gradients
                optimizer.zero_grad()

                # input
                output = siamese(s1, s2)
                output = output.squeeze(0)

                # loss backward
                loss = criterion(output, Variable(label))
                loss.backward()
                optimizer.step()
                train_loss.append(loss.data.cpu())
                train_loss_sum.append(loss.data.cpu())

                # Every once and a while check on the loss
                if ((idx + 1) % 5000) == 0:
                    print(train_log_string % (datetime.now(), epoch, idx + 1, len(train), np.mean(train_loss)))
                    train_loss = []

            # Record at every epoch
            print 'Train Loss at epoch{}: {}\n'.format(epoch, np.mean(train_loss_sum))
            train_loss_record.append(np.mean(train_loss_sum))

            # Valid
            print 'Epoch{} Validating...'.format(epoch)

            # loss
            valid_loss = []
            # dataloader
            valid_dataloader = DataLoader(dataset=validDS, shuffle=True, num_workers=2, batch_size=1)

            for idx, data in enumerate(valid_dataloader, 0):
                # get data
                s1, s2, label = data

                # input
                output = siamese(s1, s2)
                output = output.squeeze(0)

                # loss
                loss = criterion(output, Variable(label))
                valid_loss.append(loss.data.cpu())

            print(valid_log_string % (datetime.now(), epoch, np.mean(valid_loss)))
            # Record
            valid_loss_record.append(np.mean(valid_loss))
            epoch += 1
            # Keep track of best record
            if np.mean(valid_loss) < best_record:
                best_record = np.mean(valid_loss)
                # save the best model
                state_dict = {
                    'epoch': epoch,
                    'siamese': siamese.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(state_dict, ckpt_path)
                print 'Model saved!\n'

    """ Inference """

    if config['taks'] == 'inference':
        testDS = mytestDS(test_data, all_sents)
        # Do not shuffle here
        test_dataloader = DataLoader(dataset=testDS, num_workers=2, batch_size=1)

        result = []
        for idx, data in enumerate(test_dataloader, 0):

            # get data
            s1, s2 = data

            # input
            output = siamese(s1,s2)
            output = output.squeeze(0)

            # feed output into softmax to get prob prediction
            sm = nn.Softmax(dim=1)
            res = sm(output.data)[:,1]
            result += res.data.tolist()

        result = pd.DataFrame(result)
        print 'Inference Done.'
        res_path = os.path.join(config['result']['filepath'], config['result']['filename'])
        result.to_csv(res_path, header=False, index=False)
        print 'Result has writtn to', res_path, ', Good Luck!'



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='Configuration file.')
    FLAGS, _ = parser.parse_known_args()
    main(_)