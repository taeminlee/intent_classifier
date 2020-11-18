""" 프로그램 변수 """

# python
import argparse
import multiprocessing as mp

# 3rd-party
from pytorch_lightning import Trainer


def get_args():
    parser = argparse.ArgumentParser(description='QA')
    parser = Trainer.add_argparse_args(parser)  # pytorch lightning의 trainer 환경 변수 추가 (-h 명령어로 전체 목록 확인 가능)

    optim_args = parser.add_argument_group('Optimization related arguments')
    optim_args.add_argument('--bs', type=int,  default=6, help='Batch Size')
    optim_args.add_argument('--lr', type=float,  default=5e-5, help='Initial Learning rate')
    optim_args.add_argument('--alpha', type=float,  default=0.5, help='LookAHead params')
    optim_args.add_argument('--k', type=int,  default=6, help='LookAHead params')
    optim_args.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    optim_args.add_argument("--eps", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    optim_args.add_argument('--num_labels', type=int, default=7, help="number of target labels (0..6)")

    model_args = parser.add_argument_group('Bert Model related arguments')
    model_args.add_argument('--model', type=str, default='monologg/kobert', help='Model name')

    rnn_args = parser.add_argument_group('RNN Model related arguments')
    rnn_args.add_argument('--rnn_hidden_size', type=int, default=768, help='hideen size')
    rnn_args.add_argument('--rnn_model', type=str, default='lstm', help='rnn cell type (lstm | gru)')
    rnn_args.add_argument('--rnn_num_layer', type=int, default=2, help='number of rnn layers')
    rnn_args.add_argument('--rnn_bidirection', type=bool, default=True, help='bidirection')
    rnn_args.add_argument('--rnn_dropout', type=float, default=0.1, help='dropout ratio')

    misc_args = parser.add_argument_group('Logging related & Misc arguments')
    misc_args.add_argument('--seed', type=int, default=777, help='Random Seed')
    misc_args.add_argument('-t', '--tags', nargs='+', default=[], help='W&B Tags to associate with run')
    misc_args.add_argument('--workers', type=int, default=min(8, mp.cpu_count()-1), help='Number of parallel worker threads')

    data_args = parser.add_argument_group('Data related arguments')
    data_args.add_argument('--train_file', type=str, default='data/fci_train_val.txt', help='학습 데이터 경로')
    data_args.add_argument('--dev_file', type=str, default='data/fci_test.txt', help='dev 검증 데이터 경로')

    api_args = parser.add_argument_group('API related arguments')
    api_args.add_argument('--port', type=int, default=41063, help='API 포트')
    api_args.add_argument('--host', type=str, default='0.0.0.0', help='API 오픈 호스트. 0.0.0.0로 설정 시 외부 노출.\
                                                                       내부 접근만 허용할 경우 192.168.0.1등 접속 IP 명시.')

    args = parser.parse_args()
    return args
