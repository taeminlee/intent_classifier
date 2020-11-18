""" 데이터 처리 """

# 3rd-party
import nlp
from nlp import Dataset
from transformers import BatchEncoding

# module
from kobert_tokenizer import KoBertTokenizerFast


# 토크나이저 초기화
# kobert의 토크나이저 사용
# 모델 마다 토크나이저가 다르므로, 모델 변경 시 토크나이저도 변경해야 함
tokenizer = KoBertTokenizerFast.from_pretrained('taeminlee/kobert', use_fast=True, model_max_length=512)


def ds3i4k(args) -> Dataset:
    """3i4k 데이터를 읽고, 학습 dataset으로 변환한다.

    Args:
        args (argparse): 입력 변수. args.train_file과 args.dev_file이 설정되어 있어야 함.
                         args.train_file: 학습 데이터 파일
                         args.dev_file: 검증 데이터 파일

    Returns:
        Dataset: nlp 패키지의 Dataset 형식. dataset['train']으로 학습 집합, dataset['validation']으로 검증 집합을 사용한다.
    """
    data_files = {'train': args.train_file,
                  'validation': args.dev_file}

    dataset = nlp.load_dataset('csv', data_files=data_files, delimiter='\t', column_names=['labels', 'utterance'])
    dataset = dataset.shuffle()
    dataset = dataset.map(convert_to_features, batched=True, batch_size=args.bs)

    # print(dataset)
    # print(random.choice(dataset['train']))
    # print(random.choice(dataset['validation']))

    return dataset


def convert_to_features(example_batch) -> BatchEncoding:
    # batch_encode_plus 함수로 자연어 텍스트를 토큰화(tokenize)하여 토큰 인덱스 집합으로 변환한다.
    # model max_length보다 큰 경우 잘라내기(truncation)한다.
    encodings = tokenizer.batch_encode_plus(example_batch['utterance'], padding=True, truncation=True, return_token_type_ids=True)

    encodings.update({'labels': example_batch['labels']})

    return encodings


if __name__ == '__main__':
    """ data 처리 테스트 """
    import argparse
    parser = argparse.ArgumentParser(description='Data')
    parser.add_argument('--model', type=str, default='monologg/kobert', help='Model name')

    data_args = parser.add_argument_group('학습 데이터 관련 인자')
    data_args.add_argument('--train_file', type=str, default='data/fci_train_val.txt', help='학습 데이터 경로')
    data_args.add_argument('--dev_file', type=str, default='data/fci_test.txt', help='dev 검증 데이터 경로')

    args = parser.parse_args()
    args.workers, args.bs, args.fast_dev_run = 10, 100, True
    ds3i4k(args)
