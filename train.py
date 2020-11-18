""" 학습 모듈 """

# python
import datetime

# 3rd-party
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pprint import pprint

# module
from utils import make_opts
import opts
import utils
from bert_model import BERTModel

# 옵션 파일 생성 확인
make_opts()


def experiment(args):
    # fix default tpu_cores args to use wandb correctly
    if(type(args.tpu_cores) is not str and type(args.tpu_cores) is not int):
        args.tpu_cores = None

    utils.seed_everything(seed=args.seed)  # 시드 설정

    if('bert' in args.model):
        model = BERTModel(hparams=args)  # 모델 초기화
    else:
        raise Exception('--model에 지원하지 않는 모델을 입력하셨습니다.')

    model.init_dataset(hparams=args)  # 데이터 로드

    if(not args.fast_dev_run):  # fast_dev_run 일 때에는 로깅 사용 안함
        wandb_logger = WandbLogger(project='intent', tags=args.tags, offline=args.fast_dev_run)
        wandb_logger.watch(model, log='all')
        args.logger = wandb_logger

    if(args.tags is None):
        args.tags = []

    folder = ['checkpoints'] + args.tags + [str(datetime.datetime.now()), '/']  # 중간 저장 위치 설정

    # 체크포인트 콜백, f1 값 상위 10개 저장
    checkpoint_callback = ModelCheckpoint(
        filepath='/'.join(folder) + "_{epoch}-{val_loss:.4f}-{accuracy:.4f}",
        save_top_k=10,
        monitor='val_loss',
        verbose=True,
        mode='min'
    )

    args.checkpoint_callback = checkpoint_callback

    trainer = pl.Trainer.from_argparse_args(args)  # 트레이너 초기화

    trainer.fit(model)  # 학습 시작


if __name__ == '__main__':
    args = opts.get_args()
    pprint(vars(args))  # 실험 변수 출력
    experiment(args)  # 실험 시작
