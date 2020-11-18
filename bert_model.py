""" kobert classifier 모델 """

# 3rd-party
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import transformers as tfs
from ext.ranger import Ranger

# module
from data import ds3i4k


class BERTModel(pl.LightningModule):
    """ kobert 기반의 classifier 모델

    학습 과정은 pytorch lightning 프레임워크를 이용함.
    신경망 구조는 huggingface의 transformers 프레임워크를 이용함.
    """
    def __init__(self, hparams) -> None:
        """모델 초기화

        transformers의 sequence classification 모델 자동 생성을 이용하여 신경망 구조를 만든다.
        자동 생성은 모델 명에 근거하여 적절한 구조를 만든다.
        예: monologg/kobert 모델 사용 시, bert 레이어 + classification을 위한 fc 레이어로 초기화 된다.
        자세한 모델 구조는 https://github.com/huggingface/transformers/blob/master/src/transformers/modeling_bert.py 을 참조.
        """
        super().__init__()
        self.hparams = hparams

        if type(hparams) is dict:  # API 호출 시에는 hparams가 dict로 들어옴
            self.model = tfs.AutoModelForSequenceClassification.from_pretrained(hparams['model'],
                                                                                num_labels=hparams['num_labels'])
        else:
            self.model = tfs.AutoModelForSequenceClassification.from_pretrained(hparams.model,
                                                                                num_labels=hparams.num_labels)

    def init_dataset(self, hparams) -> None:
        """ 학습 과정에 필요한 데이터 집합을 준비한다.
        """
        self.dataset = ds3i4k(hparams)
        # 학습 과정에 필요한 feature를 선택한다.
        if('distil' in hparams.model):  # distil-bert의 경우 token_type_ids를 사용하지 않는다.
            columns_to_return = ['input_ids', 'attention_mask', 'labels']
        else:
            columns_to_return = ['input_ids', 'token_type_ids', 'attention_mask', 'labels']
        # feature 들은 numpy 포맷으로 저장되어 있으므로, torch로 변환하도록 set_format 한다.
        self.dataset.set_format(type='torch', columns=columns_to_return)

    def forward(self, x) -> tuple:
        """ 모델에 데이터를 입력하고 실행하여 결과 값을 받는다.

        Args:
            x (dict): 학습 데이터 집합의 mini-batch

        Returns:
            SequenceClassifierModelOutput: (loss, logits, hidden_states, attentions)
        """
        # print(x)
        return self.model(**x)

    def training_step(self, batch, batch_idx):
        """ 학습 step forward 결과 값을 토대로 신경망의 가중치를 조정하도록 loss 값을 반환한다.
        가중치 조정은 pytorch_lightning 프레임워크를 이용하여 진행한다.
        """
        outputs = self(batch)
        loss = outputs[0]
        self.log('loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """검증 단계의 step 결과 값 중 loss, logits을 반환한다.
        반환된 값은 메모리에 모이고, 이를 epoch_end 단계에서 사용한다.
        """
        outputs = self(batch)
        loss, logits = outputs
        self.log('val_loss', loss)
        return [outputs, batch]

    def validation_epoch_end(self, outputs):
        """검증 단계의 결과 값을 모아 모델의 성능을 평가한다.
        """
        val_loss_mean = torch.cat([output[0][0].unsqueeze(0) for output in outputs]).mean()
        # logits 중 가장 높은 값을 가지는 index를 찾는다.
        y_pred = torch.cat([torch.argmax(output[0][1], dim=1) for output in outputs])
        y_true = torch.cat([output[1]['labels'] for output in outputs])

        correct = y_pred.eq(y_true).sum().item()
        accuracy = correct / len(y_true)

        self.log('accuracy', accuracy, prog_bar=True, on_epoch=True)
        self.log('val_loss', val_loss_mean, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        """ 옵티마이저 설정
        """
        ranger_params = {'lr': self.hparams.lr, 'alpha': self.hparams.alpha, 'k': self.hparams.k}
        optimizer = Ranger(self.model.parameters(), **ranger_params)
        return optimizer

    def train_dataloader(self):
        """ 학습 데이터 집합 사용 준비
        """
        return DataLoader(self.dataset['train'], batch_size=self.hparams.bs, num_workers=self.hparams.workers)

    def val_dataloader(self):
        """ 검증 데이터 집합 사용 준비
        """
        return DataLoader(self.dataset['validation'], batch_size=self.hparams.bs, num_workers=self.hparams.workers)
