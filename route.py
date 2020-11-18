# 3rd-party
from flask import request
from flask_restplus import Namespace, Resource, reqparse
import opts
import torch
from torch.nn.functional import softmax

# framework
from bert_model import BERTModel
from data import tokenizer


# 실행 변수 불러오기
args = opts.get_args()
args.resume_from_checkpoint = 'model/intent_distilkobert.ckpt'
if(type(args.gpus) is int and args.gpus > 0):  # gpu 사용 확인
    args.device = "cuda"
else:
    args.device = "cpu"

# 모델 불러오기
model = BERTModel.load_from_checkpoint(args.resume_from_checkpoint)
model.freeze()
model.to(args.device)

# 네임스페이스 생성

ns = Namespace('intent_classifier', description="""intent classifier
- response types
    - FRAG : Fragments
    - STAT : Statements
    - QUES : Questions
    - COMM : Commands
    - RHEQ : rhetorical questions
    - RHEC : rhetorical commands
    - INTO : Intonation-dependent utterances""")

# request parser 생성

utt_parser = reqparse.RequestParser()
utt_parser.add_argument('utterance', location='form', type=str, default="",
                        help="""자연어로 표현된 대화문.
                        예) 등록금이 얼마인지 궁금합니다.
                        """)
query_parser = reqparse.RequestParser()
query_parser.add_argument('query', location='json', default="",
                          help="""query like transformer-cli forward.
                          json example
                          ```
                          {
                              "inputs": [
                                  {"utterance" : "반갑습니다." },
                                  {"utterance" : "오늘 날씨는 어떤가요?" }
                              ]
                          }
                          ```
                          """)


# 라우팅
def get_encodings(query):
    """ 자연어로 표현되어 있는 query를 신경망의 입력 벡터로 인코딩한다.

    Args:
        query (dict): API로 전달받은 query. 복수의 (질문, 답변을 찾을 문장) 으로 구성되어 있다.
                    {"inputs": [
                        {"utterance": 인텐션을 조사할 대화문},
                          ...
                    ]}

    Returns:
        BatchEncoding: 인코딩 된 query
    """
    def cleaning(text):
        return text.replace(u'\xa0', u' ')  # 인코딩 오류 유니코드 수정
    input_pairs = [(cleaning(input_data['utterance'])) for input_data in query['inputs']]
    encodings = tokenizer.batch_encode_plus(input_pairs, padding=True, truncation=True,
                                            return_token_type_ids=False, return_tensors='pt').to(args.device)  # pytorch용 텐서로 변환
    return encodings


def get_top_prob(encodings):
    logits = model(encodings)[0]
    probs = softmax(logits, dim=1)
    top = torch.topk(probs, dim=1, k=1)
    return top.indices, top.values


@ns.doc(parser=utt_parser)
@ns.route("/test/")
class IntentTest(Resource):
    def post(self):
        args = utt_parser.parse_args()
        utterance = args['utterance']
        query = {'inputs': [
            {'utterance': utterance}
        ]}
        encodings = get_encodings(query)
        tops, probs = get_top_prob(encodings)
        top_idx = tops[0].item()
        top_prob = probs[0].item()

        if top_idx == 0:
            return '[FRAG {0:.2f}] 부가 정보가 필요합니다...'.format(top_prob)
        elif top_idx == 1:
            return '[STAT {0:.2f}] 그렇군요. 알겠습니다.'.format(top_prob)
        elif top_idx == 2:
            return '[QUES {0:.2f}] 궁금하신 점이 있으신 것 같네요.'.format(top_prob)
        elif top_idx == 3:
            return '[COMM {0:.2f}] 요청 접수되었습니다.'.format(top_prob)
        elif top_idx == 4:
            return '[RHEQ {0:.2f}] 진심으로 대답을 원하시는 건 아니죠?'.format(top_prob)
        elif top_idx == 5:
            return '[RHEC {0:.2f}] 일단 알겠습니다.'.format(top_prob)
        else:
            return '[INTO {0:.2f}] 목소리가 들렸으면 좋겠어요.'.format(top_prob)

        return top_idx[0].item()


@ns.doc(parser=query_parser)
@ns.route("/forward/")
class IntentForward(Resource):
    def post(self):
        query = request.get_json()
        encodings = get_encodings(query)
        tops, probs = get_top_prob(encodings)
        print(tops, probs)
        return [{'cls': top.item(), 'prob': prob.item()} for top, prob in zip(tops, probs)]
