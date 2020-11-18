# intent_classifier

Korean Intention classifier with pytorch lightning ⚡

## About

This is an intention classifier that was learned using [3i4k dataset to identify the intonation intention of Korean](https://github.com/warnikchow/3i4k).

- implemented using phytorch lighting. ⚡
- trained based on Transformers' distil-KoBERT PLM and has approx. 90% accuracy.
- use it as a web API powerd by flask-restplus.

## Installation

- clone
- `pip install -r requirement.txt`
- [download model file](https://drive.google.com/file/d/1Cs02CwBrnkCTfTHPyj2pJwOZLfhmmMiE/view?usp=sharing)
- copy into `/model/intent_distilkobert.ckpt`

## Train

- `bash train_distil.sh` to use distil-KoBERT model

![](https://i.imgur.com/64JVtsi.png)

- `bash train.sh` to use KoBERT model

## Inference

- `python app.py` to start web server
- open `{SERVER_IP}:41063` in web browser
- using swagger UI to `try out`

![](https://i.imgur.com/xmIIBFX.png)

- response types
    - FRAG : Fragments
    - STAT : Statements
    - QUES : Questions
    - COMM : Commands
    - RHEQ : rhetorical questions
    - RHEC : rhetorical commands
    - INTO : Intonation-dependent utterances

## References

- [3i4k dataset to identify the intonation intention of Korean](https://github.com/warnikchow/3i4k)
- [monologg/distil-kobert](https://github.com/monologg/DistilKoBERT)