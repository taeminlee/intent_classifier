# -*- coding: utf-8 -*-
""" APP serve """

# 3rd-party
from flask import Flask
from flask_cors import CORS

# fix flask_restplus import
from flask_restplus_import_error_fix import fix_resource
from flask_restplus import Api

# framework
from utils import make_opts
from logger import JSONExceptionHandler, RotateLoggingHandler
import opts
from route import ns

# 옵션 파일 생성 확인
make_opts()

# 실행 변수 불러오기
args = opts.get_args()

# Flask App 생성
app = Flask('intent classifier')
app.config['DEBUG'] = True
app.config['JSON_SORT_KEYS'] = False
app.config['SWAGGER_UI_DOC_EXPANSION'] = 'list'

# Logger 설정
json_except_handler = JSONExceptionHandler(app)
log_handler = RotateLoggingHandler(app)

# CORS 설정
CORS(app)

# Flask-rest Api 생성
fix_resource()
api = Api(app, version='1.0', title='intent classifier', description='intent classifier')

api.add_namespace(ns)


if __name__ == '__main__':
    app.run(port=args.port, host=args.host, threaded=True)
