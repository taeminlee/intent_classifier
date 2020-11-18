""" MRC QA CLI 로깅 """

# python
import os
import logging
from logging.handlers import RotatingFileHandler
import time
from logging import Formatter

# 3rd-party
from werkzeug.exceptions import default_exceptions
from werkzeug.exceptions import HTTPException
from flask import jsonify, request

logger = logging.getLogger("intent")


def logger_init(filename='intent.log'):
    print("logger init")
    logger.setLevel(logging.DEBUG)

    if not os.path.exists('logs'):
        os.makedirs('logs')

    logFilePath = os.path.join("logs", filename)
    needRoll = os.path.isfile(logFilePath)
    file_handler = RotatingFileHandler(logFilePath, backupCount=100)
    logger.addHandler(file_handler)
    if needRoll:
        logger.handlers[0].doRollover()

    formatter = logging.Formatter('%(asctime)s [%(levelname)-5.5s] %(message)s')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler()
    logger.addHandler(stream_handler)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)


def info(*args):
    logger.info(args)


def debug(*args):
    logger.debug(args)


def warning(*args):
    logger.warning(args)


def error(*args):
    logger.error(args)


def exception(exception):
    logger.error(exception, exc_info=True)


def print_args(args):
    logger.info(" **************** CONFIGURATION **************** ")
    for key, val in sorted(vars(args).items()):
        keystr = "{}".format(key) + (" " * (30 - len(key)))
        logger.info("%s -->   %s", keystr, val)
    logger.info(" **************** CONFIGURATION **************** ")


class JSONException(Exception):
    """JSON 에러를 담는 exception 클래스

    Arguments:
        Exception -- 기본 Exception 상속
    """
    def __init__(self, description, code=-1):
        """초기화

        Arguments:
            description {str} -- 에러 명세를 담는다.

        Keyword Arguments:
            code {int} -- 에러 코드를 담는다. (default: {-1})
        """
        super().__init__(description)
        self.description = description
        self.code = code

    def __str__(self):
        """str

        Returns:
            str -- description
        """
        return self.description


class RotateLoggingHandler(object):
    """로테이팅 방식의 로그 생성기

    Arguments:
        object -- 기본 object 상속
    """
    def __init__(self, app):
        """초기화

        Keyword Arguments:
            app {Flask} -- 로깅을 수행할 Flask 앱
        """
        self.init_app(app)

    def init_app(self, app):
        """로거 설정

        Arguments:
            app {Flask} -- 로깅을 수행할 Flask 앱
        """
        print("="*100)
        print("rotate logger init_app")
        print("="*100)
        if not os.path.exists('logs'):
            os.makedirs('logs')

        # 로거 설정
        app.config['LOGGING_LEVEL'] = logging.DEBUG
        app.config['LOGGING_LOCATION'] = 'logs/'
        app.config['LOGGING_FORMAT'] = '%(asctime)s %(levelname)s: %(message)s in %(filename)s:%(lineno)d]'
        app.config['LOGGING_FILENAME'] = '{}.log'.format('intent')
        app.config['LOGGING_MAX_BYTES'] = 1000000
        app.config['LOGGING_BACKUP_COUNT'] = 100

        file_handler = RotatingFileHandler(
            app.config['LOGGING_LOCATION'] + app.config['LOGGING_FILENAME'],
            maxBytes=app.config['LOGGING_MAX_BYTES'],
            backupCount=app.config['LOGGING_BACKUP_COUNT'])
        file_handler.setFormatter(Formatter(app.config['LOGGING_FORMAT']))
        file_handler.setLevel(app.config['LOGGING_LEVEL'])
        if(app.logger.level == 0):
            app.logger.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)
        app.logger.info("logging start")

        @app.teardown_request
        def teardown_request(exception):
            """teardown request 오버라이딩

            Arguments:
                exception {Exception}} -- Exception 발생 시 not None
            """
            if(exception):
                app.logger.error(str(exception))
            else:
                app.logger.info("[req ] " + str(request))
                app.logger.info("[data] " + str(request.form))


class JSONExceptionHandler(object):
    def __init__(self, app=None):
        if app:
            self.init_app(app)

    def json_handler(self, error):
        response = jsonify({
            'code': error.code,
            'code_msg': error.description
        })
        response.status_code = error.code if isinstance(error, HTTPException) \
            else 400
        return response

    def std_handler(self, error):
        response = jsonify({
            'code': -1,
            'code_msg': str(error)
        })
        self.app.logger.error(str(error))
        response.status_code = error.code if isinstance(error, HTTPException) \
            else 500
        return response

    def init_app(self, app):
        self.app = app
        self.register(JSONException)
        for code, v in default_exceptions.items():
            self.register(code)

    def register(self, exception_or_code, handler=None):
        self.app.errorhandler(exception_or_code)(handler or self.std_handler)


def time_usage(func):
    """함수의 시간 비용을 측정하는 데코레이터

    사용 예:
    ```py
    @time_usage
    def some_loop():
        for i in range(1000):
            i ** i
    ```

    Args:
        func ([type]): [description]
    """
    def wrapper(*args, **kwargs):
        beg_ts = time.time()
        retval = func(*args, **kwargs)
        end_ts = time.time()
        debug(func.__module__, func.__name__, "elapsed time: %fs" % (end_ts - beg_ts))
        return retval
    return wrapper
