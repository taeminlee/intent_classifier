""" FLASK Restplus 에러 수정 """

import werkzeug
werkzeug.cached_property = werkzeug.utils.cached_property


def fix_resource():
    print('flask-restplus import error fix using \
          https://github.com/noirbizarre/flask-restplus/issues/777')
