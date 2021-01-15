#计算pr、recall、和f1值

import json
import conlleval
def get_result(evalseq):
    count = conlleval.evaluate(evalseq)
    res = conlleval.report(count)
    return res

