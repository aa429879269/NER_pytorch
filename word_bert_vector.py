
import numpy as np
def get_bert_vector(input):
    from bert_serving.client import BertClient
    bc = BertClient()
    temp =  bc.encode(input  )
    return temp

def create_vector():
    from exc_text import read_data_clean,read_data

    test_x,test_y,test_data =  read_data_clean('data/example.test')
    for i in range(len(test_data)):
        data = test_x[i]
        data = [ 'BIN' ] +data + ['EOS']
        target_data = get_bert_vector(data)
        np.save(f'data/test_npy/{i}',target_data)
        print(i/len(test_data))
