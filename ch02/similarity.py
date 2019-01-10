import sys
sys.path.append('c:\\Users\\ryosh\\Python\\learn')
from common.util import preprocess, create_to_matrix, cos_similarity

text = 'You say goodbye and i say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_to_matrix(corpus, vocab_size)

c0 = C[word_to_id['you']]
c1 = C[word_to_id['i']]
print(cos_similarity(c0, c1))