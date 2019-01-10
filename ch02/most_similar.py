import sys
sys.path.append('c:\\Users\\ryosh\\Python\\learn')
from common.util import preprocess, create_to_matrix, most_similar

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_to_matrix(corpus, vocab_size)

most_similar('you', word_to_id, id_to_word, C, top=5)