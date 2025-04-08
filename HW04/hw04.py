from konlpy.tag import Okt
from gensim.models import Word2Vec


with open("naver_shopping_corpus.txt", "r", encoding="utf-8") as file:
    corpus = [line[1:].strip() for line in file.readlines()]

okt = Okt()
tokens = [okt.morphs(sentence) for sentence in corpus]

model = Word2Vec(tokens, vector_size=100, min_count=5, sg=0)

while True:
    target_word = input("단어를 입력하세요(Quit: q): ")
    if target_word == "q" or target_word == "Q":
        break

    similar_words = model.wv.similar_by_word(word=target_word, topn=5)
    print(similar_words)
