import os
import sys
import numpy as np
import keras
from keras import layers
import jieba


whole = open('白夜行.txt', encoding='utf-8').read()
all_words = list(jieba.cut(whole, cut_all=False))  # jieba分词
words = sorted(list(set(all_words)))
word_indices = dict((word, words.index(word)) for word in words)

maxlen = 30
sentences = []
next_word = []

for i in range(0, len(all_words) - maxlen):
    sentences.append(all_words[i: i + maxlen])
    next_word.append(all_words[i + maxlen])
print('提取的句子总数:', len(sentences))

x = np.zeros((len(sentences), maxlen), dtype='float32') # Embedding的输入是2维张量（句子数，序列长度）
y = np.zeros((len(sentences)), dtype='float32')
for i, sentence in enumerate(sentences):
    for t, word in enumerate(sentence):
        x[i, t] = word_indices[word]
    y[i] = word_indices[next_word[i]]

print(np.round((sys.getsizeof(x) / 1024 / 1024 / 1024), 2), "GB") 


main_input = layers.Input(shape=(maxlen, ), dtype='float32') 
model_1 = layers.Embedding(len(words), 128, input_length=maxlen)(main_input)
model_1 = layers.Bidirectional(layers.GRU(256, return_sequences=True))(model_1)
model_1 = layers.Bidirectional(layers.GRU(128))(model_1)
output = layers.Dense(len(words), activation='softmax')(model_1)  
model = keras.models.Model(main_input, output)

optimizer = keras.optimizers.RMSprop(lr=3e-3)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer)
model.fit(x, y, epochs=100, batch_size=1024, verbose=2)


def sample(preds, temperature=1.0):
    if not isinstance(temperature, float) and not isinstance(temperature, int):
        print('\n\n', "temperature must be a number")
        raise TypeError
        
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)

    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def write_2(model, temperature, word_num, begin_sentence):
    gg = begin_sentence[:30]
    print(''.join(gg), end='/// ')
    for _ in range(word_num):
        sampled = np.zeros((1, maxlen)) 
        for t, char in enumerate(gg):
            sampled[0, t] = word_indices[char]
    
        preds = model.predict(sampled, verbose=0)[0]
        if temperature is None:
            next_word = words[np.argmax(preds)]
        else:
            next_index = sample(preds, temperature)
            next_word = words[next_index]
            
        gg.append(next_word)
        gg = gg[1:]
        sys.stdout.write(next_word)
        sys.stdout.flush()


begin_sentence = whole[50003: 50100]
print("初始句：", begin_sentence[:30])
begin_sentence = list(jieba.cut(begin_sentence, cut_all=False))


write_2(model, None, 300, begin_sentence)

write_2(model, 0.5, 300, begin_sentence)