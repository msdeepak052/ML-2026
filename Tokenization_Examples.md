# Tokenization using NLTK

Below is a **step-by-step practical guide for Tokenization using NLTK**, starting from installation to different tokenization techniques. Each step includes **simple examples and Python code** so you can run it easily.

---

# 1. Install and Setup NLTK

First install **NLTK**.

```bash
pip install nltk
```

Now download required tokenizer models.

```python
import nltk
nltk.download('punkt')
```

`punkt` is a **pretrained sentence tokenizer model** used by NLTK.

---

# 2. Sentence Tokenization

Sentence tokenization splits **paragraph → sentences**.

Example text:

```
Hello everyone. Welcome to NLP learning. Tokenization is the first step.
```

### Python Code

```python
import nltk
from nltk.tokenize import sent_tokenize

text = "Hello everyone. Welcome to NLP learning. Tokenization is the first step."

sentences = sent_tokenize(text)

print(sentences)
```

### Output

```
['Hello everyone.', 
 'Welcome to NLP learning.', 
 'Tokenization is the first step.']
```

Explanation:

```
Paragraph
↓
Sentence Tokenization
↓
3 Sentences
```

---

# 3. Word Tokenization

Word tokenization splits **sentence → words**.

Example:

```
"I love learning NLP"
```

### Python Code

```python
from nltk.tokenize import word_tokenize

sentence = "I love learning NLP"

words = word_tokenize(sentence)

print(words)
```

### Output

```
['I', 'love', 'learning', 'NLP']
```

Explanation

```
Sentence
↓
Word Tokenization
↓
Individual words
```

---

# 4. Word Tokenization with Punctuation

NLTK automatically separates punctuation.

Example sentence:

```
Hello! How are you doing today?
```

### Python Code

```python
sentence = "Hello! How are you doing today?"

tokens = word_tokenize(sentence)

print(tokens)
```

### Output

```
['Hello', '!', 'How', 'are', 'you', 'doing', 'today', '?']
```

Important point:

```
Punctuation also becomes tokens
```

---

# 5. Tokenization of a Paragraph

Let’s combine **sentence + word tokenization**.

Text:

```
NLP is amazing. It helps computers understand language.
```

### Python Code

```python
text = "NLP is amazing. It helps computers understand language."

sentences = sent_tokenize(text)

for sentence in sentences:
    words = word_tokenize(sentence)
    print(words)
```

### Output

```
['NLP', 'is', 'amazing', '.']
['It', 'helps', 'computers', 'understand', 'language', '.']
```

Explanation

```
Paragraph
↓
Sentence Tokenization
↓
Word Tokenization
```

---

# 6. Using Treebank Word Tokenizer

NLTK also provides **TreebankWordTokenizer**, which follows **Penn Treebank rules**.

### Example

```python
from nltk.tokenize import TreebankWordTokenizer

tokenizer = TreebankWordTokenizer()

sentence = "I'm learning NLP with Python."

tokens = tokenizer.tokenize(sentence)

print(tokens)
```

### Output

```
['I', "'m", 'learning', 'NLP', 'with', 'Python', '.']
```

Notice:

```
"I'm → I + 'm
```

This helps models better understand grammar.

---

# 7. Regexp Tokenizer

You can define **custom tokenization rules using regex**.

Example: Remove punctuation.

Sentence:

```
Hello!!! NLP is fun :)
```

### Python Code

```python
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')

sentence = "Hello!!! NLP is fun :)"

tokens = tokenizer.tokenize(sentence)

print(tokens)
```

### Output

```
['Hello', 'NLP', 'is', 'fun']
```

Explanation:

```
\w+  → words only
```

Punctuation is removed.

---

# 8. Tokenizing Numbers and Words

Example:

```
I bought 5 apples and 10 oranges.
```

### Code

```python
sentence = "I bought 5 apples and 10 oranges."

tokens = word_tokenize(sentence)

print(tokens)
```

### Output

```
['I', 'bought', '5', 'apples', 'and', '10', 'oranges', '.']
```

Numbers also become tokens.

---

# 9. Real Small NLP Pipeline Example

Text:

```
Machine learning is powerful. NLP is part of AI.
```

### Python Code

```python
from nltk.tokenize import sent_tokenize, word_tokenize

text = "Machine learning is powerful. NLP is part of AI."

sentences = sent_tokenize(text)

for sentence in sentences:
    tokens = word_tokenize(sentence)
    print("Sentence:", sentence)
    print("Tokens:", tokens)
    print()
```

### Output

```
Sentence: Machine learning is powerful.
Tokens: ['Machine', 'learning', 'is', 'powerful', '.']

Sentence: NLP is part of AI.
Tokens: ['NLP', 'is', 'part', 'of', 'AI', '.']
```

---

# 10. Common Tokenizers in NLTK

| Tokenizer             | Purpose                    |
| --------------------- | -------------------------- |
| sent_tokenize         | split text into sentences  |
| word_tokenize         | split sentences into words |
| TreebankWordTokenizer | grammar-aware tokenizer    |
| RegexpTokenizer       | custom tokenization        |

---

# 11. Visual Flow

```
Raw Text
↓
Sentence Tokenizer
↓
Sentences
↓
Word Tokenizer
↓
Tokens
```

Example

```
"I love NLP. It is powerful."

↓

["I love NLP.", "It is powerful."]

↓

["I","love","NLP"]
["It","is","powerful"]
```

---

# 12. When Tokenization is Used in NLP

Tokenization is used in:

* Sentiment Analysis
* Machine Translation
* Chatbots
* Text Classification
* Search Engines
* Large Language Models

Pipeline

```
Text
↓
Tokenization
↓
Vectorization
↓
Machine Learning Model
```

---

# 13. Small Practice Exercises (Recommended)

Try these yourself.

### Exercise 1

Tokenize this paragraph:

```
Deep learning is changing the world. NLP is a part of it.
```

---

### Exercise 2

Remove punctuation using **RegexpTokenizer**.

```
Wow!!! NLP is amazing :)
```

---

### Exercise 3

Count total tokens.

```
"I love machine learning and natural language processing."
```

---


I can also show you **an end-to-end NLP preprocessing pipeline in Python**, which is something **every ML engineer should know.**
