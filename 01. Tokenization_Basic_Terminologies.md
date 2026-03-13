# 1. What is Tokenization?

**Tokenization** is the process of splitting text into **smaller units called tokens**.

Tokens can be:

* words
* subwords
* characters
* sentences

### Example

Sentence:

```
"I love learning NLP"
```

After tokenization:

```
["I", "love", "learning", "NLP"]
```

Each element is a **token**.

So the pipeline becomes:

```
Text → Tokens → Numerical Representation → ML Model
```

Example pipeline:

```
"I love NLP"

Tokenization
["I","love","NLP"]

Vocabulary mapping
[12, 54, 91]

Input to model
```

---

# 2. Why Tokenization is Needed

ML models **cannot understand text directly**.

They only understand **numbers**.

So we convert:

```
Text → Tokens → Token IDs → Embeddings
```

Example:

```
"I love NLP"

Tokens:
["I","love","NLP"]

Token IDs:
[1, 2, 3]

Embeddings:
[[0.12,0.44,...],
 [0.23,0.11,...],
 [0.91,0.55,...]]
```

---

# 3. Types of Tokenization

## 3.1 Word Tokenization

Splits text by **words**.

Example:

```
Text:
"I love machine learning"

Tokens:
["I","love","machine","learning"]
```

### Python Example

```python
text = "I love machine learning"

tokens = text.split()

print(tokens)
```

Output

```
['I', 'love', 'machine', 'learning']
```

Problem:

```
"playing", "played", "player"
```

All treated as **different words**.

---

## 3.2 Character Tokenization

Split text into **characters**.

Example

```
Text:
"hello"

Tokens:
["h","e","l","l","o"]
```

### Python

```python
text = "hello"

tokens = list(text)

print(tokens)
```

Output

```
['h','e','l','l','o']
```

Pros

* Handles unknown words

Cons

* Sequence becomes very long.

---

## 3.3 Subword Tokenization (Modern NLP)

Modern models like **BERT, GPT, T5** use **subword tokenization**.

Example:

```
Word: unbelievable

Tokens:
["un", "believ", "able"]
```

Why?

Because models can handle **unknown words better**.

Example:

```
Word: transformerization
```

Word tokenizer → unknown

Subword tokenizer →

```
["transform", "er", "ization"]
```

---

# 4. Important NLP Terminologies

Now let’s understand the **basic terms used in tokenization**.

---

# 4.1 Corpus

A **corpus** is a **large collection of text data**.

Example:

```
Corpus = {
 "I love NLP",
 "NLP is amazing",
 "Deep learning is powerful"
}
```

Examples of famous corpora

* Wikipedia text
* Common Crawl
* Books dataset

In NLP training:

```
Corpus → Tokenization → Vocabulary → Model training
```

---

# 4.2 Vocabulary

Vocabulary is the **set of all unique tokens** in a dataset.

Example corpus:

```
"I love NLP"
"I love AI"
```

Tokens:

```
["I","love","NLP","I","love","AI"]
```

Vocabulary:

```
{"I","love","NLP","AI"}
```

Vocabulary size:

```
|V| = 4
```

---

# 4.3 Token

A **token** is a single unit produced by tokenization.

Example

```
Sentence:
"I love NLP"

Tokens:
["I","love","NLP"]
```

Number of tokens = **3**

---

# 4.4 Token ID

Each token gets a **numeric ID**.

Example vocabulary:

```
{
"I":1
"love":2
"NLP":3
"AI":4
}
```

Sentence:

```
"I love NLP"
```

Token IDs:

```
[1,2,3]
```

---

# 4.5 Out Of Vocabulary (OOV)

When the model encounters a **word not present in vocabulary**.

Example vocabulary

```
{"I","love","NLP"}
```

Sentence:

```
"I love transformers"
```

"transformers" → not present.

It becomes:

```
<UNK>
```

Example tokenization:

```
["I","love","<UNK>"]
```

---

# 4.6 Special Tokens

Modern NLP models use **special tokens**.

Common ones:

| Token   | Meaning               |
| ------- | --------------------- |
| `<PAD>` | padding               |
| `<UNK>` | unknown word          |
| `<CLS>` | classification token  |
| `<SEP>` | sentence separator    |
| `<BOS>` | beginning of sentence |
| `<EOS>` | end of sentence       |

Example:

```
Sentence:
"I love NLP"
```

Tokenized for BERT:

```
[CLS] I love NLP [SEP]
```

---

# 5. Padding

Neural networks require **equal length sequences**.

Example sentences

```
"I love NLP"
"NLP is great"
"I like deep learning"
```

Token lengths:

```
3
3
4
```

We pad shorter sentences.

Example:

```
["I","love","NLP","<PAD>"]
["NLP","is","great","<PAD>"]
["I","like","deep","learning"]
```

---

# 6. End-to-End Example

Text:

```
"I love learning NLP"
```

### Step 1 — Tokenization

```
["I","love","learning","NLP"]
```

### Step 2 — Vocabulary

```
{
"I":1
"love":2
"learning":3
"NLP":4
}
```

### Step 3 — Token IDs

```
[1,2,3,4]
```

### Step 4 — Model Input

```
Tensor([1,2,3,4])
```

---

# 7. Python Example with NLTK

Install:

```
pip install nltk
```

Code:

```python
import nltk
nltk.download('punkt')

from nltk.tokenize import word_tokenize

text = "I love learning NLP"

tokens = word_tokenize(text)

print(tokens)
```

Output

```
['I', 'love', 'learning', 'NLP']
```

---

# 8. Tokenization in Modern LLMs (Important)

Modern models use advanced tokenizers.

Examples:

| Model | Tokenizer     |
| ----- | ------------- |
| GPT   | BPE           |
| BERT  | WordPiece     |
| T5    | SentencePiece |

Example:

```
unhappiness
```

Tokenized as:

```
["un","happi","ness"]
```

This reduces vocabulary size.

---

# 9. Real Example from GPT Tokenization

Sentence

```
ChatGPT is amazing
```

Possible tokens:

```
["Chat", "G", "PT", " is", " amazing"]
```

Notice **spaces are tokens too**.

---

# 10. Summary

| Term           | Meaning                         |
| -------------- | ------------------------------- |
| Tokenization   | splitting text into tokens      |
| Token          | smallest unit of text           |
| Corpus         | collection of text              |
| Vocabulary     | set of unique tokens            |
| Token ID       | numeric representation of token |
| OOV            | word not in vocabulary          |
| Padding        | making sequences equal length   |
| Special Tokens | tokens like CLS, PAD            |

---

# 11. How This Connects to NLP Models

Pipeline:

```
Text
 ↓
Tokenization
 ↓
Token IDs
 ↓
Embeddings
 ↓
Neural Network
 ↓
Prediction
```

Example tasks:

* Sentiment Analysis
* Translation
* Chatbots
* Question Answering
* LLMs

---

