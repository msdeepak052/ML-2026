# 1. Why Convert Text to Vectors?

Machine learning models **cannot understand text**.

Example sentence:

```
"I love NLP"
```

A model cannot process words directly.

So we convert text into **numerical vectors**.

Pipeline:

```
Text
↓
Tokenization
↓
Vocabulary
↓
Vector Representation
↓
ML Model
```

One of the simplest methods:

```
One Hot Encoding
```

---

# 2. What is One Hot Encoding?

**One Hot Encoding** converts each word into a **binary vector** where:

* One position = **1**
* All others = **0**

Example vocabulary:

```
["I", "love", "NLP"]
```

Assign an index:

| Word | Index |
| ---- | ----- |
| I    | 0     |
| love | 1     |
| NLP  | 2     |

Vector size = **3** (size of vocabulary)

Now each word becomes a vector.

| Word | Vector  |
| ---- | ------- |
| I    | [1,0,0] |
| love | [0,1,0] |
| NLP  | [0,0,1] |

---

# 3. Simple Visual Example

Sentence:

```
"I love NLP"
```

Vocabulary:

```
["I","love","NLP"]
```

Vectors:

```
I    → [1,0,0]
love → [0,1,0]
NLP  → [0,0,1]
```

Sentence representation:

```
[[1,0,0],
 [0,1,0],
 [0,0,1]]
```

Each row = **word vector**

---

# 4. Example with More Words

Sentence:

```
"I love machine learning"
```

Vocabulary:

```
["I","love","machine","learning"]
```

Vectors:

| Word     | Vector    |
| -------- | --------- |
| I        | [1,0,0,0] |
| love     | [0,1,0,0] |
| machine  | [0,0,1,0] |
| learning | [0,0,0,1] |

Sentence vector representation:

```
[[1,0,0,0],
 [0,1,0,0],
 [0,0,1,0],
 [0,0,0,1]]
```

---

# 5. Important Concept — Vocabulary

Vocabulary = **all unique words in dataset**

Example dataset:

```
"I love NLP"
"I love AI"
```

Vocabulary:

```
["I","love","NLP","AI"]
```

Vectors:

| Word | Vector    |
| ---- | --------- |
| I    | [1,0,0,0] |
| love | [0,1,0,0] |
| NLP  | [0,0,1,0] |
| AI   | [0,0,0,1] |

---

# 6. Example With Two Sentences

Sentences:

```
1. I love NLP
2. NLP loves AI
```

Vocabulary:

```
["I","love","NLP","loves","AI"]
```

Vectors:

| Word  | Vector      |
| ----- | ----------- |
| I     | [1,0,0,0,0] |
| love  | [0,1,0,0,0] |
| NLP   | [0,0,1,0,0] |
| loves | [0,0,0,1,0] |
| AI    | [0,0,0,0,1] |

Sentence 1:

```
I love NLP

[[1,0,0,0,0],
 [0,1,0,0,0],
 [0,0,1,0,0]]
```

Sentence 2:

```
NLP loves AI

[[0,0,1,0,0],
 [0,0,0,1,0],
 [0,0,0,0,1]]
```

---

# 7. Python Implementation (Manual)

Example sentence:

```
"I love NLP"
```

### Step 1 — Create vocabulary

### Step 2 — Generate vectors

```python
sentence = ["I","love","NLP"]

vocab = list(set(sentence))

print(vocab)
```

Example vocabulary:

```
['NLP', 'love', 'I']
```

---

### Step 3 — One Hot Encoding

```python
import numpy as np

vocab = ["I","love","NLP"]

one_hot = {}

for i,word in enumerate(vocab):
    
    vector = [0]*len(vocab)
    vector[i] = 1
    
    one_hot[word] = vector

print(one_hot)
```

Output:

```
{
'I': [1,0,0],
'love': [0,1,0],
'NLP': [0,0,1]
}
```

---

# 8. Convert Sentence to One Hot Vectors

```python
sentence = ["I","love","NLP"]

vectors = [one_hot[word] for word in sentence]

print(vectors)
```

Output:

```
[[1,0,0],
 [0,1,0],
 [0,0,1]]
```

---

# 9. Using Scikit-Learn OneHotEncoder

Example:

```python
from sklearn.preprocessing import OneHotEncoder
import numpy as np

words = np.array(["I","love","NLP"]).reshape(-1,1)

encoder = OneHotEncoder()

encoded = encoder.fit_transform(words)

print(encoded.toarray())
```

Output:

```
[[1 0 0]
 [0 1 0]
 [0 0 1]]
```

---

# 10. Major Problem with One Hot Encoding

### Problem 1 — No semantic meaning

Example:

```
cat → [1,0,0]
dog → [0,1,0]
car → [0,0,1]
```

Distance between words:

```
cat vs dog
cat vs car
```

Both appear **equally distant**, even though:

```
cat and dog are similar
cat and car are not
```

Model **cannot capture word meaning**.

---

### Problem 2 — Huge vectors

Vocabulary size example:

```
100,000 words
```

Vector size:

```
100,000 dimensions
```

This becomes **very sparse and inefficient**.

---

# 11. Visual Representation

Vocabulary:

```
["I","love","machine","learning"]
```

Word vectors:

```
I         [1 0 0 0]
love      [0 1 0 0]
machine   [0 0 1 0]
learning  [0 0 0 1]
```

Each word activates **one position only**.

That’s why it's called:

```
One Hot
```

---

# 12. When One Hot Encoding is Used

Used in:

* Basic NLP models
* Small datasets
* Teaching fundamentals
* Classical ML models

Not used in modern NLP because better methods exist.

---

# 13. Summary

| Concept          | Meaning                          |
| ---------------- | -------------------------------- |
| One Hot Encoding | converts word into binary vector |
| Vector size      | vocabulary size                  |
| Value            | one position = 1                 |
| Others           | 0                                |

Example:

Vocabulary:

```
["I","love","NLP"]
```

Vectors:

```
I    → [1,0,0]
love → [0,1,0]
NLP  → [0,0,1]
```

---

