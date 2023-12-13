# Data Processing and Cleaning

### Downloading the Dataset
To process data, you need to obtain the CTB7 files and but it under under Data_Processing 
and rename it to LDC07T36. 

### Preprocessing and POS tagging
We will preprocess the POS-tagged files, 
an example of a sentence inside a file looks like this:
```
今天_NT 在_P 台湾_NR 一_CD 台_M 电视_NN 节目_NN 上_LC ，_PU 李敖_NR 和_CC 周玉蔻_NR 人_NN 批评_VV 陈水扁_NR 今天_NT 说_VV 台湾_NR 股市_NN 可_VV 上_VV 万_CD 点_M ，_PU 根本_AD 是_VC 骗_VV 人_NN 的_DEC 恶_JJ 梦_NN 。_PU
```
This is quite nice, because the words are already segmented and POS tagged. 

### POS-tags of the dataset
The dataset consists of 33 tags, which are
- VA: Predicative adjective
- VC: Copula (like the English "be" in certain contexts)
- VE: Existential verb (similar to the English "have" in contexts like "There is")
- VV: Other verbs
- NR: Proper noun
- NT: Temporal noun (referring to time-related concepts)
- NN: Common noun
- LC: Localizer (indicating a spatial or temporal location)
- PN: Pronoun
- DT: Determiner
- CD: Cardinal number
- OD: Ordinal number
- M: Measure word (quantifiers and classifiers)
- AD: Adverb
- P: Preposition
- CC: Coordinating conjunction
- CS: Subordinating conjunction
- DEC: A particle used after a verb to form a relative clause (similar to "that" in English)
- DEG: Associative particle (equivalent to the English "'s" or "of")
- DER: A particle used after a verb or an adjective in certain cleft constructions
- DEV: A particle used to form adverbial expressions
- SP: Sentence-final particle
- AS: Aspect particle (indicating aspects like completed action)
- ETC: Etcetera (used in lists to indicate "and so on")
- MSP: Other particles
- IJ: Interjection
- ON: Onomatopoeia
- PU: Punctuation
- JJ: Other noun-modifier (adjective)
- FW: Foreign words (words borrowed from other languages)
- LB: Long Bei construction marker (a special structure in Chinese grammar)
- SB: Short Bei construction marker
- BA: Ba construction marker (another unique structure in Chinese grammar)

And the result of a processed data would look something like this
```
洲	S-NN
冠	B-NN
军	M-NN
杯	E-NN
小	B-NN
组	M-NN
赛	E-NN
已	B-AD
经	E-AD
过	B-VV
半	E-VV
```

Where S represents a single chunk, B, E, M represents beginning, end
and middle of a chunk. 
