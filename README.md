# nlp-ceo-speech
Description of using Natural Language Processing in creating business intelligence from a research perspective

## Background
The following work is part of academic research being conducted by Dr. Shavin Malhotra (University of Waterloo) and Dr. Phil Zhu (University of San Diego). Materials related to research were obtained with permission from Thomson Reuters (TR). 

The objective of the research is to identify "Big Five Personality Traits" and speech patterns of CEOs from earnings call transcripts. Upon identifying scores about such traits correlations with CEO performance such as salary or number of successful acquisitions can be measured.  

[Here is a link](http://journals.sagepub.com/doi/abs/10.1177/0001839217712240) to a recent journal article on the subject of CEO extraversion and their performance in acquisitions and mergers.

[This link](https://asqblog.com/2018/01/17/malhotra-reus-zhu-roelofsen-2017-the-acquisitive-nature-of-extraverted-ceos/) provides an interview with the authors of the papers, discussing a few more interesting details about the study. 

This work was carried out using Python, NLTK, PyRegex on Amazon Web Services cloud servers in an Ubuntu 12.04LTS environment. 

## Available Data
Earnings Call Transcripts of 2,381 CEOs from S&P 1500 firms were collected from the period 2004 to 2015. This yielded 78,000+ PDF files. 

A file typically looks something like this: [Interim 2016 AGL Energy Ltd Earnings Call](https://pastebin.com/Npcp3HCM) 

The structure of the PDF file often has these sections:
* Document Title with data-time of publication 
* Call Participants
  * Names and job titles of executives present
  * Names and job titles of analysts present
* Prepared speech by executives
* Q&A session with free form speech
* Copyright information

Each main section above is preceeded by a set of "======" signs and subsection preceeded by set of "-----" to create section breaks. 

The prepared speech is typically crafted by the executive and their team of assistants and advisors. As such, this section doesn't reveal the tone or voice of a single author. 

The Q&A session has impromptu speech of an individual. This section of the document was of main interest for the research study. 

## Processing Pipeline
Most of machine learning and any other kind of industrial automation design is quite simply months of brute force manual labor stashed into a set of "if-then" routines. The janitorial work that goes into curating and cleaning up the raw data stream is more of an art than a science. I will describe some of this art work here. 

### Extract Different Sections
The sections can be easily extracted from the pdf using regex. Also, each call participants name was preceeded by " * ". Such landmarks are artistically sought out to extract text from a file and store it in a format suitable for machine learning pipeline (database or text files). 

The main challenge while extracting the text was that in each call from a company, the same person could be identified by a variety of aliases. This job was too complicated to solve using simple regression techniques. So 2333 unique CEO names were identified along with a list of known aliases for each CEO by first visually identifying names that seemed similar within a company and then doing quite a bit of online research about the person. This list in itself is very valuable in making a social graphs of "aliases and known associates" of CEOs.

### File Naming Convention 
An earnings call PDF file obtained from TR Eikon has a naming convention of the form:
```
a) Date_CompanyTicker.ExchangeIdentifier_DocumentID.pdf
b) Date_CompanyTicker-ExchangeIdentifier_DocumentID.pdf
c) Date_CompanyTicker.ExchangeIdentifier^Alphanumerics_DocumentID.pdf
d) Date_CompanyTicker-ExchangeIdentifier^Alphanumerics_DocumentID.pdf
```
The Exchange Identifier indicates NYSE or NASDAQ. 

Quite often documents in TR Eikon database that are placed under the category of "Earnings Calls" may instead be a press release that had no executives or analysts present. This became evident to me much later when a large number of files didn't seem to have any CEO present during the call. 

The `Date` is unfortunately formatted to use words to identify months so files don't neatly arrange themselves into alphabetical order that matches an ascending or descending timeline. The best way to name dates, in my opinion, is `yyyy-mm-dd`. Likewise `date-time` can be `yyyy-mm-dd_hh-mm-ss`. This results in a neat arrangement of files that is easy to parse while writing scripts and also easy to visually identify in a folder. 

Using whitespaces, multiple dots and special characters in a file name creates various problems while dealing with the file in a cross platform setting. It is best to establish a convention and then stick to it. 

For the research, only files that had a CEO present in the call were of interest. So required files were renamed to:
```
CompanyTicker-ExchangeIdentifier_CEOName_Date.txt
```

### Get Personality Scores
The text samples were parsed using IBM Watson Personality and Tone Analyser (version dated 2016-09-22) as well as an algorithm created by [François Mairesse](http://s3.amazonaws.com/mairesse/research/index.html) in 2007 called [Personality Recognizer](http://s3.amazonaws.com/mairesse/research/personality/recognizer.html). 

Most of this work was straight forward after painfully figuring out how to use the IBM API that wasn't documented very well in 2016.

[Here](https://docs.google.com/presentation/d/e/2PACX-1vTk1dvbbTXZh_DJyPj9RsftUu7qV3GYhfRpl0VbSTtzPlTTOzeawbmX-o1aFWib7giWZkyF7oY-Urfk/pub?start=false&loop=false&delayms=3000&slide=id.p) is a presentation comparing the two algorithms. It highlights how the models are built from training data and compares outputs.

### Personality Scores
The output of Mairesse's 2007 Personality Recognizer are on a scale from 1 to 10 and that of IBM are from 0 to 1. The higher score in a category represents higher likelihood of that personality trait being evident in the text. 

Note: Extraversion is also spelt as extroversion. 

Extraversion - extr  Stability or Neurotism - stab  Agreeableness - agre  Conscientiousness - cons  Openness - open


| ticker | ceo           | extr  | stab  | agre  | cons  | open  |   | ibm_openn | ibm_consc | ibm_extra | ibm_agree | ibm_emoti |
|--------|---------------|-------|-------|-------|-------|-------|---|-----------|-----------|-----------|-----------|-----------|
| CMA-N  | Ralph-Babb    | 9.478 | 3.803 | 1.619 | 8.514 | 6.51  |   | 0.733813  | 0.885174  | 0.853204  | 0.808269  | 0.700272  |
| BKE-N  | Dennis-Nelson | 9.351 | 3.101 | 3.344 | 7.865 | 6.933 |   | 0.764356  | 0.933339  | 0.668271  | 0.630478  | 0.753369  |
| BOH-N  | Al-Landon     | 9.002 | 3.891 | 3.195 | 7.13  | 6.457 |   | 0.72944   | 0.634533  | 0.665527  | 0.571722  | 0.637639  |
| VSI-N  | Tom-Tolworthy | 4.745 | 0.858 | 2.243 | 4.96  | 7.917 |   | 0.225339  | 0.980869  | 0.727986  | 0.775569  | 0.917083  |
| NKE-N  | Mark-Parker   | 4.744 | 2.061 | 2.543 | 4.012 | 7.135 |   | 0.592408  | 0.471671  | 0.693078  | 0.510776  | 0.349136  |
| RRD-N  | Bob-Johnson   | 4.739 | 1.729 | 2.652 | 5.238 | 6.628 |   | 0.740491  | 0.957471  | 0.520197  | 0.521162  | 0.837391  |

Naturally there wasn't a strong correlation between the two types of scores because the two models are fundamentally different in how they associate text features with personality ratings. The two models also use different sets of text features to begin with. Though GloVe is now the basis of IBM Watson's text analysis algorithms, it is more popular and is technologically superior to LIWC based methods, the approach shown by Mairesse in 2007 was pioneering for its day. Mairesse contributed some of the general techniques still used in doing tone or sentiment analysis from text data. Of course much more modern methods with deep learning and abstraction hierarchies exist now a days in 2018. 

## Insights
The process of gathering data and crafting machine learning models is a very laborious one but once a model is created a bunch of interesting results can be obtained from large data that would otherwise be impossible to notice with the naked eye. Personality scores for each CEO from the Mairesse 2007 app were then compared with CEO performance. It was evident that higher extraversion scores correlated more significantly with higher paid CEOs who also had participated in higher number of successful acquisitions or mergers for their company. 

## Other Insights
While combing through the data I noticed a few things (things not required by the study but intriguing):
* There have been only a handful of CEOs with African descent in the past decades. The ones that do or have existed were mostly in very advanced and progressive companies like IBM or Microsoft. (IBM and Microsoft look drastically diverse compared to most S&P companies from the period 2004-2015).
* There were very few women CEOs in the past decades among S&P 1500 companies. 
  * Their extraversion scores in general are much lower than than most male CEOs. 
  * They are far more emotionally stable with a much lower emotional stability/neuroticism score (most male CEOs have medium or high neuroticism score. Higher scores indicate a more neurotic personality), 
  * Women had similar agreeableness to men, with high scores (high score indicates competitiveness and assertiveness), 
  * Women had similar conscientiousness to men, with high scores 
  * Women were open to new views or experiences, with a high scores.   


Also,
* The mere presence or absence of a CEO is worth noting. The CEO is often present when the news to be delivered is either very good or very challenging.
* Many pharma and tech companies always have their CEO present in each earnings calls.
* Exxon Mobil has never sent an executive to a conference call. Only a vice-president shows up to the calls and usually it is a different one in every other call.
* Content with a negative sentiment is often delivered on a Friday afternoon near the closing time. Supposedly so that the market can digest the news over the weekend and not have a kneejerk reaction. 
* Many companies coincide their bad news with a much more interesting bad news from another company. 
* Many executives across companies share family ties (same family name and usually brothers). It isn't surprising that the pool of executives isn't particularly diverse. 
* The analysts are ruthless and brutal. Their questions are very incisive and they usually don't excuse the CEO for having a generic answer, and ask lots of follow-up questions for details. At which point, the CEO usually says something like "I'll let the CFO take that question". Most of the Q&A with neutral or negative sentiment are handled by a COO or CFO or somebody else with a more detailed knowledge of the company's day to day operations.
* While all CEOs have much higher extraversion and openness scores than the general public, their agreeableness scores are much lower. Meaning that they will assess and evaluate a new opinion that may be different than their own belief, and will eventually assert their existing belief. If their belief has changed due to new opinions, the new opinion becomes theirs and isn't advertised as an opinion of the orignal commentator. 

## Conclusion
While this work is very interesting and fun it is still at the level of academic research and the more lucrative application of such procedures are being explored in the area of:
* Detecting sudden emergence of speech impediments as a marker for cognitive impairment
* Sentiment analysis for call centre or customer service representatives to devise better techniques in handling tough customers 
* Good old Amazon Alexa and Google Assistant that monitor every sound through various devices to achieve things that are yet to be imagined. 

There is a very bright future in this field for those excited by and skilled in machine learning techniques. 
