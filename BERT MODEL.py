# -*- coding: utf-8 -*-

pip install transformers   # install huggingface transformers library

import torch              # using pytorch


# Using `BertForQuestionAnswering` class from the `transformers` library for QnA from the reference text
from transformers import BertForQuestionAnswering    #Using the BERT-large model with 24 layers and embedding size of 1024 for 340M parameters

#Model trained on SQUAD Version 1 
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

#loading the tokenizer that helps in differentiating questions and answers and helps in assigning scores to each word to get a clarified output

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
# asked a question from the reference text

question = "When symptoms begin?"
#reference text as the abstractive summary obtained from PEGASUS model
answer_text = "The main symptoms of TS are tics. Symptoms usually begin when a child is 5 to 10 years of age. The first symptoms often are motor tics that occur in the head and neck area. Tics usually are worse during times that are stressful or exciting. They tend to improve when a person is calm or focused on an activity.The types of tics and how often a person has tics changes a lot over time. Even though the symptoms might appear, disappear, and reappear, these conditions are considered chronic.In most cases, tics decrease during adolescence and early adulthood, and sometimes disappear entirely. However, many people with TS experience tics into adulthood and, in some cases, tics can become worse during adulthood.1Although the media often portray people with TS"


# Apply the tokenizer to the input text, treating them as a text-pair.
input_ids = tokenizer.encode(question, answer_text)

print('The input has a total of {:} tokens.'.format(len(input_ids)))
#printing token IDs to get a clear image of the role of tokens

# BERT only needs the token IDs, but for the purpose of inspecting the 
# tokenizer's behavior, let's also get the token strings and display them.
tokens = tokenizer.convert_ids_to_tokens(input_ids)

# For each token and its id...
for token, id in zip(tokens, input_ids):
    
    # If this is the [SEP] token, add some space around it to make it stand out as it tends to separate Question tokens from the answer tokens
    if id == tokenizer.sep_token_id:
        print('')
    
    # Print the token string and its ID in two columns.
    print('{:<12} {:>6,}'.format(token, id))

    if id == tokenizer.sep_token_id:
        print('')



# Search the input_ids for the first instance of the `[SEP]` token.
sep_index = input_ids.index(tokenizer.sep_token_id)

# The number of segment A tokens includes the [SEP] token istelf. That consists of the question tokens
num_seg_a = sep_index + 1

# The remainder are segment B. That consists of the answer tokens
num_seg_b = len(input_ids) - num_seg_a

# Construct the list of 0s and 1s.
segment_ids = [0]*num_seg_a + [1]*num_seg_b

# There should be a segment_id for every input token.
assert len(segment_ids) == len(input_ids)



# Running example through the model.
outputs = model(torch.tensor([input_ids]), # The tokens representing our input text.
                             token_type_ids=torch.tensor([segment_ids]), # The segment IDs to differentiate question from answer_text
                             return_dict=True) 

start_scores = outputs.start_logits
end_scores = outputs.end_logits
 

# Find the tokens with the highest `start` and `end` scores.
answer_start = torch.argmax(start_scores)
answer_end = torch.argmax(end_scores)

# Combine the tokens in the answer and print it out. So it goes from the beginning and finds the starting token with highest score
#Similary it starts from the end and finds the one with the highest score then prints all the letters between the 2 highest scoring tokens
answer = ' '.join(tokens[answer_start:answer_end+1])

print('Answer: "' + answer + '"')



# Start with the first token.
answer = tokens[answer_start]


for i in range(answer_start + 1, answer_end + 1):
    
    
    if tokens[i][0:2] == '##':
        answer += tokens[i][2:]
    
    
    else:
        answer += ' ' + tokens[i]

print('Answer: "' + answer + '"')

# With the help of googlecolab codes on net, I plotted the scores 

import matplotlib.pyplot as plt
import seaborn as sns

# Use plot styling from seaborn.
sns.set(style='darkgrid')

plt.rcParams["figure.figsize"] = (16,8)


# Pull the scores out of PyTorch Tensors and convert them to 1D numpy arrays.
s_scores = start_scores.detach().numpy().flatten()
e_scores = end_scores.detach().numpy().flatten()

token_labels = []
for (i, token) in enumerate(tokens):
    token_labels.append('{:} - {:>2}'.format(token, i))


#visualizing end and start scores on the plot

import pandas as pd

# Store the tokens and scores in a DataFrame. 
# Each token will have two rows, one for its start score and one for its end
# score. The "marker" column will differentiate them. 
scores = []
for (i, token_label) in enumerate(token_labels):

    # Add the token's start score as one row.
    scores.append({'token_label': token_label, 
                   'score': s_scores[i],
                   'marker': 'start'})
    
    # Add  the token's end score as another row.
    scores.append({'token_label': token_label, 
                   'score': e_scores[i],
                   'marker': 'end'})
    
df = pd.DataFrame(scores)

# Draw a grouped barplot to show start and end scores for each word.
# The "hue" parameter differentiates start and end tokens
g = sns.catplot(x="token_label", y="score", hue="marker", data=df,
                kind="bar", height=6, aspect=4)

g.set_xticklabels(g.ax.get_xticklabels(), rotation=90, ha="center")


g.ax.grid(True)


def answer_question(question, answer_text):
    

    input_ids = tokenizer.encode(question, answer_text)

    print('Query has {:,} tokens.\n'.format(len(input_ids)))

   
    sep_index = input_ids.index(tokenizer.sep_token_id)

    
    num_seg_a = sep_index + 1

    num_seg_b = len(input_ids) - num_seg_a

    
    segment_ids = [0]*num_seg_a + [1]*num_seg_b

    
    assert len(segment_ids) == len(input_ids)

    
    outputs = model(torch.tensor([input_ids]),
                    token_type_ids=torch.tensor([segment_ids]), 
                    return_dict=True) 

    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

   
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)

    
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    
    answer = tokens[answer_start]

    
    for i in range(answer_start + 1, answer_end + 1):
        
        
        if tokens[i][0:2] == '##':
            answer += tokens[i][2:]
        
       
        else:
            answer += ' ' + tokens[i]

    print('Answer: "' + answer + '"')

import textwrap

# Wrap text to 80 characters.
wrapper = textwrap.TextWrapper(width=80) 
## references texts taken from the answers obtained from the PEGASUS model which further got the data from articles on Pubmed
bert_abstract = "The main symptoms of TS are tics. Symptoms usually begin when a child is 5 to 10 years of age. The first symptoms often are motor tics that occur in the head and neck area. Tics usually are worse during times that are stressful or exciting. They tend to improve when a person is calm or focused on an activity.The types of tics and how often a person has tics changes a lot over time. Even though the symptoms might appear, disappear, and reappear, these conditions are considered chronic.In most cases, tics decrease during adolescence and early adulthood, and sometimes disappear entirely. However, many people with TS experience tics into adulthood and, in some cases, tics can become worse during adulthood.1Although the media often portray people with TS"
print(wrapper.fill(bert_abstract))


question = "When symptoms begin?"
question2= "WHen are tics worse?"
answer_question(question, bert_abstract)
answer_question(question2, bert_abstract)




'''reference from # Question Answering with a Fine-Tuned BERT colab notebook
*by Chris McCormick*
(https://youtu.be/l8ZYCvgGu0o)
(https://rajpurkar.github.io/SQuAD-explorer/explore/1.1/dev/Super_Bowl_50.html?model=r-net+%20(ensemble)%20(Microsoft%20Research%20Asia)&version=1.1) on the topic of Super Bowl 50.
(https://github.com/huggingface/transformers/)
(https://colab.research.google.com/github/omarsar/pytorch_notebooks/blob/master/pytorch_quick_start.ipynb?authuser=1#scrollTo=hhuQyU7AYE6)
'''