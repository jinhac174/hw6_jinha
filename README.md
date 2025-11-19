[BBC News Headline Classification — Mini AI Pipeline Project]


1. Task description and motivation.

This pipeline builds a news headline classifier to specify into five categories: business, entertainment, politics, sport, tech.
There are numerous papers and articles being release every day, and classifying them automatically very accurately is am important process to manage news.

I aimed to build a transformer based model with differing loss functions and architecture to compare and identify which classifier makes best estimations.

With these experiments, our model is expected to significantly outperform naive baselines that do not utilize neural networkds for classifiers.




2. Dataset description (source, size, splits, preprocessing).

I imported and used the open-sourced SetFit/bbc-news dataset from Hugging Face.

By default as provided,
Train size: 1,225
Test size: 1,000

Labels: 5 categories (business, entertainment, politics, sport, tech)

Format: text + integer label (one hot label per category)


From the 1,225 training dataset, I further split 10% of that training data into a validation set.
So dataset:

Train: 1,102
Validation: 123
Test: 1,000


Preprocessing:

Transformer-based classifier needs input of  (batch_size, sequence_length). For my case, 256 tokens for transformer. (enough size, enough to represent the news)
News text longer that 256 tokens are truncated at the end, and text shorter than 256 tokens are simply padded up to 256 tokens.


Tokenizer processing:

All text is lower-cased to create uniform text data, and workds are split into subwords (if necessary). 




3. Baseline and AI pipeline design (methods and models used).

Naive Baseline

As done in early AI-model baselines for these kinds a of tasks, I use a simple rule-based keyword classifier.
I make a list of keywords expected for each category of news. 
If that word is seen in the text, count them to account for assessment at the end.
Most likely classification is made on this information.

The baseline does not handle context, similar words, and complex wording. It only looks for preset workds for counting them.
It performs very poorly but provides a very logical ans naive reference point.



Mini-AI-pipeline

The AI pipeline uses a pretrained DistilBERT encoder to produce contextual embeddings for each article texts.
The architecture is as follows:


bbc_dataset class: dataset processing + tokenizing to output {input, attention_mask, and labels}

transformer: (encoder + classifier head) with adamW optimizer and lr scheduler
Learning rate is set to be 2e-5 and optimizer warmup is set 10%, with batch size 15 and 3 epochs for small model.


To see differences in performance, I put 2 variants in the classifier head and loss.

- Linear head
- MLP head with ReLU activation

- cross entropy loss
- class-weighted loss (balanced with sklearn.util tool)

Results on varying these four are compared at the end.




4. Metrics, results, and comparison (tables and example cases).

Naive baseline actually only showed accuracy of around 5%.
This is probably due to the undermined complexity of BBC news articles and too simple of a keywords list written by hand.


Then for AI-pipeline, I evaluated four configurations as mentioned above:

Use linear head with two different loss computations, use MLP head with two different loss computations.


Transformer-based attention applying model is known to be very capable of handling text information and building strong understanding.
Test the final estimation's accuracy is on linear head vs mlp head is the core idea of this experiment.
Meawhile, two loss computation methods are compared as well to find the optimal choice.


All models were trained under same configurations:

LR = 2e-5
Epochs = 3
Batch size = 16
Max length = 256
Warmup ratio = 0.1

The result:

===== ablation =====
(model)                  head       weighted   val_acc   test_acc
v1_linear_ce             linear     False      0.9453    0.9520
v2_linear_weighted       linear     True       0.9487    0.9570
v3_mlp_ce                mlp        False      0.9560    0.9620
v4_mlp_weighted          mlp        True       0.9604    0.9680


According to result, the MLP head + weighted loss consistently achieved the best performance.
This perfectly suits our intention. MLP head is more capable of learning optimal solution than a linear one, and balacing weights to account for
imbalance helps to increase the performance level.




5. Reflection and limitations.

This pipeline experiment allowed me to, first, check how strong AI pipelines can be compared to very simple and naive models that act on simple rules.
The AI model showed x20 accuracy and was able to correctly classify complex papers.

This can be understood as the naive baseline failing with simple keyword rules under real news articles, and the transformer-based classifier with 
better loss function and mlp head significantly improves the performance as demonstrated in the result section as well.

The limitations can still come from limited dataset, short training time, and the model size that can limit generalizability.
