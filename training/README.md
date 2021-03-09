# Generate MCQ questions with T5 transformer



## Training the model
The training and validation datasets are present in the **squad_t5** folder.
Install the necessary libraries from **requirements.txt**.
Use any **GPU** machine and run the command **python train.py**

If you want to look at all the arguments that you can override just run **python train.py --help**

Training this model for 3 epochs (default) took about 24 hrs on **p2.xlarge** (AWS ec2)


##Understanding Code :
Code is written in Pytorch Lightning using Huggingface Transformers. We use 
the T5 (text-to-Text transfer transformer) model to train a sequence to sequence 
task using Squad Dataset.

The "QuestionGenerationDataset" class in train.py is the main class than needs to be changed 
to train with a different dataset.

Follow this video to know more on Pytorch Lightning -
https://www.youtube.com/watch?v=QHww1JH7IDU

Follow this blog on understanding T5 and training a text to text task -
[T5 transformer](https://towardsdatascience.com/paraphrase-any-question-with-t5-text-to-text-transfer-transformer-pretrained-model-and-cbb9e35f1555?source=friends_link&sk=3bbd1018eba7a0ef68beaac066e5e8e2)

##Training format
The input and target(output) are both text sequences to train the T5 model.

**Input format given for training :**

input = "context: %s  answer: %s </s>" % (passage, answer)

**Output format given for training :**

target = "question: %s </s>" % (str(question))
 
