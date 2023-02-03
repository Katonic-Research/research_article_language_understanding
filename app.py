import streamlit as st
from pathlib import Path
from PIL import Image
import plotly.graph_objects as go

st.set_page_config(
    page_title='Interactive research article using Streamlit',  
    layout = 'centered', 
    initial_sidebar_state = 'auto'
)
st.markdown('*Research article*')
st.title('Improving Language Understanding by Generative Pre-Training')
st.subheader("Authors")
st.warning('''
Alec Radford, OpenAI, alec@openai.com

Karthik Narasimhan, OpenAI, karthikn@openai.com

Tim Salimans, OpenAI, tim@openai.com

Ilya Sutskever, OpenAI, ilyasu@openai.com
''')

def read_markdown_file(markdown_file):
    return Path(markdown_file).read_text()

st.header('Abstract')
abstract = read_markdown_file("pages/abstract.md")
st.info(abstract)

st.markdown('''**Keywords:**, *NLP*, *BERT*, *“align mask, and select” (AMS)*, *commonsense reasoning(Stories Cloze Test)*, *question answering (RACE)*, *textual entailment (MultiNLI)*''')

st.header('Introduction')
intro_markdown = read_markdown_file("pages/introduction.md")
st.write(intro_markdown, unsafe_allow_html=True)

st.header('Related Work')
st.subheader('Semi-supervised learning for NLP')
with st.expander("Expand"):
    st.write('''
    Our work broadly falls under the category of semi-supervised learning for natural language. This paradigm has attracted significant interest, with applications to
    tasks like sequence labeling [24, 33, 57] or text classification [41, 70]. The earliest approaches used unlabeled data to compute word-level or phrase-level statistics, which were then used as features in a
    supervised model [33]. Over the last few years, researchers have demonstrated the benefits of using word embeddings [11, 39, 42], which are trained on unlabeled corpora, to improve performance on a
    variety of tasks [8, 11, 26, 45]. These approaches, however, mainly transfer word-level information,whereas we aim to capture higher-level semantics.
    Recent approaches have investigated learning and utilizing more than word-level semantics from unlabeled data. Phrase-level or sentence-level embeddings, which can be trained using an unlabeled
    corpus, have been used to encode text into suitable vector representations for various target tasks [28,32, 1, 36, 22, 12, 56, 31].
    ''')

st.subheader('Unsupervised pre-training')
with st.expander("Expand"):
    st.write('''
    Unsupervised pre-training is a special case of semi-supervised learning where the goal is to find a good initialization point instead of modifying the supervised learning
objective. Early works explored the use of the technique in image classification [20, 49, 63] and regression tasks [3]. Subsequent research [15] demonstrated that pre-training acts as a regularization
scheme, enabling better generalization in deep neural networks. In recent work, the method has been used to help train deep neural networks on various tasks like image classification [69], speech
recognition [68], entity disambiguation [17] and machine translation [48].The closest line of work to ours involves pre-training a neural network using a language modeling
objective and then fine-tuning it on a target task with supervision. Dai et al. [13] and Howard and Ruder [21] follow this method to improve text classification. However, although the pre-training
phase helps capture some linguistic information, their usage of LSTM models restricts their prediction ability to a short range. In contrast, our choice of transformer networks allows us to capture longerrange linguistic structure, as demonstrated in our experiments. Further, we also demonstrate the
effectiveness of our model on a wider range of tasks including natural language inference, paraphrase detection and story completion. Other approaches [43, 44, 38] use hidden representations from a pre-trained language or machine translation model as auxiliary features while training a supervised
model on the target task. This involves a substantial amount of new parameters for each separate target task, whereas we require minimal changes to our model architecture during transfer.
    ''')

st.subheader('Auxiliary training objectives')
with st.expander("Expand"):
    st.write('''
    Adding auxiliary unsupervised training objectives is an alternative form of semi-supervised learning. Early work by Collobert and Weston [10] used a wide variety of
    auxiliary NLP tasks such as POS tagging, chunking, named entity recognition, and language modeling to improve semantic role labeling. More recently, Rei [50] added an auxiliary language modeling
    objective to their target task objective and demonstrated performance gains on sequence labeling tasks. Our experiments also use an auxiliary objective, but as we show, unsupervised pre-training
    already learns several linguistic aspects relevant to target tasks.
    ''')

st.header('Framework')
st.write(
    '''
    Our training procedure consists of two stages. The first stage is learning a high-capacity language
    model on a large corpus of text. This is followed by a fine-tuning stage, where we adapt the model to
    a discriminative task with labeled data.
    '''
)
st.subheader('Unsupervised pre-training')
st.write('''
Given an unsupervised corpus of tokens U = {u1, . . . , un}, we use a standard language modeling
objective to maximize the following likelihood
    ''')
st.latex(r'''\begin{equation}L_{1}(U) = \sum_{i} \log P(u_{i}|u_{i-k},......,u_{i-1}|\theta)\end{equation}''')

st.write('''
where k is the size of the context window, and the conditional probability P is modeled using a neural network with parameters Θ. These parameters are trained using stochastic gradient descent [51].
In our experiments, we use a multi-layer Transformer decoder [34] for the language model, which is a variant of the transformer [62]. This model applies a multi-headed self-attention operation over the
input context tokens followed by position-wise feedforward layers to produce an output distribution over target tokens:
''')

st.latex(r'''h_{0} = UW_{\epsilon} + W_{p}''')
st.latex(r'''\begin{equation}h_{l} = transformerblock(h_{l-1}) \forall i \in\mathcal [1,n]\end{equation}''')
st.latex(r'''P_{u} = softmax(h_{n}W_{\epsilon}^T)''')
st.write('where')
st.latex(r'''U = u_{-k},......,u_{-1}''')

st.subheader(' Supervised fine-tuning')
st.write('''
After training the model with the objective in Eq. 1, we adapt the parameters to the supervised target task. We assume a labeled dataset C, where each instance consists of a sequence of input tokens,
x1, . . . , xm, along with a label y. The inputs are passed through our pre-trained model to obtain the final transformer block’s activation h
ml, which is then fed into an added linear output layer with parameters Wy to predict y:
    ''')

st.latex(r'''\begin{equation}P(y|x^{1},......,x^{m}) = softmax(h_{l}^mW_{y})\end{equation}''')
st.write('This gives us the following objective to maximize:')
st.latex(r'''\begin{equation}L_{2}(C) = \sum_{(x,y)} \log P(y|x^{1},......,x^{m})\end{equation}''')

st.write(
    '''
    We additionally found that including language modeling as an auxiliary objective to the fine-tuning helped learning by (a) improving generalization of the supervised model, and (b) accelerating
convergence. This is in line with prior work [50, 43], who also observed improved performance with such an auxiliary objective. Specifically, we optimize the following objective (with weight λ):
    '''
)

st.latex(r'''\begin{equation}L_{3}(C) = L_{2}(C) + \lambda L_{1}(C)\end{equation}''')
st.write('Overall, the only extra parameters we require during fine-tuning are Wy, and embeddings for delimiter tokens.')
st.image('image/fig-2.png')
st.caption('''
Figure 1: (left) Transformer architecture and training objectives used in this work. (right) Input
transformations for fine-tuning on different tasks. We convert all structured inputs into token
sequences to be processed by our pre-trained model, followed by a linear+softmax layer.

''')
st.subheader('Task-specific input transformations')
st.write('''
For some tasks, like text classification, we can directly fine-tune our model as described above. Certain other tasks, like question answering or textual entailment, have structured inputs such as
ordered sentence pairs, or triplets of document, question, and answers. Since our pre-trained model was trained on contiguous sequences of text, we require some modifications to apply it to these tasks.
Previous work proposed learning task specific architectures on top of transferred representations [44].Such an approach re-introduces a significant amount of task-specific customization and does not
use transfer learning for these additional architectural components. Instead, we use a traversal-style approach [52], where we convert structured inputs into an ordered sequence that our pre-trained
model can process. These input transformations allow us to avoid making extensive changes to the architecture across tasks. We provide a brief description of these input transformations below and
Figure 1 provides a visual illustration. All transformations include adding randomly initialized start and end tokens (<s>, <e>).

    ''')

st.write('**Textual entailment**')
with st.expander("Expand"):
    st.write('''
    For entailment tasks, we concatenate the premise p and hypothesis h token sequences, with a delimiter token ($) in between.
    ''')

st.write('**Similarity**')
with st.expander("Expand"):
    st.write('''
   For similarity tasks, there is no inherent ordering of the two sentences being compared.To reflect this, we modify the input sequence to contain both possible sentence orderings (with a
delimiter in between) and process each independently to produce two sequence representations (hml) which are added element-wise before being fed into the linear output layer
    ''')

st.write('**Question Answering and Commonsense Reasoning**')
with st.expander("Expand"):
    st.write('''
   For these tasks, we are given a context document z, a question q, and a set of possible answers {ak}. We concatenate the document context
and question with each possible answer, adding a delimiter token in between to get [z; q; $; ak]. Each of these sequences are processed independently with our model and then normalized via a softmax
layer to produce an output distribution over possible answers.
    ''')

st.header('Experiments')
exp_1 = read_markdown_file("pages/experiment.md")
st.write(exp_1, unsafe_allow_html=True)
st.image('image/table-3.png')
st.caption('''
Table 2: Experimental results on natural language inference tasks, comparing our model with current
state-of-the-art methods. 5x indicates an ensemble of 5 models. All datasets use accuracy as the
evaluation metric.
''')
exp_2 = read_markdown_file("pages/experiment_2.md")
st.write(exp_2, unsafe_allow_html=True)

st.image('image/table-4.png')
st.caption('''
Table 3: Results on question answering and commonsense reasoning, comparing our model with
current state-of-the-art methods.. 9x means an ensemble of 9 models.
''')

st.image('image/table-5.png')
st.caption('''
Table 4: Semantic similarity and classification results, comparing our model with current state-of-theart methods. All task evaluations in this table were done using the GLUE benchmark. (mc= Mathews
correlation, acc=Accuracy, pc=Pearson correlation)
''')

st.write('''
Overall, our approach achieves new state-of-the-art results in 9 out of the 12 datasets we evaluate on, outperforming ensembles in many cases. Our results also indicate that our approach works well
across datasets of different sizes, from smaller datasets such as STS-B (≈5.7k training examples) – to the largest one – SNLI (≈550k training examples).
''')
st.header('Analysis')
st.write('**Impact of number of layers transferred**')
st.write('''
   We observed the impact of transferring a variable number of layers from unsupervised pre-training to the supervised target task. Figure 2(left) illustrates the
performance of our approach on MultiNLI and RACE as a function of the number of layers transferred.We observe the standard result that transferring embeddings improves performance and that each
transformer layer provides further benefits up to 9% for full transfer on MultiNLI. This indicates that each layer in the pre-trained model contains useful functionality for solving target tasks.
    ''')

st.image('image/fig-1.png')
st.caption('''
Figure 2: (left) Effect of transferring increasing number of layers from the pre-trained language
model on RACE and MultiNLI. (right) Plot showing the evolution of zero-shot performance on
different tasks as a function of LM pre-training updates. Performance per task is normalized between
a random guess baseline and the current state-of-the-art with a single model.
''')

st.write('**Zero-shot Behaviors**')
with st.expander("Expand"):
    st.write('''
   We’d like to better understand why language model pre-training of transformers is effective. A hypothesis is that the underlying generative model learns to perform many of the
tasks we evaluate on in order to improve its language modeling capability and that the more structured attentional memory of the transformer assists in transfer compared to LSTMs. We designed a series
of heuristic solutions that use the underlying generative model to perform tasks without supervised finetuning. We visualize the effectiveness of these heuristic solutions over the course of generative
pre-training in Fig 2(right). We observe the performance of these heuristics is stable and steadily increases over training suggesting that generative pretraining supports the learning of a wide variety
of task relevant functionality. We also observe the LSTM exhibits higher variance in its zero-shot performance suggesting that the inductive bias of the Transformer architecture assists in transfer.
For CoLA (linguistic acceptability), examples are scored as the average token log-probability the generative model assigns and predictions are made by thresholding. For SST-2 (sentiment analysis),
we append the token very to each example and restrict the language model’s output distribution to only the words positive and negative and guess the token it assigns higher probability to as the prediction.
For RACE (question answering), we pick the answer the generative model assigns the highest average token log-probability when conditioned on the document and question. For DPRD [46] (winograd
schemas), we replace the definite pronoun with the two possible referrents and predict the resolution that the generative model assigns higher average token log-probability to the rest of the sequence
after the substitution.
    ''')
st.image('image/table-6.png')
st.caption('''
Table 5: Analysis of various model ablations on different tasks. Avg. score is a unweighted average
of all the results. (mc= Mathews correlation, acc=Accuracy, pc=Pearson correlation)
''')

st.write('**Ablation studies**')
with st.expander("Expand"):
    st.write('''
   We perform three different ablation studies (Table 5). First, we examine the performance of our method without the auxiliary LM objective during fine-tuning. We observe that
the auxiliary objective helps on the NLI tasks and QQP. Overall, the trend suggests that larger datasets benefit from the auxiliary objective but smaller datasets do not. Second, we analyze the effect of the
Transformer by comparing it with a single layer 2048 unit LSTM using the same framework. We observe a 5.6 average score drop when using the LSTM instead of the Transformer. The LSTM only
outperforms the Transformer on one dataset – MRPC. Finally, we also compare with our transformer architecture directly trained on supervised target tasks, without pre-training. We observe that the lack
of pre-training hurts performance across all the tasks, resulting in a 14.8% decrease compared to our full model
    ''')

st.header('Conclusion')
st.info('''
We introduced a framework for achieving strong natural language understanding with a single task-agnostic model through generative pre-training and discriminative fine-tuning. By pre-training
on a diverse corpus with long stretches of contiguous text our model acquires significant world knowledge and ability to process long-range dependencies which are then successfully transferred to
solving discriminative tasks such as question answering, semantic similarity assessment, entailment determination, and text classification, improving the state of the art on 9 of the 12 datasets we
study. Using unsupervised (pre-)training to boost performance on discriminative tasks has long been an important goal of Machine Learning research. Our work suggests that achieving significant
performance gains is indeed possible, and offers hints as to what models (Transformers) and data sets(text with long range dependencies) work best with this approach. We hope that this will help enable
new research into unsupervised learning, for both natural language understanding and other domains,further improving our understanding of how and when unsupervised learning works.
''')

st.header('Reference Link')
st.write("Research article -https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf")

st.header('References')

with st.expander("Expand"):
    references = read_markdown_file("pages/references.md")
    st.write(references, unsafe_allow_html=True)


















