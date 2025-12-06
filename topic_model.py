from gensim import corpora
from gensim.models import LdaModel

class TopicModel:
    ''' Topic Modeling using LDA on complaint texts.'''
    def __init__(self, num_topics=5, passes=10):
        self.num_topics = num_topics
        self.passes = passes
        self.model = None
        self.dictionary = None
        self.corpus = None

    def prepare_corpus(self, df, text_column="cleaned_text"):
        '''
        Prepare corpus and dictionary for LDA from the cleaned text data.
        '''
        df_small = df.sample(n=200_000, random_state=42)
        texts = [text.split() for text in df_small[text_column]]    
        self.dictionary = corpora.Dictionary(texts)
        self.corpus = [self.dictionary.doc2bow(text) for text in texts]
        return self.corpus

    def train_lda(self):
        ''' 
        Train the LDA model on the prepared corpus.
        '''
        print("📚 Training LDA topic model...")
        self.model = LdaModel(corpus=self.corpus, num_topics=self.num_topics,
                              id2word=self.dictionary, passes=self.passes, random_state=42)
        print("✅ LDA training complete")
        return self.model

    def show_topics(self, num_words=10):
        ''' 
        Display the topics discovered by the LDA model.
        '''
        topics = self.model.show_topics(num_words=num_words)
        for topic_num, topic in topics:
            print(f"Topic {topic_num}: {topic}")
