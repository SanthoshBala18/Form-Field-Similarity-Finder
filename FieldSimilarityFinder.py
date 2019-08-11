from flair.embeddings import FlairEmbeddings, BertEmbeddings, WordEmbeddings, StackedEmbeddings, DocumentPoolEmbeddings, ELMoEmbeddings
from flair.embeddings import Sentence
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from math import*
import operator

class FormFieldSimilarityFinder:
  """
  The purpose of this class is to generate a Vector for each of the form field based on 3 predefined label description
  and store those in a pickle file
  """
  def __init__(self):
    # Initialize Form fields and their description
    self.name_field = ['Name of a person','A word by which a person is known',"Identity to call a person"]
    self.age_field = ['Age of a person','Number which tells how old a person is','The length of time a person has lived']
    self.address_field = ['Home Address of a person','A place of residence','Place of stay']
    
    # Intialize a dictionary with form fields and corresponding description list
    self.form_fields = {'Name':self.name_field, 'Age':self.age_field, 'Address':self.address_field}
    
    # Load all the Pretrained-Models
    self.elmo_embedding = ELMoEmbeddings()
    self.flair_forward_embedding = FlairEmbeddings('multi-forward')
    self.flair_backward_embedding = FlairEmbeddings('multi-backward')
    self.bert_embedding = BertEmbeddings('bert-base-multilingual-uncased')
    
    # Stack all the embeddings using DocumentPoolEmbeddings
    self.stacked_embedding = DocumentPoolEmbeddings(embeddings=[self.elmo_embedding,
                                                       self.flair_forward_embedding,self.flair_backward_embedding,self.bert_embedding])
    # A threshold value, only above which the match is considered
    self.threshold_value = 0.70
    
  def construct_vector(self, original_sentence):
    """
    Given a sentence, Contruct and return a vector based on different stacked embeddings
    """
    
    sentence = Sentence(original_sentence)
    self.stacked_embedding.embed(sentence)
    sentence_embedding = sentence.get_embedding()
    sentence_embedding_array = sentence_embedding.detach().numpy()

    return sentence_embedding_array
   
  
  def construct_category_vector(self, category_definitions):
    """
    Given a set of Category definitions, construct vector for each using Stacked embedding and return mean of all the vectors
    """
    
    category_vectors = []
    for each in category_definitions:
        sentence_embedding_array = self.construct_vector(each)
        category_vectors.append(sentence_embedding_array)
        single_vector = np.mean(category_vectors,0)
    return single_vector

  def store_category_vectors(self):
      """
      Build a Vector for each category and store it in a npz file
      """
      field_vector_dict = {}
      for field, description_list in self.form_fields.items():
        # Get a vector for each of the category using Stacked Embedding
        field_vector = construct_category_vector(description_list)
        field_vector_dict[field] = field_vector
      
      np.savez("field_vector.npz",**field_vector_dict)
  
  @staticmethod
  def find_similarity(vector1, vector2, method = "cosine"):
    """
    Find Similarity between two vectors based on the given similarity measure
    """
    sim_score = 0
    if "cosine":
      sim_score = cosine_similarity(vector1, vector2)
    elif "manhattan":
      sim_score =  sum(abs(val1-val2) for val1,val2 in zip(vector1,vector2))

    return sim_score
        
  def find_matching_field(self, user_field):
    """
    Method to find the closest matching field for a given form field
    """
    field_vectors = np.load('field_vector.npz')
    
    user_field_vector = self.construct_vector(user_field)
    similarity_dict = {}
    for field, vector in field_vectors.items():
      similarity_dict[field] = find_similarity(vector.reshape(1,-1),user_field_vector.reshape(1,-1))
    
    similarity_dict = {key: value for key, value in similarity_dict.items() if value>self.threshold_value}
    
    if similarity_dict:
      max_pair = max(similarity_dict.items(), key=operator.itemgetter(1))
      confidence = float("{0:.2f}".format(max_pair[1][0][0]))*100
      print(f"Closest Match to the field is '{max_pair[0]}' with confidence: {confidence}%")
      return max_pair
    else:
      print("No Confident Match is found!!!")
      return None
  
