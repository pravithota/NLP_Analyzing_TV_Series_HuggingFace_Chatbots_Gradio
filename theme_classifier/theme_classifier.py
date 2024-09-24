import torch
import nltk
from nltk import sent_tokenize
import numpy as np
from transformers import pipeline
import os
import sys
import pathlib
import pandas as pd
folder_path = pathlib.Path(__file__).parent.resolve()
sys.path.append(os.path.join(folder_path, '../'))
from utils import load_subtitles_dataset
nltk.download('punkt')
nltk.download('punkt_tab')

class ThemeClassifier():
    def __init__(self, theme_list):
        self.model_name = "facebook/bart-large-mnli"
        #self.device = 0 if torch.cuda.is_available() else 'cpu'
        self.theme_list = theme_list
        self.theme_classifier = self.load_model()
        
    def load_model(self):
        theme_classifier = pipeline(
            "zero-shot-classification", 
             model=self.model_name  
             )
        return theme_classifier
    
    def get_themes_inference(self, script):
        script_sentences = sent_tokenize(script[0])
        print(script_sentences)
        #Batch Sentence
        sentence_batch_size = 20
        script_btaches=[]
        for index in range(0, len(script_sentences), sentence_batch_size):
            sent = " ".join(script_sentences[index:index+sentence_batch_size])
            script_btaches.append(sent)
        # Run Model
        #pipe = self.load_model()
        theme_out = self.theme_classifier(
            script_btaches[:2],
            self.theme_list,
            multi_label=True
        )
        # Wrangle output
        themes = {}
        for output in theme_out:
            for label,score in zip(output['labels'], output['scores']):
                if label not in themes:
                    themes[label] = []
                themes[label].append(score)   
        themes = {key: np.mean(np.array(val)) for key,val in themes.items()}
        return themes 
    def get_themes(self, dataset_path, save_path=None):

        # Read Save Path if exists
        #if save_path and os.path.exists(save_path):
            #return pd.read_csv(save_path)
        
        # Load Model
        df = load_subtitles_dataset(dataset_path)
        df = df.head(2)
        # Run inference
        output_themes = df['script'].apply(self.get_themes_inference)

        themes_df = pd.DataFrame(output_themes.tolist())
        df[themes_df.columns] = themes_df
        #print(df)
        # Save output
        #if save_path:
            #df.to_csv(save_path, index=True)

        return df
    
if __name__ == '__main__':
    '''theme_list = ['betrayal','battle','sacrifice','self development','friendhip','hate','happy']
    theme_classifier = ThemeClassifier(theme_list)
    dataset_path = r'D:\\DataScience\\DeepLearning\\NLP_TV_Series_Project\\data\\Subtitles'
    save_path = 'D:\\DataScience\\DeepLearning\\NLP_TV_Series_Project\\stubs'
    df = theme_classifier.get_themes(dataset_path, save_path)'''
    #print(df)    


