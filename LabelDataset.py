from sentence_transformers import SentenceTransformer, util
from PIL import Image
import os
import torch
import sys
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

def main():
    # We use the original CLIP model for computing image embeddings and English text embeddings
    #print( " using clip")
    en_model = SentenceTransformer('clip-ViT-B-32')

    img_names = []
    for f in listdir('SampleImages/'):
        if not f.startswith('.') and isfile(join('SampleImages/', f)):
            img_names.append(join('SampleImages/', f))
    img_names = sorted(img_names)
    #print(img_names)

    for filename in img_names:
        if not os.path.exists(filename):
            print("Error: file",filename,"does not exist")

    # Compute the embeddings for these images
    img_emb = en_model.encode([Image.open(filepath) for filepath in img_names], convert_to_tensor=True)

    # Then, we define our labels as text. Here, we use these labels
    labels = ['person', 'human', 'male', 'female', 'woman', 'nature', 'outdoor', 'indoor', 'car','kitchen','urban','inside home']

    # Compute the text embeddings for these labels
    en_emb = en_model.encode(labels, convert_to_tensor=True)

    # Now, we compute the cosine similarity between the images and the labels
    cos_scores = util.cos_sim(img_emb, en_emb)

    #print(cos_scores)

    
    #pred_labels = torch.argmax(cos_scores, dim=1)

    #print(pred_labels)
    top_n_labels = 5
    threshold = 0.1840

    # Then we look which labels has the highest cosine similarities with the given images and score is above threshold 
    for index,img_name in enumerate(img_names):
        #print(len(cos_scores[index]))
        #print("---- IMAGE NAME:", img_name,":",cos_scores[index])
        print(index,"):IMAGE NAME:", img_name)

        img_cos_scores = cos_scores[index]
        img_cos_scores_sorted = img_cos_scores.argsort(dim=0, descending=True)[:top_n_labels]
        keyword_list = ""
        for label_index in img_cos_scores_sorted:
            if(cos_scores[index][label_index].item() > threshold):
                #print(labels[label_index],":", cos_scores[index][label_index].item())
                if keyword_list == "":
                    keyword_list = labels[label_index]
                else:
                    keyword_list = keyword_list + "," +labels[label_index]
        
        # Load image as greyscale
        im = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
        if im is None:
            print(f'ERROR: Unable to load {img_name}')
            continue
        # Calculate mean brightness as percentage
        meanpercent = np.mean(im) * 100 / 255
        #print(meanpercent)
        if(meanpercent < 20):
            light_classification = "low light"
        elif (meanpercent >= 20 and meanpercent < 60):
            light_classification = "mid light"
        else:
            light_classification = "bright light" 
        print("\t",light_classification,",labels:",keyword_list)        
        #print(f'{img_name}: {light_classification} ({meanpercent:.1f}%)')

if __name__ == '__main__':
    main()



