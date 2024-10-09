# Labelling Datasets using CLIP model

Objective of this project is to label a video/image dataset videos and images as per some tags that we provide. These tags can be based on our use case.<br />
In this project we use CLIP model and sentence transformer to label videos and images. Currently only supports images.<br />
We provide a list of labels in LabelDataset.py and sample input images in SampleImages folder. <br />
The model will then create vector embeddings for each sample image in the folder. Then we encode our list of labels/words/tags. Once done, we calculate cosine scores of similiarites between each image and each label.<br />
Then we can find top N labels with max cos score from our label list for each image predicted by the model to be similar to the image. This way we can tag images!<br />
I have also added a low light, mid light and bright light classification by calculating mean brightness in the greyscaled image.<br />
This is WIP project. <br />

Our ideal goal is to: 
1. take videos 
2. get frames 
3. compare frames with tags using CLIP model 
4. find the best tags that match the frames in the video(best scores) 
5. note down the tags for each video
6. Create parquet where each video has a row with metadata such as Asset_name, width, height, light level, size, md5, keywords, description, PII yes/no, face yes/no. etc.
