import os
import json
import sys
from pathlib import Path
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
from clip.open_clip import create_model_and_transforms as openclip_model_and_transform, get_tokenizer as openclip_tokenizer
from PIL import Image
import re
import torchvision.transforms as transforms
from itertools import zip_longest
def pre_caption(caption,max_words=50):
    caption = re.sub(
        r"([.!\"()*#:;~])",       
        ' ',
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    #truncate caption
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])
            
    return caption

class flickr30k_train(Dataset):
    def __init__(self, transform, image_root, ann_root, tokenizer, max_words=30, prompt='', unique=False, clip=True):        
        '''
        image_root (string): Root directory of images (e.g. flickr30k/)
        ann_root (string): directory to store the annotation file
        '''        
        # url = 'https://storage.googleapis.com/sfr-vision-language-research/datasets/flickr30k_train.json'
        filename = 'flickr30k_train.json'

        # download_url(url,ann_root)
        
        self.annotation = json.load(open(os.path.join(ann_root,filename),'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words      
        self.prompt = prompt
        self.tokenizer = tokenizer
        self.unique = unique

        if self.unique:
            unique_annotations = []
            seen_image_ids = set()
            for item in self.annotation:
                image_id = item.get('image_id')
                if image_id is not None and image_id not in seen_image_ids:
                    unique_annotations.append(item)
                    seen_image_ids.add(image_id)
            self.annotation = unique_annotations 
        elif clip:
            grouped_data = {}
            for item in self.annotation:
                if item['image_id'] not in grouped_data:
                    grouped_data[item['image_id']] = []
                grouped_data[item['image_id']].append(item)

            interleaved_data = []

            group_lists = [grouped_data[key] for key in sorted(grouped_data.keys())]

            for items in zip_longest(*group_lists, fillvalue=None):
                interleaved_data.extend([item for item in items if item is not None])
            self.annotation = interleaved_data

    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        ann = self.annotation[index]
        
        image_path = os.path.join(self.image_root,ann['image'])        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)
        
        caption = self.prompt+pre_caption(ann['caption'], self.max_words) 

        # return image, caption, self.img_ids[ann['image_id']] 
        return image, self.tokenizer([caption])[0]


class flickr30k_retrieval_eval(Dataset):
    def __init__(self, transform, image_root, ann_root, split, max_words=30):  
        '''
        image_root (string): Root directory of images (e.g. flickr30k/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        '''
        urls = {'val':'https://storage.googleapis.com/sfr-vision-language-research/datasets/flickr30k_val.json',
                'test':'https://storage.googleapis.com/sfr-vision-language-research/datasets/flickr30k_test.json'}
        filenames = {'val':'flickr30k_val.json','test':'flickr30k_test.json'}
        
        download_url(urls[split],ann_root)
        
        self.annotation = json.load(open(os.path.join(ann_root,filenames[split]),'r'))
        self.transform = transform
        self.image_root = image_root
        
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        
        txt_id = 0
        for img_id, ann in enumerate(self.annotation):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['caption']):
                self.text.append(pre_caption(caption,max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1
                                    
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        image_path = os.path.join(self.image_root, self.annotation[index]['image'])        
        image = Image.open(image_path).convert('RGB')    
        image = self.transform(image)  

        return image, index    
    

if __name__ == "__main__":
    # tokenizer = openclip_tokenizer("ViT-g-14")
    # dt = flickr30k_train(transform=transforms.PILToTensor(),
    #                 image_root="/public/scccse/dataset/Flick30k/Images/",
    #                 ann_root="/public/scccse/dataset/Flick30k",
    #                 tokenizer=tokenizer
    #                 )

    # print(len(dt))
    # print(dt[0])

    ann = json.load(open(os.path.join("/public/scccse/dataset/Flick30k/flickr30k_train.json"),'r'))
    print(len(ann))

    grouped_data = {}
    for item in ann:
        if item['image_id'] not in grouped_data:
            grouped_data[item['image_id']] = []
        grouped_data[item['image_id']].append(item)


    interleaved_data = []
    max_len = 5

    group_lists = [grouped_data[key] for key in sorted(grouped_data.keys())]


    for items in zip_longest(*group_lists, fillvalue=None):
        interleaved_data.extend([item for item in items if item is not None])

    print(interleaved_data[0])
    print(interleaved_data[1])
    print(interleaved_data[2])
    print(interleaved_data[3])
    print(interleaved_data[-3])
    print(interleaved_data[-2])
    print(interleaved_data[-1])