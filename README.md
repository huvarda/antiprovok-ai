# antiprovok-ai
TAMIRCV TAKIMI PROJESI

A program to detect if social media posts/comments are trying to provke people or not

Made for Turkish Language Processing Competition 2021

www.acikhack.com.tr

Link to trained model weights:
https://drive.google.com/file/d/1FGhe-WVRiMeeuDyLWG0-2PakYFK7JNNn/view

# Instructions
Kendiniz train etmek isterseniz tweetSentiment.py programı çalıştırılarak train edilebilir, yoksa hazır model weightlerini indirip çalıştırabilirsiniz
Bundan sonra testModel.ipynb notebook'undaki

```
filename = "output/model.bin"
model = SentimentClassifier(2)
model.load_state_dict(torch.load(filename))
model = model.to(device)
```

kod blockunda filename'i modelin nereye kayıtlı olduğu şeklinde değiştirip blockları sırasıyla run ederek test edebilirsiniz 
