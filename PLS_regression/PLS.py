from sklearn.cross_decomposition import PLSRegression
import numpy as np
import time
import pickle
from build_vocab import Vocabulary
def ranking(query_embedds, target_embedds, img_ids):
    """
    @ param query_embedds = (n, d)
    @ param target_embedds = (n, d)
    @ param img_ids = (n,)
    """

    cos_sim = torch.mm(query_embedds,target_embedds.T)/ \
                torch.mm(query_embedds.norm(2, dim=1, keepdim=True),
                        target_embedds.norm(2, dim=1, keepdim=True).T)
    _, idx = torch.topk(cos_sim, len(query_embedds)//100, dim=1)
    top20 = idx.cpu().numpy()
    img_ids = np.array(img_ids)
    count = 0
    with open('answer.csv', 'w') as f:
        f.write("Descritpion_ID,Top_20_Image_IDs\n")
        for i, img_id in enumerate(img_ids):
            top_imgs = img_ids[top20[i]]
            top_imgs_str = " ".join(list(top_imgs))
            text_id = img_id.split(".")[0]+".txt"
            f.write(text_id+","+top_imgs_str+"\n")
            if img_id in list(top_imgs):
                count+=1
        print("count", count)

print("----- Loading Vocab -----")
vocab_cap = pickle.load(open('vocab_cap.pkl', 'rb'))
vocab_tag = pickle.load(open('vocab_tag.pkl', 'rb'))
print(f"vocab cap size: {len(vocab_cap)}, vocab tag cap size: {len(vocab_tag)}")
print('----- Loading Note -----')

train = pickle.load(open('train.pkl', 'rb'))
test = pickle.load(open('test.pkl', 'rb'))

print(train['tags'][0])

print("=== fit model  ===")
pls2 = PLSRegression(n_components=20)
t = time.time()
pls2.fit(train['image_2048'], train['tags'])
print(time.time()-t)

print("=== test training data ===")
predict_tags = pls2.predict(train['image_2048'])
ranking(train['tags'], predict_tags, train['image_id'])


print("=== test testing data ===")
predict_tags = pls2.predict(test['image_2048'])
ranking(test['tags'], predict_tags, test['image_id'])

