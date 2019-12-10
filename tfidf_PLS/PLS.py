from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from numpy import linalg as LA
import numpy as np
import time
import pickle
import torch


def ranking(query_embedds, target_embedds, img_ids, file_name):
    """
    @ param query_embedds = (n, d)
    @ param target_embedds = (n, d)
    @ param img_ids = (n,)
    """
    print(query_embedds.shape)
    print(target_embedds.shape)

    cos_sim = cosine_similarity(query_embedds, target_embedds)
    print(cos_sim.shape)

    top20 = np.argsort(cos_sim, axis=1)[:, :20]
    # top20 = idx.cpu().numpy()
    print(top20.shape)

    img_ids = np.array(img_ids)
    count = 0
    with open(file_name, 'w') as f:
        f.write("Descritpion_ID,Top_20_Image_IDs\n")
        for i, img_id in enumerate(img_ids):
            top_imgs = img_ids[top20[i]]
            top_imgs_str = " ".join(list(top_imgs))
            text_id = img_id.split(".")[0]+".txt"
            f.write(text_id+","+top_imgs_str+"\n")
            if img_id in list(top_imgs):
                count+=1
        print("count", count)




train = pickle.load(open('train.pkl', 'rb'))
test = pickle.load(open('test.pkl', 'rb'))

print(train['captions'][0])
print(train['tags'][0])
print(train['image_2048'])
# print(train['image_2048'].shape)

print("=== fit model  ===")
pls2 = PLSRegression(n_components=1000)
t = time.time()
pls2.fit(train['image_2048'], train['captions'])
# ridge.fit(train['captions'], train['image_2048'])
print(time.time()-t)

print("=== test training data ===")
predict_captions = pls2.predict(train['image_2048'])
# predict_image_features = ridge.predict(train['captions'])
print(predict_captions.shape)
ranking(train['captions'], predict_captions, train['image_id'], 'training_test_answer.csv')


print("=== test testing data ===")
predict_captions = pls2.predict(test['image_2048'])
print(predict_captions.shape)
ranking(test['captions'], predict_captions, test['image_id'], 'answer.csv')

