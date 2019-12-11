from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from numpy import linalg as LA
import numpy as np
import time
import pickle
import torch
import os
import xgboost as xgb

def ranking(query_embedds, target_embedds, img_ids, file_name, testing=False):
    """
    @ param query_embedds = (n, d)
    @ param target_embedds = (n, d)
    @ param img_ids = (n,)
    """
    print(query_embedds.shape)
    print(target_embedds.shape)

    cos_sim = cosine_similarity(query_embedds, target_embedds)
    if testing:
        save_dir = ".././cos_sim/"
        if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        path = save_dir + 'xgb'
        np.save(path, cos_sim)

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
X_train, Y_train = train['image_2048'], train['captions']
test = pickle.load(open('test.pkl', 'rb'))
X_test, Y_test = test['image_2048'], test['captions']

print(train['captions'][0])
print(train['tags'][0])

print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

dtrain = xgb.DMatrix(X_train, Y_train)
dtest = xgb.DMatrix(X_test, Y_test)

xgb_params = {
    'n_trees': 520, 
    'eta': 0.0045,
    'max_depth': 4,
    'subsample': 0.93,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}
steps = 1250
# print(train['image_2048'].shape)

print("=== fit model  ===")
# model = xgb.train(xgb_params, dtrain, steps)
model_dir = './modelsxgboost_trees_520_depth_4'
print("Loading model")
model = xgb.Booster({'nthread': 4})
model.load_model(model_dir)
t = time.time()
# pls2.fit(train['image_2048'], train['captions'])
# ridge.fit(train['captions'], train['image_2048'])
print(time.time()-t)

print("=== test training data ===")
predict_captions = model.predict(dtrain)
print(predict_captions)
print(predict_captions.shape)
train_best_preds = np.asarray([np.argmax(line) for line in predict_captions])
ranking(train['captions'], train_best_preds, train['image_id'], 'training_test_answer.csv')


print("=== test testing data ===")
predict_captions = model.predict(dtest)
test_best_preds = np.asarray([np.argmax(line) for line in predict_captions])
print(predict_captions.shape)
ranking(test['captions'], test_best_preds, test['image_id'], 'answer.csv', True)

