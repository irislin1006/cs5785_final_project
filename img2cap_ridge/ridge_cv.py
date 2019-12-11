from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from numpy import linalg as LA
import numpy as np
import time
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import os

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
        path = save_dir + 'ridge_cv.npy'
        np.save(path, cos_sim)
        print('Cosine sim of ridge save to '+path)
    print(cos_sim.shape)

    top20 = np.argsort(cos_sim, axis=1)[:, ::-1][:, :20]
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

def cv(X_train,y_train):
    pipe_svr = Pipeline([('scl', StandardScaler()),
        ('reg', Ridge())])

    alphas = np.array([1,0.1,0.01,0.001,0.0001,0])
    params = {
        "reg__alpha" : alphas
    }
    gs_svr = GridSearchCV(estimator=pipe_svr, param_grid=params)

    gs_svr = gs_svr.fit(X_train,y_train)
    return gs_svr.best_estimator_


train = pickle.load(open('train.pkl', 'rb'))
test = pickle.load(open('test.pkl', 'rb'))

print(train['captions'][0])
print(train['tags'][0])

# print(train['image_2048'].shape)

print("=== fit model  ===")
ridge = cv(train['image_2048'], train['captions'])
t = time.time()
ridge.fit(train['image_2048'], train['captions'])
print(time.time()-t)


save_dir = "./ridge_cv_models/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
path = save_dir + 'ridge_cv.pkl'
pickle.dump(ridge, open(path, 'wb'))
print("Ridge model save to "+path)

print("=== test training data ===")
predict_caps = ridge.predict(train['image_2048'])
print(predict_caps.shape)
ranking(train['captions'], predict_caps, train['image_id'], 'training_test_answer.csv')


print("=== test testing data ===")
predict_caps = ridge.predict(test['image_2048'])
print(predict_caps.shape)
#ranking(predict_image_features, test['image_2048'], test['image_id'], 'answer.csv')
ranking(test['captions'], predict_caps, test['image_id'], 'answer.csv', True)

