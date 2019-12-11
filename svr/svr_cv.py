from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from numpy import linalg as LA
import numpy as np
import time
import pickle
import torch
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

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
        path = save_dir + 'svr_cv'
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


def cv(X_train,y_train):
    pipe_svr = Pipeline([('scl', StandardScaler()),
        ('reg', MultiOutputRegressor(SVR()))], verbose=1)

    grid_param_svr = {
        'reg__estimator__C': [0.1, 1, 10]
    }

    gs_svr = GridSearchCV(estimator=pipe_svr,
                        param_grid=grid_param_svr,
                        cv=2,
                        scoring = 'neg_mean_squared_error',
                        n_jobs = -1,
                        verbose=1)

    gs_svr = gs_svr.fit(X_train,y_train)
    return gs_svr.best_estimator_


train = pickle.load(open('train.pkl', 'rb'))
test = pickle.load(open('test.pkl', 'rb'))

#deminsion reduction
pca = PCA(n_components=1000)
pca.fit(train['captions'])
train_caps = pca.transform(train['captions'])
test_caps = pca.transform(test['captions'])

print(train['captions'][0])
print(train['tags'][0])

print("=== fit model  ===")
# svr2 = MultiOutputRegressor(SVR(C=1.0, epsilon=0.2))
svr2 = cv(train['image_2048'], train_caps)
print(svr2)

#t = time.time()
#svr2.fit(train['image_2048'], train['captions'])
#print(time.time()-t)

save_dir = "./svr_cv_models/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
path = save_dir + 'svr_cv.pkl'
pickle.dump(svr2, open(path, 'wb'))
print("SVR model save to "+path)

print("=== test training data ===")
predict_captions = svr2.predict(train['image_2048'])
print(predict_captions.shape)
ranking(train_caps, predict_captions, train['image_id'], 'training_test_answer.csv')


print("=== test testing data ===")
predict_captions = svr2.predict(test['image_2048'])
print(predict_captions.shape)
ranking(test_caps, predict_captions, test['image_id'], 'answer.csv', True)

