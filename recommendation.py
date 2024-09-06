import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import random

from pymongo import MongoClient

client = MongoClient('localhost', 27017)
cluster_db = client['clusters']
cluster_collection = cluster_db['mycollection']

user_db = client['users']
user_collection = user_db['mycollection']

posts_db = client['posts']
post_collection = posts_db['mycollection']

ratings_db = client['ratings']
rating_collection = ratings_db['mycollection']

class Database_utility:
    def __init__(self, cluster_collection, user_collection, post_collection, rating_collection):
        self.cluster_collection = cluster_collection
        self.user_collection = user_collection
        self.post_collection = post_collection
        self.rating_collection = rating_collection

    def get_post_ids(self):
        posts_with_ids = self.post_collection.find({}, {'id': 1, '_id': 0})
        return [post['id'] for post in posts_with_ids]
    
    def get_user_ids(self):
        users_with_ids = self.user_collection.find({}, {'id': 1, '_id': 0})
        return [user['id'] for user in users_with_ids]

    def get_users_count(self):
        return self.user_collection.count_documents({})

    def get_posts_count(self):
        return self.post_collection.count_documents({})

    def get_rating_count(self, post_id : int):
        return self.rating_collection.count_documents({"postId" : post_id})

    def get_avg_rating(self, post_id : int):
        pipeline = [
            {"$match": {"postId": post_id}},  
            {"$group": {  
                "_id": "$postId",  
                "averageRating": {"$avg": "$rating"}  
            }}
        ]

        result = self.rating_collection.aggregate(pipeline)
        for doc in result:
            return doc['averageRating']
        
        return None
    
class Loader(Dataset):
    def __init__(self, db_util: Database_utility):
        
        self.postId2idx = {post_id: idx for idx, post_id in enumerate(db_util.get_post_ids())}
        self.userId2idx = {user_id: idx for idx, user_id in enumerate(db_util.get_user_ids())}
        self.idx2postId = {idx: post_id for post_id, idx in self.postId2idx.items()}
        self.idx2userId = {idx: user_id for user_id, idx in self.userId2idx.items()}

        ratings_cursor = db_util.rating_collection.find({}, {"_id": 0, "userId": 1, "postId": 1, "rating": 1})
        ratings_list = list(ratings_cursor)

        for doc in ratings_list:
            doc['userId'] = self.userId2idx.get(doc['userId'])
            doc['postId'] = self.postId2idx.get(doc['postId'])

        self.x = np.array([(doc['userId'], doc['postId']) for doc in ratings_list if doc['userId'] is not None and doc['postId'] is not None])
        self.y = np.array([doc['rating'] for doc in ratings_list if doc['userId'] is not None and doc['postId'] is not None])

        self.x, self.y = torch.tensor(self.x, dtype = torch.long), torch.tensor(self.y, dtype = torch.float)

    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    def __len__(self):
        return len(self.y)
    
class MatrixFactorization(torch.nn.Module):
    def __init__(self, n_users, n_items, n_factors=20):
        super().__init__()
        self.user_factors = torch.nn.Embedding(n_users, n_factors)
        self.item_factors = torch.nn.Embedding(n_items, n_factors)
        self.user_factors.weight.data.uniform_(0, 0.05)
        self.item_factors.weight.data.uniform_(0, 0.05)

    def forward(self, data):
        users, items = data[:,0], data[:,1]
        return (self.user_factors(users)*self.item_factors(items)).sum(1)

    def predict(self, user, item):
        return self.forward(user, item)

class Training:
    def __init__(self, db_util : Database_utility, loss_fn = torch.nn.MSELoss(), optim = torch.optim.Adam, lr = 1e-3) -> None:
        
        self.db_util = db_util
        self.loss_fn = loss_fn
        self.optim = optim
        self.lr = lr

    def training(self, model : object, save_model_path : str, num_of_epochs = 120):

        train_loader = Loader(self.db_util)
        train_dataloader = DataLoader(train_loader, 128, shuffle=True)

        optimizer = self.optim(model.parameters(), lr=self.lr)

        for it in range(num_of_epochs):
            losses = []
            for x, y in train_dataloader:

                optimizer.zero_grad()
                outputs = model(x)
                loss = self.loss_fn(outputs.squeeze(), y.type(torch.float32))
                losses.append(loss.item())
                loss.backward()
                optimizer.step()

            print("iter #{}".format(it), "Loss:", sum(losses) / len(losses))

        torch.save(model, save_model_path)

    def initial_training(self, save_model_path, num_of_epochs = 120):
        n_users = self.db_util.get_users_count()
        n_posts = self.db_util.get_posts_count()

        model = MatrixFactorization(n_users, n_posts, n_factors = 8)
        self.training(model, save_model_path, num_of_epochs)

    def incremental_training(self, model_path, save_model_path, num_of_epochs = 120):
        model = torch.load(model_path)
        self.training(model, save_model_path, num_of_epochs)

def get_clusters(model_path : str, num_of_clusters : int, db_util : object):

    model = torch.load(model_path)
    trained_embeddings = model.item_factors.weight.data.cpu().numpy()
    print(len(trained_embeddings))
    kmeans = KMeans(n_clusters = num_of_clusters, random_state = 0).fit(trained_embeddings)

    train_set = Loader(db_util)

    clusters = []

    for cluster_idx in range(num_of_clusters):
        users = []

        for idx in np.where(kmeans.labels_ == cluster_idx)[0]:
            postId = train_set.idx2postId[idx]
            rat_count = db_util.get_rating_count(postId)
            users.append((postId, rat_count))

        cluster = [user[0] for user in sorted(users, key=lambda tup: tup[1], reverse=True)[:10]]
        clusters.append(cluster)

    return clusters

def update_cluster_db(model_path, num_of_clusters, train_ds, db_util : Database_utility):
    clusters = get_clusters(model_path, num_of_clusters, train_ds)

    for cluster_num, cluster in enumerate(clusters):
        if not db_util.cluster_collection.find_one({'cluster_num': cluster_num}):
            new_document = {
                'cluster_num': cluster_num,
                'post_ids': list(map(int, clusters))
            }
            db_util.cluster_collection.insert_one(new_document)

        for new_post_id in cluster:
            db_util.cluster_collection.update_one(
                {'cluster_num': cluster_num},
                {'$push': {
                    'post_ids': int(new_post_id)
                }})

class Feed:

    def __init__(self, posts, clusters, db) -> None:
        self.clusters = clusters
        self.displayed_posts = []
        self.posts_to_display = posts
        self.db = db

    def get_avg_rating(self, postid: int) -> float:
        if postid in self.posts_to_display:
            return self.db.loc[self.db["postId"] == postid, "rating"].mean()
        else:
            return -1.0

    def get_cluster_with_prob_p(self, cluster_num, p=0.5) -> list:
        random_num = random.randint(0, 1)
        if random_num <= p:
            return self.clusters[cluster_num]
        return []

    def get_best_post_in_cluster(self, cluster: list) -> any:
        return sorted(cluster, key=lambda x: self.get_avg_rating(x))[-1]

    def get_content(self, num_of_items: int, user_preferences: list) -> list:

        clusters_to_use = list(filter(lambda x: x != [],
                                [self.get_cluster_with_prob_p(cluster_num, cluster_prob) \
                                for cluster_num, cluster_prob in enumerate(user_preferences)]))

        while len(self.displayed_posts) < num_of_items:
            for cluster in clusters_to_use[:num_of_items]:
                self.displayed_posts.append(
                    self.get_best_post_in_cluster(cluster))

        return self.displayed_posts
