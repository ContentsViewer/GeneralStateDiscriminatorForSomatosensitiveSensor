import tensorflow as tf
import numpy as np
import threading
import queue
import time
from itertools import product
from .stdlib.collections import dotdict
from .stdlib.stopwatch import stopwatch
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def triplet_loss(alpha=0.2):
    # 先に訓練データ(過去のデータ)とそれに対応する教師(正負)のセットを作る
    # そうすると, できるはず
    def loss(y_true, y_pred):
        """
        Parameters
        ----------
        y_true : [ positive, negative ]
        """
        # print(y_pred, y_true)

        # Note:
        #   Error:
        #       # " slice index 1 of dimension 0 out of bounds"
        #           >>>
        #             For me, the issue resolved when I changed for example, y_pred[0] to y_pred[:,0,:]. You might need to change your implementation suitably.
        #           >>>
        #           https://gist.github.com/JustinhoCHN/a243056d87f4c2728de9e7ea4923de01
        #
        anchor = y_pred[:,0,:]
        positive = y_true[:,0,:]
        negative = y_true[:,1,:]

        pos_dist = tf.math.reduce_sum(tf.math.square(
            tf.math.subtract(anchor, positive)), 1)
        neg_dist = tf.math.reduce_sum(tf.math.square(
            tf.math.subtract(anchor, negative)), 1)

        basic_loss = tf.math.add(tf.math.subtract(pos_dist, neg_dist), alpha)
        loss = tf.math.reduce_mean(tf.math.maximum(basic_loss, 0.0), 0)
        return loss

    return loss


def select_triplets(anchor_embedding, positive_embeddings, negative_embeddings, alpha=0.2):
    triplets = []
    neg_dists = tf.math.reduce_sum(tf.math.square(
        anchor_embedding - negative_embeddings), 1)
    pos_size = tf.shape(positive_embeddings)[0]
    for pos_idx in range(pos_size):
        pos_dist = tf.math.reduce_sum(tf.math.square(
            anchor_embedding - positive_embeddings[pos_idx]))
        neg_indices = np.where(neg_dists - pos_dist < alpha)[0]
        nrof_neg_indices = tf.shape(neg_indices)[0]
        if nrof_neg_indices > 0:
            rnd_idx = np.random.randint(nrof_neg_indices)
            neg_idx = neg_indices[rnd_idx]
            triplets.append(
                [anchor_embedding, positive_embeddings[pos_idx], negative_embeddings[neg_idx]])

    np.random.shuffle(triplets)
    return triplets



class Estimator():
    def __init__(self, model=None, precedents_dict=None):
        self.is_running = False
        self.results = queue.Queue()
        self.model = model
        self.inputs = None
        self.precedents_dict = precedents_dict
        self._prev_centers = None

    def run(self, inputs=None, supervised_state_label=None):
        if self.is_running:
            return
        self.is_running = True
        self.inputs = inputs
        self.supervised_state_label = supervised_state_label
        self.worker_thread = threading.Thread(target=self.worker)
        self.worker_thread.start()

    def worker(self):
        # print(self.inputs)
        [embedding], _ = self.model(self.inputs)

        # print(embedding)
        result = dotdict({
            'inputs': None,
            'embedding': None,
            'estimated_state': None,
            'clustered': None,
            'cluster_centers': None
        })
        n_states = len(self.precedents_dict) - 1
        result.inputs = self.inputs
        result.embedding = embedding
        result.estimated_state = n_states  # not categorized state
        result.supervised_state_label = self.supervised_state_label
        
        embeddings = []
        for precedents in self.precedents_dict:
            embeddings.extend(
                [precedent.embedding for precedent in precedents])
        embeddings.append(embedding)

        if len(embeddings) > n_states:
            kmeans = KMeans(
                n_clusters=n_states
            )
            
            result.clustered = kmeans.fit_predict(embeddings)

            id_remap = [i for i in range(len(kmeans.cluster_centers_))]

            if self._prev_centers is not None:
                result.cluster_centers = [None] * len(kmeans.cluster_centers_)
                pairs = nearest(kmeans.cluster_centers_, self._prev_centers)

                for ifrom, ito in pairs:
                    id_remap[ifrom] = ito
                    result.cluster_centers[ito] = kmeans.cluster_centers_[ifrom]
            else:
                result.cluster_centers = kmeans.cluster_centers_

            self._prev_centers = result.cluster_centers
            result.estimated_state = id_remap[result.clustered[-1]]
            result.id_remap = id_remap

            # print(result.clustered[-1])
            # print(kmeans.cluster_centers_[0], kmeans.cluster_centers_[1])
        # pca = PCA(n_components=2)
        # pca.fit(embedding)
        # reduced = pca.fit_transform(embedding)
        # print(reduced)

        self.results.put(result)
        self.is_running = False

class Trainor():
    def __init__(self, precedents_dict=None):
        self.is_running = False
        self.precedents_dict = precedents_dict
        self.results = queue.Queue()

    def run(self, model=None, anchor=None):
        if self.is_running:
            return
        self.is_running = True
        self.model = model
        self.anchor = anchor
        self.worker_thread = threading.Thread(target=self.worker)
        self.worker_thread.start()

    def worker(self):
        positive_embeddings = []
        negative_embeddings = []
        anchor_state = self.anchor.get('supervised_state')
        if anchor_state is None:
            anchor_state = self.anchor.get('estimated_state')

        for state, precedents in enumerate(self.precedents_dict):
            if state == len(self.precedents_dict) - 1:
                break
            if state == anchor_state:
                positive_embeddings.extend([precedent.embedding for precedent in precedents])
            else:
                negative_embeddings.extend([precedent.embedding for precedent in precedents])

        if len(positive_embeddings) <= 0 or len(negative_embeddings) <= 0:
            self.is_running = False
            return

        triplets = select_triplets(self.anchor.embedding, positive_embeddings, negative_embeddings)
        if len(triplets) <= 0:
            self.is_running = False
            return
        inputs = [self.anchor.inputs[0]] * len(triplets)
        targets = []
        for anchor, positive, negative in triplets:
            targets.append([positive, negative])

        inputs = np.array(inputs)
        targets = np.array(targets)

        self.model.fit(inputs, targets, batch_size=1)
        self.is_running = False

def nearest(a, b):
    na, nb = len(a), len(b)

    # Combinations of a and b
    comb = product(range(na), range(nb))

    # [[distance, index number(a), index number(b)], ... ]
    l = [[np.linalg.norm(a[ia] - b[ib]), ia, ib] for ia, ib in comb]

    # Sort with distance
    l.sort(key=lambda x: x[0])

    pairs = []
    for _ in range(min(na, nb)):
        m, ia, ib = l[0]
        pairs.append([ia, ib])
        # Remove items with same index number
        l = list(filter(lambda x: x[1] != ia and x[2] != ib, l))

    return pairs

def get_major_state(estimated):
    state = estimated.get('supervised_state')
    if state is None:
        state = estimated.get('estimated_state')
    
    return state

