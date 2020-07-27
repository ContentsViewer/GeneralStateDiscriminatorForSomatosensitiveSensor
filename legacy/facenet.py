import tensorflow as tf

# 先に訓練データ(過去のデータ)とそれに対応する教師(正負)のセットを作る
# そうすると, できるはず
def triplet_loss(alpha=0.2):
    




def triplet_loss(anchor, positive, negative, alpha):
    """
    Calculate the triplet loss according to the FaceNet paper

    Parameters
    ----------
    anchor : 
        the embeddings for the anchor images.
    positive : 
        the embeddings for the positive images.
    negative : 
        the embeddings for the negative images.

    Returns
    -------
    loss :
        the triplet loss according to the FaceNet paper as a float tensor.
    """

    """
    tensorflow では, 変数を作成するために
        x = tf.Variable(3, name='x')

    とする.

    変数の取得は, 
        x = tf.get_variable('x', shape=[1], initializer=init)
    とする.

    tf.get_variableは, 既に存在すれば取得し, なければ変数を作成する関数です.

    # 変数の使いまわし
        TensorFlowの名前空間は2種類あります. tf.variable_scopeとtf.name_scope

        例えば, 四つのスコープで25の変数を作りたい時
        ```
            for i in range(4):
                with tf.variable_scope('scope-{}'.format(i)):
                    for j in range(25):
                        v = tf.Variable(1, name=str(j))
        ```

        一般的に, tf.get_variableで変数を作成する際には, 同様の名前を付けることはできません.
        一方で, tf.Variableであれば, 同じ名前を付けることができます. かぶると自動で名前に数字を足す.

    """

    with tf.variable_scope('triplet_loss'):
        """
        # anchor と positive を引き算
        # 要素ごとに二乗をとる
        # 二次元目(行ごとに)で足し合わせ
            ex):
                [[ 1.  2.  3.]
                [ 4.  5.  6.]
                [ 7.  8.  9.]]

                -> [ 6. 15. 24.]
        """
        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)

        """
        # pos_dist と pos_dist を引き算
        # 引き算したものと alpha を足す -> basic_loss
        # basic_loss の要素でマイナスの要素をゼロでクリップ
        # 最後に1次元目(この行列は[x, x, x, ...] の形なので各要素)の平均をとる
        """
        basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
        loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)
    return loss

def train(sess, epoch_size, learning_rate):
    batch_number = 0

    while batch_number < epoch_size:


def select_triplets(embeddings):
