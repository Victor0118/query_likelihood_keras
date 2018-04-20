from keras.engine import Layer
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, Convolution1D, GlobalMaxPooling1D, GlobalAveragePooling1D, \
    Lambda, TimeDistributed, SpatialDropout1D, Reshape, RepeatVector
from keras.layers.merge import Dot, Concatenate, Multiply, Add
from keras.models import Model
from keras import backend as K

ATTENTION_DEEP_LEVEL = 1

def keras_diagonal(x):
    seq_len = 10 if 'seq_len' not in K.params else K.params['seq_len']
    if seq_len != 10: print("dim of diagonal matrix in keras_diagonal: %d" % seq_len)
    return K.sum(K.eye(seq_len) * x, axis=1)

def elementwise_prod(x, y):
    (_, vec_size,) = K.int_shape(x)
    attention_level = K.params['attention_level']
    return x*y[:, attention_level, 0:vec_size]

def attention_weighting_prod(attention, weights):
    attention_level = K.params['attention_level']
    return attention * weights[:, attention_level, :, :]

def repeat_vector(x, rep, axis):
    # rep = int(rep.get_value())
    # axis = int(axis.get_value())
    return K.repeat_elements(x, rep, axis)

def repeat_vector_1(x):
    max_doc_len = K.params['max_doc_len']
    return K.repeat_elements(x, max_doc_len, -2)

def repeat_vector_2(x):
    max_query_len = K.params['max_query_len']
    return K.repeat_elements(x, max_query_len, -1)

def max_pooling(x):
    return K.max(x, axis=1)

def mean_pooling(x):
    return K.mean(x, axis=1)

def max_pooling_with_mask(x, query_mask):
    # x is batch_size * |doc| * |query|
    # query_mask is batch_size * |query| (with masks as 0)
    return K.max(x, axis=1) * query_mask

def add_mask(x, query_mask):
    # x is batch_size * |doc| * |query|
    # query_mask is batch_size * |query| (with masks as 0)
    max_query_len = K.params['max_query_len']
    reshaped_query_mask = Reshape((1, max_query_len),
                                  input_shape=(max_query_len,))(query_mask)
    masked_query = x * repeat_vector_1(reshaped_query_mask)
    return masked_query

def mean_pooling_with_mask(x, doc_mask, query_mask):
    # x is batch_size * |doc| * |query|
    # doc_mask is batch_size * |doc| (with masks as 0)
    # query_mask is batch_size * |query| (with masks as 0)
    ZERO_SHIFT = 0.1
    doc_mask_sum = (K.sum(doc_mask, axis=-1, keepdims=True) + ZERO_SHIFT)
    return query_mask * K.batch_dot(x, doc_mask, axes=[1, 1]) / doc_mask_sum

def add_onehot_embed_layer(vocab_emb, vocab_size, embed_size, train_embed, dropout_rate, layer_name=None):
    emb_layer = Sequential(name=layer_name) if layer_name is not None else Sequential()
    if vocab_emb is not None:
        print("Embedding with initialized weights")
        print(vocab_size, embed_size)
        emb_layer.add(Embedding(input_dim=vocab_size, output_dim=embed_size, weights=[vocab_emb],
                                    trainable=train_embed, mask_zero=False))
    else:
        print("Embedding with random weights")
        emb_layer.add(Embedding(input_dim=vocab_size, output_dim=embed_size, trainable=True, mask_zero=False))
    emb_layer.add(SpatialDropout1D(dropout_rate))
    return emb_layer

def add_embed_layer(vocab_emb, vocab_size, embed_size, train_embed, dropout_rate, layer_name=None):
    emb_layer = Sequential(name=layer_name) if layer_name is not None else Sequential()
    if vocab_emb is not None:
        print("Embedding with initialized weights")
        print(vocab_size, embed_size)
        emb_layer.add(Embedding(input_dim=vocab_size, output_dim=embed_size, weights=[vocab_emb],
                                    trainable=train_embed, mask_zero=False))
    else:
        print("Embedding with random weights")
        emb_layer.add(Embedding(input_dim=vocab_size, output_dim=embed_size, trainable=True, mask_zero=False))
    emb_layer.add(SpatialDropout1D(dropout_rate))
    return emb_layer

def add_dot_layer(query_embedding, doc_embedding, layer_name, query_mask=None, doc_mask=None, mask=False):
    dot_prod = Dot(axes=-1, name=layer_name)([doc_embedding, query_embedding])
    norm_sim = Activation('softmax')(dot_prod)
    if mask:
        max_sim = Lambda(lambda x: max_pooling_with_mask(x[0], x[1]), output_shape=lambda inp_shp: (
            inp_shp[0][0], inp_shp[0][2],))([norm_sim, query_mask])
        mean_sim = Lambda(lambda x: mean_pooling_with_mask(x[0], x[1], x[2]), output_shape=lambda inp_shp: (
            inp_shp[0][0], inp_shp[0][2],))([norm_sim, doc_mask, query_mask])
    else:
        max_sim = Lambda(max_pooling, output_shape=lambda inp_shp: (inp_shp[0], inp_shp[2], ))(norm_sim)
        mean_sim = Lambda(mean_pooling, output_shape=lambda inp_shp: (inp_shp[0], inp_shp[2],))(norm_sim)
    return norm_sim, max_sim, mean_sim


def add_attention_layer(query_embedding, doc_embedding, layer_name, attention_level,
                                             query_weight, query_mask=None, doc_mask=None, mask=False, norm_weight=True,
                                             cos_norm=False):
    """
    Dot -> softmax -> pooling -> (mask) -> weighting

    :param query_embedding:
    :param doc_embedding:
    :param layer_name:
    :param attention_level:
    :param query_weight:
    :param query_mask:
    :param doc_mask:
    :param mask:
    :param norm_weight:
    :return:
    """
    if cos_norm:
        norm_sim = Dot(axes=-1, name=layer_name, normalize=True)([doc_embedding, query_embedding])
        norm_sim = Activation('softmax')(norm_sim)
    else:
        dot_prod = Dot(axes=-1, name=layer_name)([doc_embedding, query_embedding])
        norm_sim = Activation('softmax')(dot_prod)

    if norm_weight:
        query_weight = NormWeight()(query_weight)

    if mask:
        max_sim = Lambda(lambda x: max_pooling_with_mask(x[0], x[1]), output_shape=lambda inp_shp: (
            inp_shp[0][0], inp_shp[0][2],))([norm_sim, query_mask])
        mean_sim = Lambda(lambda x: mean_pooling_with_mask(x[0], x[1], x[2]), output_shape=lambda inp_shp: (
            inp_shp[0][0], inp_shp[0][2],))([norm_sim, doc_mask, query_mask])
    else:
        max_sim = Lambda(max_pooling, output_shape=lambda inp_shp: (inp_shp[0], inp_shp[2], ))(norm_sim)
        mean_sim = Lambda(mean_pooling, output_shape=lambda inp_shp: (inp_shp[0], inp_shp[2],))(norm_sim)
    if attention_level <= 1:
        setattr(K, 'params', {'attention_level': attention_level})
        max_sim = Lambda(lambda x: elementwise_prod(x[0], x[1]),
                     output_shape=lambda inp_shp: (inp_shp[0][0], inp_shp[0][1],))([max_sim, query_weight])
        mean_sim = Lambda(lambda x: elementwise_prod(x[0], x[1]),
                      output_shape=lambda inp_shp: (inp_shp[0][0], inp_shp[0][1]))([mean_sim, query_weight])
    return norm_sim, max_sim, mean_sim

########################## Our model implementation #########################

def create_model(max_query_len, max_doc_len, max_url_len, vocab_size, embedding_matrix, nb_filters,
                           embed_size=300, dropout_rate=0.1, trainable=True, weighting=False, mask=False,
                           conv_option="normal", model_option="word_url", cos_norm=False, pooling="all", deeplevel=5):
    global ATTENTION_DEEP_LEVEL
    ATTENTION_DEEP_LEVEL = deeplevel
    print('create attention model...')
    setattr(K, 'params', {'max_doc_len': max_doc_len['word'], 'max_query_len': max_query_len['word']})

    query_word_input = Input(shape=(max_query_len['word'], ), name="query_word_input")
    doc_word_input = Input(shape=(max_doc_len['word'],), name="doc_word_input")
    query_char_input = Input(shape=(max_query_len['3gram'],), name="query_3gram_input")
    doc_char_input = Input(shape=(max_doc_len['3gram'],), name="doc_3gram_input")
    url_char_input = Input(shape=(max_url_len['url'],), name="url_3gram_input")
    input_list = [query_word_input, doc_word_input]

    norm_sim_list, max_sim_list, mean_sim_list = [], [], []

    # Define Mask
    if mask:
        query_word_mask = Input(shape=(max_query_len['word'], ), name="query_word_mask")
        doc_word_mask = Input(shape=(max_doc_len['word'], ), name="doc_word_mask")
        input_list.extend([query_word_mask, doc_word_mask])

    query_word_weight = Input(shape=(ATTENTION_DEEP_LEVEL, max_query_len['word'],), name="query_word_weight")
    doc_word_weight = Input(shape=(ATTENTION_DEEP_LEVEL, max_doc_len['word'],), name="doc_word_weight")
    input_list.extend([query_word_weight, query_char_weight, doc_word_weight])

    # Create query-doc word-to-word attention layer
    query_word_embedding_layer = add_embed_layer(embedding_matrix, vocab_size['word'], embed_size, trainable,
                                                 dropout_rate, layer_name="word_embedding")

    query_embedding = query_word_embedding_layer(query_word_input)
    doc_embedding = query_word_embedding_layer(doc_word_input)
    conv_embedding_list = [[query_embedding], [doc_embedding]]
    for i in range(ATTENTION_DEEP_LEVEL):
        if i > 0:
            output_list, conv_output_list = add_conv_layer([query_embedding, doc_embedding], "word-conv%d" % i,
                                                           nb_filters, 2, "same", dropout_rate, strides=1,
                                                           attention_level=i, conv_option=conv_option,
                                                           prev_conv_tensors=conv_embedding_list)
            query_embedding, doc_embedding = output_list[0], output_list[1]
            conv_embedding_list[0].append(conv_output_list[0])
            conv_embedding_list[1].append(conv_output_list[1])
        if weighting == 'query':
            norm_sim, max_sim, mean_sim = add_attention_layer_with_query_weighting(
                query_embedding, doc_embedding, "word-attention%d" % i, i, query_word_weight,
                query_word_mask, doc_word_mask, mask, norm_weight, cos_norm=cos_norm)
        elif weighting == 'doc':
            norm_sim, max_sim, mean_sim = add_attention_layer_with_doc_weighting(
                query_embedding, doc_embedding, "word-attention%d" % i, i, query_word_weight, doc_word_weight,
                max_query_len['word'], max_doc_len['word'], query_word_mask, doc_word_mask, mask)
        else:
            norm_sim, max_sim, mean_sim = add_attention_layer(query_embedding, doc_embedding, "word-attention%d" % i,
                                                              query_word_mask, doc_word_mask, mask)
        norm_sim_list.append(norm_sim)
        max_sim_list.append(max_sim)
        mean_sim_list.append(mean_sim)

    if pooling == "all":
        max_sim_list.extend(mean_sim_list)
        sim_list = max_sim_list
    elif pooling == "max":
        sim_list = max_sim_list
    elif pooling == "mean":
        sim_list = mean_sim_list
    else:
        print("unsupported pooling method")

    feature_vector = Concatenate(axis=-1, name="feature_vector")(sim_list)
    feature_vector1 = Dense(150, activation='relu', name="feature_vector1")(feature_vector)
    feature_vector2 = Dense(50, activation='relu', name="feature_vector2")(feature_vector1)
    prediction = Dense(1, activation='sigmoid', name="prediction")(feature_vector2)
    return Model(input_list, [prediction])

