import torch
import torch.nn
import math
import copy

class Base(torch.nn.Module):

    def __init__(self, args, enbeddings):
        super(Base, self).__init__()

        ch_in = args.image_channels_input
        ch_out = args.image_channels_output
        emb_dim = args.embed_dim
        kernel_sizes = args.kernel_sizes
        dropout = args.dropout
        number_classes = args.number_classes


        self.emb_layer = torch.nn.Embedding.from_pretrained(torch.tensor(enbeddings.vectors))
        self.convolutions = torch.nn.ModuleList([torch.nn.Conv2d(in_channels=ch_in,
                                                                 out_channels=ch_out,
                                                                 kernel_size=(K, emb_dim)) for K in kernel_sizes])

        self.dropout = torch.nn.Dropout(dropout)
        self.fc1 = torch.nn.Linear(len(kernel_sizes) * ch_out, number_classes)

    def forward(self, x):

        x = self.emb_layer(x)
        x = x.unsqueeze(dim=1)
        x = [torch.nn.functional.relu(conv(x)).squeeze(dim=3) for conv in self.convolutions]
        x = [torch.nn.functional.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        x = self.fc1(x)
        # do not convert to softmax inside network as torch CrossEntropyLoss expects logits (and calls softmax itself)
        return x


def attention(query, key, value):
    assert (query.size(-1) == key.size(-1))
    # the dimension of queries and keys, must have equal dimension in this implementation
    d = query.size(-1)
    # implements (QK^T)/sqrt(d), the dot product attention mechanism
    attn_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d)
    weight = torch.nn.functional.softmax(attn_scores, dim=-1)
    return torch.matmul(weight, value), weight


class MultiHeadedAttention(torch.nn.Module):

    def __init__(self, num_heads, dim_model, dropout=0.1, num_linear_projections=4):
        super(MultiHeadedAttention, self).__init__()
        # the model dimension must be perfectly divisible by the number of attention heads h
        assert dim_model % num_heads == 0

        self.dim_kq = dim_model // num_heads
        self.num_heads = num_heads
        self.dim_model = dim_model
        self.dropout = dropout
        self.linear_layers = torch.nn.ModuleList([torch.nn.Linear(dim_model, dim_model)
                                                  for _ in range(num_linear_projections)])
        self.final_linear_layer = torch.nn.Linear(dim_model, dim_model)
        self.attn = None

    # Note: we omit the last parameter (mask) as it is not relevant for our implementation (encoder only)
    def forward(self, query, key, value):
        assert query.size() == key.size() == value.size()

        batch_size = value.size(0)

        # carries out the projections using the linear layers: headi=Attention(QWQi,KWKi,VWVi)
        query, key, value = [linear_projection(x).view(batch_size, -1, self.num_heads, self.dim_kq).transpose(1, 2)
                             for linear_projection, x in zip(self.linear_layers, (query, key, value))]

        # we now apply the attention mechanism to the projected values
        weighted_vectors, self.attn = attention(query, key, value, dropout=self.dropout)

        # contiguous() is needed as the attention head subvectors may be at different positions in memory
        weighted_vectors = weighted_vectors.transpose(1, 2).contiguous().view(batch_size, -1,
                                                                              self.num_heads * self.dim_kq)
        # apply a final linear projection to the weighted vectors
        return self.final_linear_layer(weighted_vectors)


class Encoder(torch.nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        # copy the passed encoder layer architecture N times
        self.layers = torch.nn.ModuleList(copy.deepcopy(layer) for _ in range(N))
        self.norm = LayerNorm(layer.size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class LayerNorm(torch.nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = torch.nn.Parameter(torch.ones(features))
        self.b_2 = torch.nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class EncoderLayer(torch.nn.Module):

    def __init__(self, size, attention_mechanism, feed_forward_mechanism, dropout):
        super(EncoderLayer, self).__init__()
        self.attention_mechanism = attention_mechanism
        self.feed_forward_mechanism = feed_forward_mechanism
        self.dropout = dropout
        # we use two sublayers in the encoder
        self.sublayer = torch.nn.ModuleList([SublayerConnection(size, dropout) for i in range(2)])
        self.size = size

    def forward(self, x):
        # each sublayer takes the mechanism it applies as an input!
        x = self.sublayer[0](x, lambda x: self.attention_mechanism(query=x, key=x, value=x))
        return self.sublayer[1](x, self.feed_forward_mechanism)


class PositionwiseFeedForward(torch.nn.Module):
    def __init__(self, d_model, d_feedforward, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()

        self.ff1 = torch.nn.Linear(d_model, d_feedforward)
        self.ff2 = torch.nn.Linear(d_feedforward, d_model)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        # apply all the above layers in order
        return self.ff2(self.dropout(torch.nn.functional.relu(self.ff1(x))))


class SublayerConnection(torch.nn.Module):

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderClassifier(torch.nn.Module):

    def __init__(self, args, enbeddings):
        super(EncoderClassifier, self).__init__()



        self.emb_layer = torch.nn.Embedding.from_pretrained(torch.tensor(enbeddings.vectors))
        self.attention_mechanism = MultiHeadedAttention(dim_model=args.emb_dim, num_heads=6)
        self.feedforward_mechanism = PositionwiseFeedForward(d_model=args.emb_dim, d_feedforward=2 * args.emb_dim)
        self.encoder = Encoder(
            layer=EncoderLayer(args.emb_dim, self.attention_mechanism, self.feedforward_mechanism, dropout=0.1), N=args.encoder_layers)
        self.convolutions = torch.nn.ModuleList([torch.nn.Conv2d(in_channels=args.ch_in,
                                                                 out_channels=args.ch_out,
                                                                 kernel_size=(K, args.emb_dim)) for K in args.kernel_sizes])

        self.dropout = torch.nn.Dropout(args.dropout)
        self.fc1 = torch.nn.Linear(len(args.kernel_sizes) * args.ch_out, args.number_classes)

    def forward(self, x):
        x = self.emb_layer(x)
        x = self.encoder(x)
        x = x.unsqueeze(dim=1)
        x = [torch.nn.functional.relu(conv(x)).squeeze(dim=3) for conv in self.convolutions]
        x = [torch.nn.functional.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        x = self.fc1(x)
        return x