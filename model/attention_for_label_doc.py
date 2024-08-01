import torch
import torch.nn as nn
import torch.nn.functional as F


# RMS 归一化函数
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps                                     # 加载分母中的一个非常小的常量，避免除数为 0
        self.weight = nn.Parameter(torch.ones(dim))        # weight : 不断学习的参数

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)      # torch.rsqrt: 平方根后取倒数

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
    

class AttentionForLabelDoc(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.attention_norm = RMSNorm(768, eps=1e-5)

        self.dropout = args.dropout

        self.n_heads = 8
        self.n_kv_heads = 8
        self.dim = 768

        self.n_kv_heads = self.n_heads if self.n_kv_heads is None else self.n_kv_heads
        self.head_dim  = self.dim // self.n_heads

        self.wq = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=False)
        self.attn_dropout = nn.Dropout(args.dropout)

    def forward(self, doc_emb, label_emb, weight):
        doc_emb = self.attention_norm(doc_emb)
        label_emb = self.attention_norm(label_emb)
        weight = self.attention_norm(weight)

        xq = self.wq(doc_emb)
        xk = self.wk(weight)
        xv = self.wv(label_emb)
        output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=True)

        output = self.wo(output)
        output = self.attn_dropout(output)

        return output
