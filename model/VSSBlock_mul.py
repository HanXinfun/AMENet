
import warnings
warnings.filterwarnings("ignore")
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import DropPath
from einops import repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

####################################################
# 1. 基础组件：通道注意力、CrossAttention、SS2D 等
####################################################

class ChannelAttention(nn.Module):
    """
    通道注意力模块（参考 RCAN）。
    """
    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.attention(x)
        return x * y


class CAB(nn.Module):
    """
    局部增强模块：卷积 + 通道注意力。
    """
    def __init__(self, num_feat, is_light_sr=False, compress_ratio=3, squeeze_factor=30):
        super(CAB, self).__init__()
        if is_light_sr:  # 轻量化配置：深度可分离卷积
            self.cab = nn.Sequential(
                nn.Conv2d(num_feat, num_feat, kernel_size=3, stride=1, padding=1, groups=num_feat),
                ChannelAttention(num_feat, squeeze_factor)
            )
        else:
            self.cab = nn.Sequential(
                nn.Conv2d(num_feat, num_feat // compress_ratio, kernel_size=3, stride=1, padding=1),
                nn.GELU(),
                nn.Conv2d(num_feat // compress_ratio, num_feat, kernel_size=3, stride=1, padding=1),
                ChannelAttention(num_feat, squeeze_factor)
            )

    def forward(self, x):
        return self.cab(x)


class CrossAttention(nn.Module):
    """
    跨模态注意力模块：query 来自一条模态，key/value 来自另一条模态。
    """
    def __init__(self, dim, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, context):
        """
        query:  (B, N, C)
        context: (B, M, C)
        """
        B, N, C = query.shape
        B2, M, C2 = context.shape
        assert B == B2 and C == C2, "Query/Context 形状不匹配"

        q = self.q(query).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # (B, heads, N, C//heads)
        kv = self.kv(context).reshape(B, M, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # (2, B, heads, M, C//heads)
        k, v = kv[0], kv[1]  # (B, heads, M, C//heads)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, M)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)  # (B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


#########################################
# SS2D 模块：选择性结构化状态空间模型（部分实现）
# 此处采用原始代码中的 SS2D 实现，请确保 selective_scan_fn 可用
#########################################
NEG_INF = -1000000

class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, out_features, in_features)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, in_features, dt_rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, in_features)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, d_state, d_inner)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, d_inner)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


####################################################
# 2. FFN：标准 Transformer 前馈网络
####################################################
class FFN(nn.Module):
    """
    标准前馈网络：Linear -> GELU -> Drop -> Linear -> Drop
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or (in_features * 4)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


####################################################
# 3. CMAMBlock：按示意图改造的多模态 Block
#    包含 SS2D -> (残差) -> CrossAttn -> (残差) -> FFN -> (残差)
####################################################
class CMAMBlock(nn.Module):
    def __init__(
        self,
        hidden_dim,
        drop_path=0.0,
        attn_drop_rate=0.0,
        d_state=16,
        expand=2.0,
        is_light_sr=False,
        num_heads=8,
        ffn_drop=0.0,
    ):
        """
        参数：
          hidden_dim: 通道数 C
          drop_path: DropPath 概率
          attn_drop_rate: 注意力层的丢弃率
          d_state, expand: SS2D 模块的关键参数
          is_light_sr: 是否使用轻量级 CAB（可忽略，本示例主要使用 FFN）
          num_heads: 跨模态注意力头数
          ffn_drop: FFN 内部的 dropout
        """
        super().__init__()
        self.drop_path = DropPath(drop_path)
        # ---------- (1) SS2D 分支 ----------
        self.ln_spec_ss2d = nn.LayerNorm(hidden_dim)
        self.ln_elev_ss2d = nn.LayerNorm(hidden_dim)
        self.ss2d_spec = SS2D(d_model=hidden_dim, d_state=d_state, expand=expand, dropout=attn_drop_rate)
        self.ss2d_elev = SS2D(d_model=hidden_dim, d_state=d_state, expand=expand, dropout=attn_drop_rate)

        # ---------- (2) Cross-Attention ----------
        self.ln_spec_ca = nn.LayerNorm(hidden_dim)
        self.ln_elev_ca = nn.LayerNorm(hidden_dim)
        self.cross_attn_spec = CrossAttention(
            dim=hidden_dim,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=attn_drop_rate
        )
        self.cross_attn_elev = CrossAttention(
            dim=hidden_dim,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=attn_drop_rate
        )

        # ---------- (3) FFN ----------
        self.ln_ffn = nn.LayerNorm(hidden_dim)
        self.ffn = FFN(hidden_dim, drop=ffn_drop)

        # 可学习的缩放或残差系数（可选）
        self.skip_scale_ss2d_spec = nn.Parameter(torch.ones(hidden_dim))
        self.skip_scale_ss2d_elev = nn.Parameter(torch.ones(hidden_dim))
        self.skip_scale_ca_spec = nn.Parameter(torch.ones(hidden_dim))
        self.skip_scale_ca_elev = nn.Parameter(torch.ones(hidden_dim))
        self.skip_scale_ffn = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, spec_input, elev_input, x_size):
        """
        spec_input/elev_input: (B, H*W, C)
        x_size: (H, W)
        返回：融合后的 (B, H*W, C)，可在后续再做多模态解码等操作
        """
        B, L, C = spec_input.shape
        H, W = x_size

        # =============== 1) SS2D 分支（每个模态各自处理） ===============
        # LN -> SS2D -> residual
        # (a) 光谱
        spec_in = spec_input  # (B, HW, C)
        spec_ln = self.ln_spec_ss2d(spec_in)         # LN 维度: (B, HW, C)
        spec_ln_4d = spec_ln.view(B, H, W, C)        # (B, H, W, C) 供 SS2D 使用
        spec_out_ss2d = self.ss2d_spec(spec_ln_4d)   # (B, H, W, C)
        spec_out_ss2d = spec_out_ss2d.view(B, L, C)
        spec_out = spec_in + self.drop_path(spec_out_ss2d) * self.skip_scale_ss2d_spec

        # (b) 高程
        elev_in = elev_input
        elev_ln = self.ln_elev_ss2d(elev_in)
        elev_ln_4d = elev_ln.view(B, H, W, C)
        elev_out_ss2d = self.ss2d_elev(elev_ln_4d)
        elev_out_ss2d = elev_out_ss2d.view(B, L, C)
        elev_out = elev_in + self.drop_path(elev_out_ss2d) * self.skip_scale_ss2d_elev

        # =============== 2) 跨模态注意力 (CA) ===============
        # LN -> CrossAttn -> residual
        spec_ln_ca = self.ln_spec_ca(spec_out)   # (B, HW, C)
        elev_ln_ca = self.ln_elev_ca(elev_out)   # (B, HW, C)

        # 以 spec_ln_ca 为 query, elev_ln_ca 为 key/value
        spec_cross = self.cross_attn_spec(spec_ln_ca, elev_ln_ca)
        # 以 elev_ln_ca 为 query, spec_ln_ca 为 key/value
        elev_cross = self.cross_attn_elev(elev_ln_ca, spec_ln_ca)

        spec_out = spec_out + self.drop_path(spec_cross) * self.skip_scale_ca_spec
        elev_out = elev_out + self.drop_path(elev_cross) * self.skip_scale_ca_elev

        # =============== 3) FFN (对融合后的单一特征或分别处理都可以) ===============
        # 本示例中，先简单融合再走 FFN；也可单独给 spec/elev 分别做 FFN
        fused = 0.5 * (spec_out + elev_out)  # (B, HW, C)
        fused_ln = self.ln_ffn(fused)
        fused_ffn = self.ffn(fused_ln)       # (B, HW, C)
        fused_out = fused + self.drop_path(fused_ffn) * self.skip_scale_ffn

        return fused_out  # (B, HW, C)


####################################################
# 4. 测试：构造随机输入并运行
####################################################
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B, H, W, C = 4, 16, 16, 768
    spec_input = torch.rand(B, H * W, C).to(device)
    elev_input = torch.rand(B, H * W, C).to(device)

    cmam_block = CMAMBlock(
        hidden_dim=C,
        drop_path=0.1,
        attn_drop_rate=0.1,
        d_state=16,
        expand=2.0,
        is_light_sr=False,
        num_heads=8,
        ffn_drop=0.1,
    ).to(device)

    output = cmam_block(spec_input, elev_input, (H, W))
    print("Input spec shape:", spec_input.size())
    print("Input elev shape:", elev_input.size())
    print("Output shape:", output.size())
