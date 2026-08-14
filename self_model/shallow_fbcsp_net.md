# shallow_fbcsp_net.py

> Standalone ShallowFBCSPNet（不依赖 braindecode / einops）。  
> 写法：**先各自定义模块 → 最后再拼接**，方便以后单独替换某一段。  
> 使用时：将下方代码另存为 `shallow_fbcsp_net.py`。

```python
"""Standalone ShallowFBCSPNet：模块先写好，最后再组装。"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import init


# ---------------------------------------------------------------------------
# 1) 各模块（可单独改 / 替换）
# ---------------------------------------------------------------------------

class Ensure4d(nn.Module):
    """(B,C,T) -> (B,C,T,1)"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        while x.ndim < 4:
            x = x.unsqueeze(-1)
        return x


class DimShuffle(nn.Module):
    """(B,C,T,1) -> (B,1,T,C)，把通道放到最后一维给空间卷积用。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 3, 2, 1).contiguous()


class ConvTime(nn.Module):
    """时间卷积：在时间维上滑窗滤波。输入 (B,1,T,C) → (B,F_t,T',C)。"""

    def __init__(self, n_filters_time: int = 40, filter_time_length: int = 25) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            1, n_filters_time, (filter_time_length, 1), stride=1, bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ConvSpat(nn.Module):
    """空间卷积：跨电极混合。输入 (B,F_t,T',C) → (B,F_s,T',1)。"""

    def __init__(
        self,
        n_chans: int,
        n_filters_time: int = 40,
        n_filters_spat: int = 40,
        *,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            n_filters_time, n_filters_spat, (1, n_chans), stride=1, bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Square(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * x


class SafeLog(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.clamp(x, min=self.eps))


class SqueezeFinalOutput(nn.Module):
    """(B,C,T,1) -> (B,C) 或 (B,C,T)。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[..., 0]
        if x.shape[-1] == 1:
            x = x.squeeze(-1)
        return x


class FinalClassifier(nn.Module):
    """分类头：按 pool 后时间长度做 Conv2d，再 squeeze。"""

    def __init__(self, n_filters: int, n_outputs: int, final_conv_length: int) -> None:
        super().__init__()
        self.conv_classifier = nn.Conv2d(
            n_filters, n_outputs, (final_conv_length, 1), bias=True
        )
        self.squeeze = SqueezeFinalOutput()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.squeeze(self.conv_classifier(x))


# ---------------------------------------------------------------------------
# 2) 组装：只负责接线，不在这里堆细节
# ---------------------------------------------------------------------------

class ShallowFBCSPNet(nn.Module):
    """Schirrmeister 2017；默认超参对齐 braindecode。"""

    def __init__(
        self,
        n_chans: int,
        n_outputs: int,
        n_times: int,
        n_filters_time: int = 40,
        filter_time_length: int = 25,
        n_filters_spat: int = 40,
        pool_time_length: int = 75,
        pool_time_stride: int = 15,
        final_conv_length: int | str = "auto",
        pool_mode: str = "mean",
        batch_norm: bool = True,
        batch_norm_alpha: float = 0.1,
        drop_prob: float = 0.5,
    ) -> None:
        super().__init__()
        self.n_chans = int(n_chans)
        self.n_outputs = int(n_outputs)
        self.n_times = int(n_times)

        pool_cls = {"max": nn.MaxPool2d, "mean": nn.AvgPool2d}[pool_mode]

        # --- 先实例化各块 ---
        self.ensuredims = Ensure4d()
        self.dimshuffle = DimShuffle()
        self.conv_time = ConvTime(n_filters_time, filter_time_length)
        self.conv_spat = ConvSpat(
            n_chans=self.n_chans,
            n_filters_time=n_filters_time,
            n_filters_spat=n_filters_spat,
            bias=not batch_norm,
        )
        self.bnorm = (
            nn.BatchNorm2d(n_filters_spat, momentum=batch_norm_alpha, affine=True)
            if batch_norm
            else nn.Identity()
        )
        self.conv_nonlin_exp = Square()
        self.pool = pool_cls(
            kernel_size=(pool_time_length, 1),
            stride=(pool_time_stride, 1),
        )
        self.pool_nonlin_exp = SafeLog()
        self.drop = nn.Dropout(p=drop_prob)

        # --- 用 backbone 推 final_conv_length（分类头未接上时）---
        if final_conv_length == "auto":
            self.eval()
            with torch.inference_mode():
                feat = self.forward_features(torch.zeros(1, self.n_chans, self.n_times))
                final_conv_length = int(feat.shape[2])
            self.train()
        self.final_conv_length = int(final_conv_length)

        self.final_layer = FinalClassifier(
            n_filters=n_filters_spat,
            n_outputs=self.n_outputs,
            final_conv_length=self.final_conv_length,
        )

        self._init_weights(batch_norm=batch_norm)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """到 Dropout 为止的特征（不含分类头）。"""
        x = self.ensuredims(x)
        x = self.dimshuffle(x)
        x = self.conv_time(x)
        x = self.conv_spat(x)
        x = self.bnorm(x)
        x = self.conv_nonlin_exp(x)
        x = self.pool(x)
        x = self.pool_nonlin_exp(x)
        x = self.drop(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.final_layer(self.forward_features(x))

    def _init_weights(self, *, batch_norm: bool) -> None:
        init.xavier_uniform_(self.conv_time.conv.weight, gain=1)
        init.constant_(self.conv_time.conv.bias, 0)
        init.xavier_uniform_(self.conv_spat.conv.weight, gain=1)
        if not batch_norm:
            init.constant_(self.conv_spat.conv.bias, 0)
        if batch_norm and isinstance(self.bnorm, nn.BatchNorm2d):
            init.constant_(self.bnorm.weight, 1)
            init.constant_(self.bnorm.bias, 0)
        init.xavier_uniform_(self.final_layer.conv_classifier.weight, gain=1)
        init.constant_(self.final_layer.conv_classifier.bias, 0)


def build_model(
    n_chans: int, n_times: int, n_outputs: int, drop_prob: float
) -> nn.Module:
    """与 train_lab baseline_shallow.build_model 同签名。"""
    return ShallowFBCSPNet(
        n_chans=n_chans,
        n_outputs=n_outputs,
        n_times=n_times,
        drop_prob=drop_prob,
    )


if __name__ == "__main__":
    m = build_model(8, 500, 2, 0.5)
    y = m(torch.randn(4, 8, 500))
    print("logits", tuple(y.shape))
    print("final_conv_length", m.final_conv_length)
```

## 结构示意

```text
模块（先写好）          组装（forward_features + final_layer）
─────────────────      ────────────────────────────────────
Ensure4d          ──┐
DimShuffle        ──┤
ConvTime          ──┤
ConvSpat          ──┼──► forward_features
BatchNorm/Id      ──┤
Square            ──┤
AvgPool / MaxPool ──┤
SafeLog           ──┤
Dropout           ──┘
FinalClassifier   ─────► forward = final(features)
```

以后要加模块：新建一个 `nn.Module`，再在 `forward_features` / `forward` 里插入一行即可。
