"""
自研模型放此处，并在模块末尾调用 register(...)。

示例（新建 my_mi_net.py 后在本文件 import）::

    from braindecode ... 或自写 nn.Module
    from ..registry import register

    def build_my_mi_net(n_chans, n_times, n_outputs, **hp):
        return MyMiNet(n_chans=n_chans, n_times=n_times, n_outputs=n_outputs, **hp)

    register(
        name="my_mi_net",
        family="custom",
        builder=build_my_mi_net,
        default_hparams={"drop_prob": 0.5},
        search_space={"lr": [5e-4, 1e-3], "drop_prob": [0.4, 0.5]},
        notes="自研；原生头；(B,8,500)->(B,n_outputs)",
    )
"""

# 在此 import 自研模块以触发 register，例如：
# from . import my_mi_net  # noqa: F401
