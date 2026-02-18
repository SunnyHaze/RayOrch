from __future__ import annotations
from typing import Any, Callable, Dict, Generic, Optional, Protocol, Tuple, Type, TypeVar, cast
from typing_extensions import ParamSpec
import time

INITP = ParamSpec("InitP")   # op_cls.__init__ 的参数
RUNP = ParamSpec("RunP")     # op_cls.run 的参数
R = TypeVar("R")
class RunOp(Protocol[INITP, RUNP, R]):
    def __init__(self, *args: INITP.args, **kwargs: INITP.kwargs) -> None: ...
    def run(self, *args: RUNP.args, **kwargs: RUNP.kwargs) -> R: ...

import ray

from .dispatch_mode import DispatchMode, get_predefined_dispatch_fn

# from image_class import ImageLoadOp, ImageSaveOP
# from yolo_class import YOLODrawOp
# from sam_class import SAMOp, OutputSAMMasksOp


# --------------------------
# Actor：运行时不要 Generic/ParamSpec, 否则会报serialize的错误
# --------------------------
@ray.remote
class RunnerActor:
    def __init__(
        self,
        op_cls: Type[Any],                 # 运行时用 Any
        init_args: Tuple[Any, ...],
        init_kwargs: Dict[str, Any],
    ):
        self.op = op_cls(*init_args, **init_kwargs)

    def run(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        return self.op.run(*args, **kwargs)

class RayModule(Generic[INITP, RUNP, R]):
    def __init__(
        self,
        op_cls: Type[RunOp[INITP, RUNP, R]],
        *,
        env: Optional[str] = None,   # ✅ 允许 None
        replicas: int = 1,
        num_gpus_per_replica: float = 0.0,
        dispatch_mode: DispatchMode | None = None,
        dispatch_fn : Optional[Callable[..., Any]] = None,
        collect_fn : Optional[Callable[..., Any]] = None,
    ):
        self._op_cls = op_cls
        self._env = env
        self._replicas = replicas
        self._num_gpus_per_replica = num_gpus_per_replica
        self.actors = []

        # 让 dispatch_fn/collect_fn 也能直接传入覆盖 registry
        if dispatch_fn is not None and collect_fn is not None:
            self._dispatch_fn = dispatch_fn
            self._collect_fn = collect_fn
        elif dispatch_mode is not None:
            spec = get_predefined_dispatch_fn(dispatch_mode)
            self._dispatch_fn = spec.dispatch_fn
            self._collect_fn = spec.collect_fn
        else:
            self._dispatch_fn = None
            self._collect_fn = None

    def pre_init(self, *args: INITP.args, **kwargs: INITP.kwargs) -> "RayModule[INITP, RUNP, R]":
        self.actors = [
            RunnerActor.options(
                runtime_env={"conda": self._env} if self._env is not None else None,
                num_gpus=self._num_gpus_per_replica,
            ).remote(self._op_cls, args, kwargs)
            for _ in range(self._replicas)
        ]
        return self

    def _fanout(self, *args: RUNP.args, **kwargs: RUNP.kwargs):
        # Single replica, without dispatch/collect
        if self._dispatch_fn is None or self._collect_fn is None or self._replicas == 1:
            ref = self.actors[0].run.remote(args, kwargs)
            out = ray.get(ref) # blocking
            return out
        # Multi-replica with dispatch/collect
        per_args, per_kwargs = self._dispatch_fn(self, *args, **kwargs)

        # Check for correctness
        for j in range(len(per_args)):
            if len(per_args[j]) != self._replicas:
                raise ValueError(f"Dispatched args[{j}] len ({len(per_args[j])}) != replicas ({self._replicas})")
        for k, v in per_kwargs.items():
            if len(v) != self._replicas:
                raise ValueError(f"Dispatched kwargs['{k}'] len ({len(v)}) != replicas ({self._replicas})")


        refs = []
        for i in range(self._replicas):
            args_i = tuple(per_args[j][i] for j in range(len(per_args)))
            kwargs_i = {k: v[i] for k, v in per_kwargs.items()}
            refs.append(self.actors[i].run.remote(args_i, kwargs_i))

        output = ray.get(refs)  # blocking
        collected = self._collect_fn(self, output)
        return cast(R, collected)

    def __call__(self, *args: RUNP.args, **kwargs: RUNP.kwargs) -> R:
        return cast(R, self._fanout(*args, **kwargs))

    def remote(self, *args: RUNP.args, **kwargs: RUNP.kwargs):
        # 简单版本：返回每个 replica 的 refs（或单个 ref）
        if self._dispatch_fn is None or self._collect_fn is None or self._replicas == 1:
            return self.actors[0].run.remote(args, kwargs)

        per_args, per_kwargs = self._dispatch_fn(self, *args, **kwargs)
        refs = []
        for i in range(self._replicas):
            args_i = tuple(per_args[j][i] for j in range(len(per_args)))
            kwargs_i = {k: v[i] for k, v in per_kwargs.items()}
            refs.append(self.actors[i].run.remote(args_i, kwargs_i))
        return refs

