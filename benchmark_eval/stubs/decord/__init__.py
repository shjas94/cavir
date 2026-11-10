"""Stand-in for decord, which has no ARM64 wheel: VLMEvalKit's video datasets import names from it when vlmeval is
imported. benchmark_eval uses image benchmarks only; every name imports, and calling any of them raises."""


def __getattr__(name):
    def unavailable(*args, **kwargs):
        raise RuntimeError(f"decord.{name}: decord is not installed on this ARM64 machine (video benchmarks are not used)")
    return unavailable
