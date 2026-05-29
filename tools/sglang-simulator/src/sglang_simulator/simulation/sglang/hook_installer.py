import sys


_HOOKS_INSTALLED = False
_CLASS_HOOKS = None


def _patch_loaded_class(hook) -> None:
    module_name = hook.HOOK_MODULE_NAME
    class_name = hook.HOOK_CLASS_NAME
    if not module_name or not class_name:
        return
    module = sys.modules.get(module_name)
    if module is None:
        return
    target = getattr(module, class_name, None)
    if target is not None:
        hook.hook(target)


def install_sglang_hooks() -> None:
    global _HOOKS_INSTALLED, _CLASS_HOOKS

    import sglang_simulator.hook as sglang_simulator_hook
    import torch
    from sglang_simulator.simulation.sglang import (
        cache_controller,
        hicache_storage,
        hiradix_cache,
        mem_cache_allocator,
        mem_pool_host,
        model_runner,
        scheduler,
        sgl_kernel_hook,
    )

    if _CLASS_HOOKS is None:
        _CLASS_HOOKS = [
            scheduler.C_SchedulerHook,
            model_runner.C_ModelRunnerHook,
            hicache_storage.C_StorageBackendFactory,
            cache_controller.C_HiCacheController,
            hiradix_cache.C_HiRadixCacheHook,
            mem_cache_allocator.C_PagedTokenToKVPoolAllocatorHook,
            mem_pool_host.C_MHATokenToKVPoolHostHook,
            mem_pool_host.C_HostKVCacheHook,
        ]

    if not _HOOKS_INSTALLED:
        _HOOKS_INSTALLED = True
        if not torch.cuda.is_available():
            sglang_simulator_hook.install_module_hooks(
                [sgl_kernel_hook.M_SGLangKernelLoadUtilHook]
            )
        sglang_simulator_hook.install_class_hooks(_CLASS_HOOKS)

    for hook in _CLASS_HOOKS:
        _patch_loaded_class(hook)
