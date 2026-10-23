# torch_musa 统一内存管理（Unified Memory Management）

> **语言**: [English](README.md) | **中文**

## 概览

M1000 架构采用 UMA（Unified Memory Addressing，统一内存寻址）设计，使 GPU 和 CPU 能够访问同一块共享的物理内存空间。

为了在 M1000 上运行模型时优化内存占用，该实现能够：
- 消除在 GPU 上重复分配内存
- 减少主机和设备之间的内存拷贝
- 让 GPU 可以直接访问由 CPU 分配器分配的内存

我们为 MUSA 后端提供了统一内存管理（Unified Memory Management）支持，从而在执行 `torch.load(map_location="musa")` 时避免 GPU 内存分配。

## 使用方法

目前支持两种调用方式。

### 方法一：通过环境变量进行全局配置

```bash
export PYTORCH_MUSA_ALLOC_CONF="cpu:unified"
```

### 方法二：使用上下文管理器

```python
import torch_musa
with torch_musa.use_unified_cpu_allocator():
    # 在此处编写你的代码
    ...
```

## 效果

在 MUSA 后端启用统一内存管理后，`torch.load(map_location="musa")` 不再需要进行 GPU 内存分配，也无需在主机与设备之间进行内存拷贝。

---

# torch_musa 双精度自动转换（double cast float）

## 概览

`torch_musa` 使用一种 `autocast 机制`：会自动将类型为 `double` 的输入张量转换为 `float` 进行计算，并在输出时再将结果转换回 `double`。

## 使用方法

### 通过环境变量进行全局配置

```bash
export TORCH_USE_MUSA_DOUBLE_CAST="true"|"1"|"ON"|"on"
```

