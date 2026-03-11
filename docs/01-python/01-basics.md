# Python 面试题汇总

## 1. 基础概念

### Q1: Python 中 `__init__` 和 `__new__` 的区别?

- `__new__`: 创建对象时调用，返回一个实例
- `__init__`: 初始化实例时调用，设置实例属性

```python
class Singleton:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        # 只初始化一次
        pass
```

### Q2: 什么是 GIL? Python 如何实现多线程?

**GIL (Global Interpreter Lock)**: CPython 的全局解释器锁,保证同一时刻只有一个线程执行 Python 字节码。

**多线程适用场景**: I/O 密集型任务 (网络请求、文件读写)
**多进程适用场景**: CPU 密集型任务 (计算)

```python
import threading
import multiprocessing

# IO 密集型 - 多线程
def io_task():
    requests.get(url)

# CPU 密集型 - 多进程
def cpu_task():
    result = sum(range(10**7))
```

### Q3: Python 中的装饰器是什么? 如何实现?

装饰器是一个接收函数作为参数,返回新函数的函数。

```python
import functools
import time

def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.time()-start:.2f}s")
        return result
    return wrapper

@timer
def slow_function():
    time.sleep(1)
    return "done"
```

带参数的装饰器:
```python
def retry(times=3):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for _ in range(times):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if _ == times - 1:
                        raise
            return None
        return wrapper
    return decorator

@retry(times=5)
def unstable_api():
    pass
```

### Q4: 什么是生成器? yield 关键字的作用?

生成器是一种特殊的迭代器,使用 `yield` 关键字来暂停函数并返回值。

```python
def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

# 使用
for num in fibonacci(10):
    print(num)

# 列表推导式 vs 生成器
squares = [x**2 for x in range(1000)]          # 列表 - 占用内存
squares_gen = (x**2 for x in range(1000))       # 生成器 - 惰性计算
```

### Q5: Python 中的深拷贝和浅拷贝?

```python
import copy

# 浅拷贝 - 只拷贝第一层
list1 = [[1, 2], [3, 4]]
list2 = copy.copy(list1)
list2[0].append(5)  # 影响原列表!

# 深拷贝 - 递归拷贝所有层级
list3 = copy.deepcopy(list1)
list3[0].append(6)  # 不影响原列表!
```

## 2. 进阶话题

### Q6: 什么是元类 (Metaclass)?

元类是创建类的类。

```python
class Meta(type):
    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)
        # 添加类属性
        cls.created_by = 'Meta'
        return cls

class MyClass(metaclass=Meta):
    pass

print(MyClass.created_by)  # Meta
```

应用场景: ORM、API 框架、插件系统

### Q7: Python 中的异步编程 (asyncio)?

```python
import asyncio

async def fetch_data(url):
    print(f"Fetching {url}")
    await asyncio.sleep(1)  # 模拟 IO 操作
    return f"Data from {url}"

async def main():
    # 并发执行
    results = await asyncio.gather(
        fetch_data("url1"),
        fetch_data("url2"),
        fetch_data("url3")
    )
    print(results)

asyncio.run(main())
```

### Q8: 什么是闭包? 闭包的作用?

闭包是指内部函数引用外部函数的变量。

```python
def make_counter():
    count = 0
    
    def counter():
        nonlocal count
        count += 1
        return count
    return counter

c = make_counter()
print(c())  # 1
print(c())  # 2
```

### Q9: Python 中的垃圾回收机制?

- **引用计数**: 主要方式,引用为 0 时回收
- **标记清除**: 解决循环引用
- **分代回收**: 新生代/老年代,提高效率

```python
import gc

# 手动触发垃圾回收
gc.collect()

# 查看对象引用计数
import sys
sys.getrefcount(obj)
```

### Q10: 什么是 Python 的上下文管理器?

```python
# 方法1: 类实现
class FileManager:
    def __init__(self, filename):
        self.filename = filename
    
    def __enter__(self):
        self.file = open(self.filename, 'r')
        return self.file
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()

# 方法2: 函数实现
from contextlib import contextmanager

@contextmanager
def file_manager(filename):
    f = open(filename, 'r')
    try:
        yield f
    finally:
        f.close()

# 使用
with FileManager('test.txt') as f:
    content = f.read()
```

## 3. 面试手写代码

### Q11: 实现一个 LRU Cache

```python
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity
    
    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        # 移到末尾表示最近使用
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            # 删除最旧的
            self.cache.popitem(last=False)
```

### Q12: 实现一个线程安全的单例模式

```python
import threading

class Singleton:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
```

### Q13: 实现一个并发任务池

```python
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed

def parallel_tasks(tasks, max_workers=10):
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(task): i for i, task in enumerate(tasks)}
        
        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                results.append((idx, result))
            except Exception as e:
                results.append((idx, f"Error: {e}"))
    
    return [r for _, r in sorted(results)]
```

---

> 更多内容见同目录下的分类文档
