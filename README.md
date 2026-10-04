# distributed-kv-store

单机 LSM 键值存储（Python 标准库实现，无第三方依赖）：预写日志、有序表、布隆过滤、快照与压实。

## 安装与入口

```
python3 -m pip install -e .
distributed-kv-store --help
distributed-kv-store describe
```

包名 `kvstore`，命令入口 `distributed-kv-store`。命令**只往 stdout 写 JSON**，错误只写 stderr（且 stderr 里**只有**那一个 JSON 文档，不含任何警告行——调用方按 JSON 解析 stderr）。

## 目录布局

```
<dir>/wal.log                预写日志（唯一的可变文件）
<dir>/table-000001.sst       有序表，编号递增；编号越大越新
```

## 写路径与崩溃恢复

一次写 = **先 WAL 后内存表**：只有日志落盘（`fsync`）之后写入才算确认，因此崩溃不会丢掉已确认的写。

日志每行一个记录：`<8 位十六进制 crc32><空格><规范 JSON>`，payload 形如
`{"op":"put","key":"a","seq":1,"value":"1"}`（`del` 无 `value`）。

恢复时**正向读取、逐条校验**，遇到第一条缺失/截断/校验和不符的记录就停下——这正是崩溃留下的形状——并把文件**截断到最后一个完好记录的末尾**。恢复结果通过 `verify` 报告：

```json
{"tables":[…],"wal":{"records":1,"truncatedBytes":42,"lastSequence":1,"putKeys":1,"deletedKeys":0}}
```

`truncatedBytes > 0` 表示**发生过修复**，此时退出码为 **3**（报告已产出但判词为负）。

## 读路径与可见性

`内存表 → 最新的表 → … → 最旧的表`。**墓碑（删除标记）命中就是答案**（该键已删除）；只有全部未命中才表示这个键从未存在。

打开封存表只常驻**布隆过滤器、条目数量与稀疏块索引**（每块首键 + 字节偏移），不保留任何键和值：布隆判否时完全不读条目区；可能命中时只读取定位该键的一个小块并解码命中值。范围扫描与压实把各层有序数据用小顶堆**增量合并**（不建整库副本），墓碑隐藏旧值、内存表最后覆盖，`limit` 凑满可见行即停止读取。

- `flush` 把内存表封成新表并重置日志；
- `compact` 把所有表与内存表并成一个，丢弃不再遮挡任何值的墓碑；
- **合并方向是契约的一部分**：从最旧到最新逐层覆盖，内存表最后 ⇒ 保留**最新**值。

## 子命令

| 命令 | 作用 | 退出码 |
|---|---|---|
| `describe` | 打印能力、布局、日志记录格式与退出码 | 0 |
| `put --dir D --key K --value V` | 写入 | 0 / 2 |
| `delete --dir D --key K` | 写入墓碑 | 0 / 2 |
| `get --dir D --key K` | 读取 | 0（命中）/ **3**（未命中）/ 2 |
| `scan --dir D [--start S] [--end E] [--limit N]` | 有序范围扫描（`start` 含、`end` 不含） | 0 / **3**（空）/ 2 |
| `stats --dir D` | 计数器（表数、条目数、内存表字节、日志记录数、截断字节） | 0 / 2 |
| `flush --dir D` | 封表 | 0 / **3**（无可封内容）/ 2 |
| `compact --dir D` | 合并 | 0 / 2 |
| `verify --dir D` | 重新打开所有产物并报告修复情况 | 0 / **3**（发生过修复）/ 2 |

所有子命令都接受 `--memtable-limit <bytes>`（默认 1 MiB，写满自动封表）。

## 保障

- 键不得为空、不得含制表符或换行（否则报 `validation_error`）。
- 每张表都带 **crc32 封条**：封条不符即报 `corruption_error` 并停止——**已封存的表损坏不可静默恢复**（这对"不能损坏旧数据"是硬要求）。
- 每张表带**布隆过滤**：不存在的键一次哈希即可判否，不必扫描；可能命中也只读一个定界块。
- 日志的撕裂尾部**是可以修复的**（预期内），表的损坏**不是**（必须报错）。
- 错误文档形如 `{"error":"<kind>","message":"…", …}`，`kind` 稳定可取。

## 目录

```
kvstore/errors.py     异常层次（kind + 上下文）
kvstore/wal.py        日志编码、崩溃恢复、撕裂尾部截断
kvstore/memtable.py   内存表与墓碑、字节计数
kvstore/sstable.py    有序表（header/entry/footer）+ 布隆过滤 + 封条校验
kvstore/store.py      读路径、写路径、flush、compact、snapshot、stats、verify
kvstore/cli.py        九个子命令与退出码
tests/                日志恢复、可见性顺序、压实、快照、CLI 契约
```
