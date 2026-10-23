# 逐时信息揭示下的机组与储能调度

基于 PyPSA v1.4.0 的运行策略树扩展，输入为合成调度研究网络。完整接口、科学定义和数值范围见 instruction.md。

## 运行

使用 Linux/amd64、Python 3.12，随包提供锁定版本和哈希的运行、测试依赖。工程保留原生 PyPSA 网络、优化和组件接口。

```sh
harbor run -p . -a oracle
harbor run -p . -a nop
```

Agent 作答预算为10800秒，运行资源为1 CPU、4 GiB；单次接口调用限90秒，Verifier限600秒。模型服务凭据由外部环境提供。

独立 Verifier 需要 root 引导，并保留 Docker 默认的 SYS_CHROOT、MKNOD、SETUID、SETGID 能力。候选以非特权身份运行，采用文件隔离和 seccomp；Landlock 不可用时使用仅含运行依赖、源码和公开输入的 chroot。候选不能读取测试、参考解或评分文件，无网络，只能写私有临时目录。隔离无法建立时明确报告环境错误。

## 来源

PyPSA v1.4.0，commit `6c45594596cbc5612511ee9202a9e269e5001d88`： https://github.com/PyPSA/PyPSA 。上游来源记录与 MIT 许可证保留在 environment/src/UPSTREAM.json 和 LICENSE-PyPSA。策略树接口为本应用新增功能。
