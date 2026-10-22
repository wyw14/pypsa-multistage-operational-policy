# 逐时信息揭示下的机组与储能调度

电网调度研究需要在负荷、可再生出力和线路停运信息逐小时揭示时，为机组和储能生成可执行的策略。请扩展 PyPSA 的运行决策接口：热机下一小时的开停安排在本小时下达，出力和储能动作则可在当期信息到达后调整。返回遵守历史、储能连续状态、故障后电气约束的最优节点策略，并按给定条件尾部风险口径计算目标。存在多个最优策略时接受任一合规结果。

这是基于 PyPSA v1.4.0 源码的作者扩展，不是上游已知 bug。来源为 `PyPSA/PyPSA`，commit `6c45594596cbc5612511ee9202a9e269e5001d88`，源码许可证见 `/app/src/LICENSE-PyPSA`。源码包保留原来的优化、网络与组件实现。现有随机优化和普通时间序列启停不能直接替代本题的运行策略树定义。

## 修改与调用边界

工作目录 `/app/src`。实现原生接口：

```python
network.optimize.policy_tree(tree, options) -> dict
# 实际入口位于 pypsa/optimization/scenario_tree.py：
optimize_policy_tree(network, tree, options) -> dict
```

`OptimizationAccessor.policy_tree` 已连接到该入口；`policy_schema.normalize` 已提供完整的公开域检查，可以使用或修改。`candidate.solve(payload)` 只是 JSON 适配器：它从记录构造真正的 PyPSA `Network`，然后调用原生接口。验收经此适配器进行。可以修改 `/app/src` 下源码并添加辅助模块；保持入口、原生组件语义和已有 `n.optimize(...)` 可用。不要修改受信验收材料，不依赖网络服务或外部求解进程。NumPy、SciPy、Linopy 和 HiGHS 均已安装，可合理使用。函数不能改变传入的 network、tree 或 options；重复调用彼此独立。

JSON 有 `network`, `tree`, `options` 三项，默认 `operation="policy"`。可选的 `operation="deterministic"` 是已提供的兼容入口，其格式和行为见 `candidate.py`，请保留。所有返回值是普通 JSON 数、字符串、数组或对象，不包含 NaN/Infinity。

## 原生网络与支持域

`network` 中每类组件都是记录数组，组件 `name` 为本类唯一的非空字符串；列表顺序没有语义。可以省略空组件数组。下列是全部支持字段，其他字段拒绝并抛 `ValueError`。省略属性使用本版本 PyPSA 默认值。

| 数组 | 原生组件和允许字段 |
|---|---|
| `buses` | Bus: `name, v_nom` |
| `lines` | Line: `name, bus0, bus1, x, s_nom` |
| `generators` | Generator: `name, bus, p_nom, committable, p_min_pu, p_max_pu, marginal_cost, stand_by_cost, start_up_cost, shut_down_cost, min_up_time, min_down_time, ramp_limit_up, ramp_limit_down, ramp_limit_start_up, ramp_limit_shut_down` |
| `loads` | Load: `name, bus` |
| `storage_units` | StorageUnit: `name, bus, p_nom, p_min_pu, p_max_pu, max_hours, efficiency_store, efficiency_dispatch, standing_loss, state_of_charge_initial, marginal_cost` |

母线 1–8 个，线路至多 12 条，发电机至多 8 台，其中可开停热机至多 3 台；负荷至多 8 个，储能至多 2 个。没有热机、储能或线路也是合法特例。所有引用母线存在，线路两端不同且 `v_nom` 相同。`v_nom>0`（kV），`x>0`（欧姆），`s_nom>=0`（MW）。所有容量固定，`p_nom>0`；没有扩容、变压器、损耗、交流无功或线型自动换算。功率 MW、电量 MWh、相角 rad、时间 h；每个节点恰为一小时，snapshot 权重均为 1。

所有成本非负有限。热机 `0<=p_min_pu<=p_max_pu<=1`。非 `committable` 机组均为可弃发可再生机组，`p_min_pu=0`；没有热机启停、爬坡和固定成本，其功率范围见下文。热机四个爬坡系数均在 [0,1]，省略/原生 NaN 按 1；最短开停时间为 0–8 的整数。储能 `-1<=p_min_pu<=0<=p_max_pu<=1`, `max_hours>0`, 效率在 (0,1]，`standing_loss` 在 [0,1]，初始电量在容量内；无天然入流、泄流或循环边界。

## 场景树与 options

`tree` 是 1–25 个节点对象的数组；每个节点必须恰含下列字段：

```text
id: 非空唯一字符串
parent: 父节点 id，根为 null
stage: 0..7 的整数
probability: 从父节点到本节点的条件概率，严格正数，根为1
load: {每个 Load 名称: 本小时非负需求 MW}
renewable_pu: {每个非可开停 Generator 名称: 本小时可用系数[0,1]}
line_status: {每条 Line 名称: true 或 false}
```

映射包含对应类别的全部名称，空类别用 `{}`。恰有一个 stage=0 的根，每条父子边的 stage 差 1，全部叶节点同一 stage；每个非叶节点的子条件概率之和为 1（输入校验误差 1e-10）。树数组不要求按时间或字典排序。每项负荷最多 1e5 MW。

`options` 必须恰含：

```text
risk_weight: λ ∈ [0,1]
alpha: α ∈ [0,0.95]
initial: {每台热机: {status:0或1, p:MW, hours:正整数, root_status:0或1}}
terminal_soc: {每台储能: 每个叶端最低电量 MWh}
shed_cost: {每个负荷: 未供电单位成本/ MWh}
throughput_cost: {每台储能: 单位充电及放电吞吐费用/ MWh}
```

初始 `status,p` 表示 root 开始前小时 -1 的状态和出力；`hours` 是截至该小时前已连续保持此开/停状态的小时数（1–10000）。初始出力必须位于此状态的容量范围。`root_status` 是事先已承诺的 root 开停状态。`terminal_soc` 位于 [0, p_nom*max_hours]。映射恰好覆盖对应组件。所有数有限。结构或支持域错误抛 `ValueError`（异常消息不限）；合法输入但约束无解返回 `{"status":"infeasible"}`。

## 显式数值支持域 v2

以下是本版本公开、可执行的数值支持域，与前述物理关系同时适用。所有区间均含端点；结构整数保持前述整数定义，有量纲实数、比例和概率按输入转换所得的 IEEE754 binary64 值解释。没有按求解器表现筛选工况的隐含条件。

| 数值字段 | 公开区间及单位 |
|---|---|
| Bus.v_nom | [1e-3,1e5] kV |
| Line.x | [1e-12,1e6] 欧姆 |
| Line.s_nom | [0,1e6] MW |
| Generator.p_nom、StorageUnit.p_nom | [1e-6,1e6] MW |
| StorageUnit.max_hours | [1e-6,1e4] h |
| efficiency_store、efficiency_dispatch | [1e-6,1] |
| Generator 的四类成本、StorageUnit.marginal_cost、shed_cost、throughput_cost | [0,1e13]，成本单位沿前文 |
| 每项节点 load | [0,1e5] MW，与前文相同 |
| 发电/储能功率比例、可再生可用系数、爬坡系数、standing_loss | 前文区间与排序不变；原生缺省爬坡 NaN 解释为1 |
| risk_weight、alpha | 分别[0,1]、[0,0.95]，不变 |
| 条件 probability | (0,1]，根为1；任意正 binary64 值包括最小正次正规数均不因概率大小拒绝 |
| initial.p、state_of_charge_initial、terminal_soc | 前文状态/容量区间不变，容量由上述范围计算 |

本题仍是宽尺度的合成调度研究模型，不将电压上限解释为常见实际电网电压。上述范围保证容量、能量、成本与电气系数的派生量不发生普通浮点溢出；这不保证任意混合整数问题都在预算内求出最优值。尤其不得将小概率与大后果的乘积直接删除。

**角度序列化预算。** 对每个节点的实际在运拓扑，按前文选每个连通分量的参考母线 r。对在运线路 l 定义 `b_l=v_nom**2/x` 及 `w_l=s_nom/b_l`。令 `T_b` 是 r 到母线 b 的所有在运路径中 `sum(w_l)` 的最小值，`T_r=0`；孤立母线也取0。零容量在运线路的 w 为0，仍属于电气连通图。因为每线 `|f|<=s_nom`，任一可行物理解都满足 `|theta_b|<=T_b`。

对于 `T>0`，定义

```text
E(T)=min(T, 2**(max(floor(log2(T))-52, -1074)-1))
E(0)=0
```

E 是区间 `[-T,T]` 内实数按最近偶数规则舍入到 binary64 的保守绝对误差界。每条在运线路都必须满足：

```text
b_l * (E(T_bus0) + E(T_bus1)) <= 2e-7 MW
```

全部节点都要满足该条件，停运线不参与；相同拓扑只需验证一次。这里用输入 binary64 数的**精确有理值**计算 b、w、最短路径、二进制指数和比较，阈值 `2e-7` 精确为 `1/5000000`。不能把 `2**-1075` 先算成浮点零，或给预算比较再加容差。公开 `policy_numeric.py` 提供完整检查，不要求求解者重复推导或采用某种优化算法。该预算是从已知容量和参考角推得的输出舍入充分条件，与未知最优策略、求解器是否成功无关；它不是条件数筛选或全域算法正确性保证。实际返回仍须满足全部原有方程、目标和数值容差，错误数值结果不能因通过此预算而被称为成功。

范围或预算不符抛 `ValueError`。求解器失败/未证最优的状态语义不变。

## 信息、机组和历史

在节点 n，下达下一小时的机组开停命令；因此同一父节点的所有子节点对每台热机必须具有相同 u。当期 p、储能动作、削负荷可依本节点观测不同。祖先只有一个状态和一套动作，不能为不同叶路径重新决定。root 的 u 固定为 `root_status`。

每节点热机启动 `start=max(0,u-u_prev)`，停机 `stop=max(0,u_prev-u)`，严格为 0/1；root 的 prev 来自 initial。`P*p_min_pu*u <= p <= P*p_max_pu*u`。按 PyPSA 本版本可开停发电机爬坡语义：

```text
p - p_prev <= P*(ramp_limit_up*u_prev + ramp_limit_start_up*(u-u_prev))
p_prev - p <= P*(ramp_limit_down*u + ramp_limit_shut_down*(u_prev-u))
```

沿每条路径遵守最短开/停时间，initial 尚未履行的义务也延续到 root 及后继；root 的状态变化和爬坡需要正常约束并计费。末端只约束窗口内实际时段，剩余义务写入输出供继续运行使用，不强迫末端前完整履行、不清除剩余义务。

非可开停机组 `0<=p<=p_nom*p_max_pu*renewable_pu`，可弃发。

## 电气与储能

在运线路从 bus0 到 bus1 为正：

```text
f = (v_nom(bus0)**2/x)*(theta_bus0-theta_bus1), |f| <= s_nom
```

停运线路 f=0 且不再关联两端相角。每个实际连通分量选按 Python 字符串顺序最小的母线作为零角度。每母线：发电 + 放电 - 充电 + 削负荷 - 需求 = 向外线路净流出。每项削负荷在 [0,该项需求]。不引入免费功率源、额外耗散或虚拟线路。

每储能节点结束电量：

```text
e = (1-standing_loss)*e_prev + efficiency_store*charge - discharge/efficiency_dispatch
0 <= charge <= -p_min_pu*p_nom
0 <= discharge <= p_max_pu*p_nom
0 <= e <= p_nom*max_hours
```

root 的 e_prev 是原生 `state_of_charge_initial`（root 开始时的电量），其余节点来自真实父节点结束电量。一个小时不能同时充放电，均为零允许。每个叶端独立满足最低终端储备。

## 成本和条件风险目标

节点实际成本 c 是发电边际成本乘 p、热机 `stand_by_cost*u + start_up_cost*start + shut_down_cost*stop`、失负荷成本、储能 `throughput_cost*(charge+discharge)+StorageUnit.marginal_cost*discharge` 的总和。节点 c 不乘概率；公共祖先仅按真实小时计一次。

叶子 `V_n=c_n`；非叶子：

```text
V_n = c_n + (1-λ)*Σ_child q(child|n)*V_child
          + λ*CVaR_α({V_child}, q(.|n))
```

离散 CVaR 是最坏 `1-α` 概率质量的平均成本；边界原子只取所需质量。最小化 `V_root`。这是逐层条件风险，不是对整条叶路径总成本做一次静态 CVaR。`α=0` 或 `λ=0` 为期望；单子节点退化为该子值。接受任一根目标最优的可行策略；特别在 λ=1 时不额外要求非尾部分支各自条件最优。每个节点报告的 V 必须根据实际返回动作重新符合定义。

## 输出、精度与资源

成功返回：

```text
{
  status: "optimal",
  root_objective: V_root,
  expected_cost: 根据真实条件概率计算的期望全周期实际成本,
  node_policy: {
    每个节点id: {
      u: {热机:0或1}, start:{热机:0或1}, stop:{热机:0或1},
      p:{所有发电机:MW}, charge:{储能:MW}, discharge:{储能:MW},
      energy:{储能:MWh}, shed:{负荷:MW}, flow:{线路:MW}, theta:{母线:rad},
      local_cost: c_n, risk_to_go: V_n
    }
  },
  leaf_carry_state: {
    每个叶id: {
      generators: {每台热机: {status, p, hours, remaining}},
      energy: {每台储能:叶端MWh}
    }
  }
}
```

`hours` 为截至叶小时结束连续处于该状态的小时数，若全窗口未变且与 initial 相同应包括窗口前 `hours`；`remaining=max(0,对应最短开/停时间-hours)`。数值映射恰覆盖所列组件。求解器未证明最优时返回 `{"status":"limit"}` 或 `{"status":"error"}`，不能称 optimal；这些状态不是成功解。

验收独立重算状态、每母线平衡、线路、历史、成本与风险；数值比较采用 `atol=2e-5, rtol=2e-6`，有向不等式容许绝对 2e-5。最优目标同样采用上述混合误差，不要求某个唯一策略或唯一出力。每次调用 90 秒上限，1 CPU、4 GiB 内存；可用临时目录。应覆盖完整支持域，不把列表顺序、某个隐藏工况或预置结果作为算法条件。
