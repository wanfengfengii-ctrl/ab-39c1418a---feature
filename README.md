# 闪烁体脉冲反卷积服务（pulse-deconvolve）

高计数率下，相邻闪烁事件的响应尾部会互相叠加，直接把局部峰值当成独立脉冲会产生误判。本服务从离散采样中**联合**反演事件序列：在所有可起始位置上联合选择非负整数幅度，用零填充离散卷积回算全部采样，而不是逐峰扣除响应尾部。

## 数学模型

观测采样 `s[0..n-1]` 建模为事件幅度序列 `x[0..n-1]`（非负整数，每项 ≤ `maxAmplitude`）与正整数响应核 `k[0..m-1]` 的零填充离散卷积的前 `n` 个输出：

```
pred[i] = Σ_{j=0}^{m-1} k[j] · x[i-j]      （t < 0 时 x[t] = 0）
residual[i] = samples[i] - prediction[i]
```

求解目标按字典序依次最小化：

1. **最大绝对残差** `max |residual[i]|`（硬约束：≤ `maxResidual`）
2. **残差绝对值总和** `Σ |residual[i]|`
3. **事件数**（非零幅度个数，硬约束：≤ `maxEvents`）
4. **幅度序列本身**的字典序

由于卷积矩阵是带状下三角的，固定 `x[0..p-1]` 后采样 `p` 只依赖最近 `m-1` 个幅度，且给定逐点残差半径时 `x[p]` 的可行取值是连续整数区间。求解器据此做精确动态规划（先二分最小可行残差半径，再在该半径下优化其余目标），保证全局最优，而非贪心逐峰剥离。

### 强脉冲削顶（可选）

强脉冲叠加会让闪烁体读出触及模数转换上限：采样停在平台值并不代表真实预测等于该值。同时提供 `saturationLevel` 与 `totalAmplitudeRange`（独立积分读数给出的事件幅度总和闭区间 `[lo, hi]`）即可启用削顶反演：

- 采样 **低于** `saturationLevel`：仍按原二侧残差 `|samples[i] - pred[i]| ≤ maxResidual` 计算；
- 采样 **等于** `saturationLevel`：只断言 `pred[i] ≥ saturationLevel`（真实预测值不低于平台），该点不计入残差目标；
- 采样 **高于** `saturationLevel`：请求非法（400）。

削顶点的单侧语义、事件数限制与幅度总和闭区间在**同一次**动态规划中求解（状态携带最近 `m-1` 个幅度、已用事件数与已用幅度总和，并用反向“可延伸总量区间”标记前向剪枝），不会先按削顶波形求解再用总量事后筛选。残差半径仍按未削顶点二分最小化。

## API

### `POST /api/pulses/deconvolve`

请求体：

```json
{
  "samples": [0, 0, 0, 0, 15, 10, 17, 8, 4, 0, 0, 0, 0, 0, 0, 0],
  "kernel": [3, 2, 1],
  "maxResidual": 0,
  "maxAmplitude": 10,
  "maxEvents": 4
}
```

| 字段 | 约束 | 含义 |
| --- | --- | --- |
| `samples` | 12–36 个整数 | 观测采样（不得出现高于 `saturationLevel` 的值） |
| `kernel` | 3–6 个正整数 | 脉冲响应核 |
| `maxResidual` | ≥ 0 的整数 | 统一逐点绝对残差上限 |
| `maxAmplitude` | ≥ 1 的整数 | 事件整数幅度上限 |
| `maxEvents` | ≥ 0 的整数 | 最大事件数 |
| `saturationLevel` | 选填，≥ 0 的整数 | ADC 削顶电平；等于它的采样视为削顶（真实预测不低于该值）。必须与 `totalAmplitudeRange` 同时填写 |
| `totalAmplitudeRange` | 选填，`[lo, hi]` 两个整数，`0 ≤ lo ≤ hi` | 独立积分读数给出的、全部事件幅度之和的闭区间。必须与 `saturationLevel` 同时填写 |

两个字段必须同时出现或同时不出现，只填一个返回 400；未填写时请求、响应与裁决完全保持原样。

**成功（200）**：给出事件位置与幅度、完整幅度序列、完整预测波形、逐点残差及目标值。

削顶模式启用时响应额外包含（未启用时不出现任何新字段）：

- `totalAmplitude`：恢复出的事件幅度总和（落在请求闭区间内）；
- `totalAmplitudeRange`：回显本次约束的闭区间；
- `saturationEvidence`：每个削顶点的证据 `{position, observed, predicted, excess}`，其中 `predicted` 是**完整未削顶**预测值，`excess = predicted - saturationLevel ≥ 0`。`prediction`/`residuals` 始终是完整未削顶波形，削顶点的残差可能为负且其绝对值不受 `maxResidual` 约束；目标值只在未削顶点上统计。

```json
{
  "status": "ok",
  "events": [{"position": 4, "amplitude": 5}, {"position": 6, "amplitude": 4}],
  "amplitudes": [0, 0, 0, 0, 5, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  "prediction": [0, 0, 0, 0, 15, 10, 17, 8, 4, 0, 0, 0, 0, 0, 0, 0],
  "residuals": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  "objectives": {"maxAbsResidual": 0, "sumAbsResidual": 0, "eventCount": 2}
}
```

**不可反演（422）**：当事件数或任一点残差无法满足限制时，返回可识别的结论，并指出**首个无法被任何完整候选解释的采样位置**——即不存在这样的完整幅度序列（各项在 `[0, maxAmplitude]` 内、事件数 ≤ `maxEvents`）能让 `samples[0..p]` 的残差全部落在 `±maxResidual` 内的最小 `p`。

```json
{
  "status": "infeasible",
  "detail": "no event train within the amplitude and event-count limits can reproduce the samples within the residual limit",
  "firstUnexplainablePosition": 5
}
```

削顶模式下，若积分闭区间与所有可解释事件列冲突（逐点削顶语义与事件数限制本身仍可满足，但没有任何完整候选的幅度总和落在区间内），422 会标明总量约束并指出**首次失去可延伸候选的位置**：

```json
{
  "status": "infeasible",
  "detail": "no event train within the amplitude, event-count and total-amplitude limits can reproduce the samples (pointwise clipping at the saturation level) within the residual limit",
  "firstUnexplainablePosition": 0,
  "constraint": "totalAmplitudeRange",
  "totalAmplitudeRange": [1, 2]
}
```

`constraint` 取值 `totalAmplitudeRange`（总量闭区间致死）或 `residualLimit`（逐点残差/削顶/事件数限制本身不可满足，此时位置含义与普通模式一致）。

当削顶约束过弱（大量平台采样叠加宽松的事件数/总量上限）使精确联合动态规划的逐层状态数超过预算时，返回 `constraint: "stateBudgetExceeded"`（仍为 422），提示收紧 `maxEvents` 或 `totalAmplitudeRange`，避免欠定输入下的组合爆炸；真实可反演标定实例（紧积分区间、少量削顶点）远低于该预算。

**请求非法（400）**：字段缺失、采样数不足/过多、响应核长度或取值越界等，返回 `{"status": "invalid_request", "detail": [...]}`。

### `GET /healthz`

健康检查端点，返回 `{"status": "ok"}`；Docker 健康检查用它确认服务可接单。

## 运行（Docker Compose）

```bash
docker compose up --build          # 启动 API，并在其健康后自动运行一次 verify
docker compose up api              # 只启动 API
API_HOST_PORT=9000 docker compose up --build   # 自定义宿主机端口（默认 8000）
```

- API 以可配置宿主机端口暴露：`API_HOST_PORT` 环境变量（默认 `8000`），容器内固定为 `8000`。
- `api` 服务配置了健康检查（轮询 `/healthz`），确认服务可接单。
- `verify` 是一次性服务：`depends_on: service_healthy` 等待 API 健康后，依次执行
  1. **代码测试**（pytest 单元/API 测试套件）；
  2. **应用构建检查**（字节码编译全部源码、导入 ASGI 应用、校验路由与 OpenAPI 模式）；
  3. **重叠脉冲 API 冒烟**（精确重叠脉冲恢复、带噪重叠脉冲、残差上限不可满足、事件数不可满足、请求校验、强脉冲削顶恢复、积分区间冲突、削顶字段非法、旧请求形态兼容回归）。

  完成后自行退出，退出码汇总结果：`0` 全部通过；否则为位掩码（`1` 代码测试失败，`2` 构建检查失败，`4` 冒烟失败）。

CI 中可用以下命令把 verify 的退出码作为整条命令的退出码：

```bash
docker compose up --build --exit-code-from verify verify
```

## 本地开发

```bash
pip install -r requirements-dev.txt
pytest                                   # 运行测试
uvicorn app.main:app --reload --port 8000
```

## 项目结构

```
app/
  deconvolve.py   # 联合整数反卷积求解器（动态规划，非逐峰剥离）
  schemas.py      # 请求/响应模型与校验
  main.py         # FastAPI 路由、健康检查、错误语义
tests/
  test_deconvolve.py  # 求解器测试（含暴力枚举交叉验证）
  test_api.py         # API 测试
verify/
  run_checks.py   # verify 一次性服务入口
Dockerfile          # 多阶段：base / api / verify
docker-compose.yml  # api（健康检查、可配置端口）+ verify（一次性）
```
