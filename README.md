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

## 强脉冲削顶（ADC 饱和）与积分读数

强脉冲叠加可能把采样顶到模数转换平台，把平台值当成精确观测会低估事件幅度。请求中可同时选填：

- `saturationLevel`：ADC 平台值。**等于**平台的采样是削顶读数，只表示真实预测值不低于平台（`pred[i] >= saturationLevel`，单边约束，不参与残差目标）；**低于**平台的采样仍按原双边残差 `|samples[i] - pred[i]| <= maxResidual` 处理；任何**高于**平台的采样都是非法输入（400）。
- `totalAmplitudeRange`：`[min, max]` 闭区间，来自独立积分读数，约束所有事件幅度之和。必须与 `saturationLevel` 同时给出。

逐点削顶语义、事件数限制与幅度总和闭区间在**同一次**动态规划中联合求解（状态带累计幅度和维度与可延伸性剪枝），不会先按削顶波形求解再用总量事后筛选。四级裁决顺序不变，残差目标只统计未削顶采样。

未填写这两个字段时，请求、响应与不可反演结论与旧版完全一致（响应中不出现 `saturation` 块）。

成功响应保留**完整未削顶预测**（`prediction` 可高于平台），并新增 `saturation` 证据块：

```json
{
  "status": "ok",
  "events": [{"position": 4, "amplitude": 5}],
  "amplitudes": [0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  "prediction": [0, 0, 0, 0, 15, 10, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  "residuals": [0, 0, 0, 0, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  "objectives": {"maxAbsResidual": 0, "sumAbsResidual": 0, "eventCount": 1},
  "saturation": {
    "totalAmplitude": 5,
    "saturatedPoints": [
      {"position": 4, "sample": 12, "prediction": 15, "excess": 3}
    ]
  }
}
```

- `saturation.totalAmplitude`：恢复出的幅度总和（受 `totalAmplitudeRange` 约束）。
- `saturation.saturatedPoints[]`：每个削顶点的位置、平台读数、未削顶预测值及**超出量** `excess = prediction - sample`。

当积分闭区间与所有可解释事件列冲突时，接口返回 422 可识别的不可反演结论，`firstUnexplainablePosition` 指出**首次失去可延伸候选**的采样位置，并通过 `totalAmplitudeRange` 回显总量约束。

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
| `samples` | 12–36 个整数 | 观测采样；启用削顶时不得超过 `saturationLevel` |
| `kernel` | 3–6 个正整数 | 脉冲响应核 |
| `maxResidual` | ≥ 0 的整数 | 未削顶采样的统一逐点残差上限 |
| `maxAmplitude` | ≥ 1 的整数 | 事件整数幅度上限 |
| `maxEvents` | ≥ 0 的整数 | 最大事件数 |
| `saturationLevel` | 选填，≥ 0 的整数 | ADC 平台值；等于它的采样按削顶单边约束处理 |
| `totalAmplitudeRange` | 选填，`[min, max]` 两个 ≥ 0 整数且 `min ≤ max` | 事件幅度总和闭区间；需与 `saturationLevel` 同时给出 |

**成功（200）**：给出事件位置与幅度、完整幅度序列、完整（未削顶）预测波形、逐点残差、目标值，以及启用削顶时的 `saturation` 证据块。

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

**不可反演（422）**：当事件数或任一点残差无法满足限制（启用削顶时还包括幅度总和闭区间）时，返回可识别的结论，并指出**首个无法被任何完整候选解释的采样位置**——即不存在这样的完整幅度序列（各项在 `[0, maxAmplitude]` 内、事件数 ≤ `maxEvents`、给定总量区间时幅度总和落在其中）能让 `samples[0..p]` 在逐点削顶语义与 `±maxResidual`（未削顶点）下全部成立的最小 `p`。给定了 `totalAmplitudeRange` 时响应回显该约束。

```json
{
  "status": "infeasible",
  "detail": "no event train within the amplitude and event-count limits can reproduce the samples within the residual limit",
  "firstUnexplainablePosition": 5,
  "totalAmplitudeRange": [0, 4]
}
```

**请求非法（400）**：字段缺失、采样数不足/过多、响应核长度或取值越界、采样高于 `saturationLevel`、`totalAmplitudeRange` 形状/取值非法或未与 `saturationLevel` 同时给出等，返回 `{"status": "invalid_request", "detail": [...]}`。

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
  3. **重叠脉冲 API 冒烟**（精确重叠脉冲恢复、带噪重叠脉冲、残差上限不可满足、事件数不可满足、请求校验，以及强脉冲削顶恢复、积分区间冲突 422、削顶非法输入与旧请求兼容回归）。

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
