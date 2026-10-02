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
| `samples` | 12–36 个整数 | 观测采样 |
| `kernel` | 3–6 个正整数 | 脉冲响应核 |
| `maxResidual` | ≥ 0 的整数 | 统一逐点残差上限 |
| `maxAmplitude` | ≥ 1 的整数 | 事件整数幅度上限 |
| `maxEvents` | ≥ 0 的整数 | 最大事件数 |

**成功（200）**：给出事件位置与幅度、完整幅度序列、完整预测波形、逐点残差及目标值。

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
  3. **重叠脉冲 API 冒烟**（精确重叠脉冲恢复、带噪重叠脉冲、残差上限不可满足、事件数不可满足、请求校验）。

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
