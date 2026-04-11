# 部分观测生态动力学中的隐藏物种推断

从可见物种的时间序列中恢复未观测的隐藏物种和环境驱动，并用生态一致性与额外解释力来检验恢复结果。

## 快速导航

| 文件 | 用途 |
|------|------|
| [CLAUDE.md](CLAUDE.md) | **入口文件** — 项目目标、架构、评估、常用命令 |
| [notes/](notes/) | 项目文档集（总览、代码地图、实验状态、设计决策、下一步） |
| [docs/research_description.md](docs/research_description.md) | 研究描述文书 |
| [codex_iteration_log.md](codex_iteration_log.md) | 4 轮迭代实验日志 |

## 快速运行

```bash
source .venv/bin/activate
python scripts/run_partial_lv_mvp.py --config configs/partial_lv_mvp.yaml
```

## 技术栈

PyTorch · 合成 Lotka-Volterra 动力学 · CPU 训练