# Segmented Physical Route SVG Design

**Date:** 2026-04-18

## Goal

在现有 `train_cal` 主数据之上，补一份“按简称拆段后的聚合线路 JSON”，并用独立实验脚本导出一张横向细长的直线 `ribbon` SVG 线路图。布局以 markdown 主数据为准，不继续维护大量手工坐标规则，但允许从文档中提炼少量“有序走廊”约束；显示跨度对超长线路使用 `log1p` 压缩，先验证简称拆段、聚合物理距离和自动排布效果，再考虑并入 `Streamlit`。

## Why This Shape

当前仓库已经有两层关键主数据和一层旧几何：

- `data/master/physical_routes.json`：聚合物理距离基础表，保留原始分支主键，例如 `L2-L12`、`Z1-L8`、`L18-修4尽头`
- `data/master/tracks.json`：简称、正式名称、有效长度、业务属性
- `data/master/topology_layout.json`：旧的手工几何，仅作为历史参考，不再作为本实验的主要布局来源

本次实验不直接改 `app.py`，也不把 `physical_routes.json` 改成新 schema，而是新增一份“拆段后的实验 JSON”并单独生成自动布局 SVG。这样可以同时满足：

- 聚合距离仍然以基础表为准
- 线路名称和 SVG 标签按简称表显示
- 多段线路的段数和顺序按简称表约束
- 现有求解器与 demo 主链路保持不动
- 布局规则足够少，后续更容易演化

## Output Files

- `data/master/segmented_physical_routes.json`
- `src/fzed_shunting/tools/segmented_routes_svg.py`
- `tests/tools/test_segmented_routes_svg.py`

可选导出产物：

- `artifacts/segmented_physical_routes.json`
- `artifacts/segmented_physical_routes.svg`

## JSON Model

顶层对象按现有聚合分支组织，每条记录保留原始分支编码，并附带简称拆段信息。

```json
{
  "branch_code": "L2-L12",
  "display_name": "存5北+存5南",
  "aggregate_physical_distance_m": 626.3,
  "status": "已确认",
  "left_node": "L2",
  "right_node": "L12",
  "segments": [
    {
      "track_code": "存5北",
      "track_name": "存5线北",
      "physical_distance_m": 417.5,
      "effective_length_m": 367.0
    },
    {
      "track_code": "存5南",
      "track_name": "存5线南",
      "physical_distance_m": 208.8,
      "effective_length_m": 156.0
    }
  ]
}
```

规则：

- `branch_code` 保留 `physical_routes.json` 中的原始主键
- `display_name` 使用简称拼接，单段时直接等于简称
- `segments[*].track_code` 必须来自 `tracks.json`
- `segments` 的数量和顺序以简称表为准，不再沿用聚合表的展示名
- `segments[*].physical_distance_m` 使用运行物理距离口径，不使用停车有效长度替代

## Split Rules

### Single-Segment Branches

下列聚合分支直接映射到一个简称轨道：

- `大门L1 -> 联6`
- `L1-L2 -> 渡1`
- `L1-L3 -> 渡2`
- `L3-L4 -> 渡3`
- `L3-L5 -> 临1`
- `L5-L6 -> 临2`
- `L5-Z2 -> 存1`
- `L6-L7 -> 渡4`
- `L6-Z1 -> 渡5`
- `Z1-Z2 -> 渡6`
- `Z2-Z3 -> 渡7`
- `L4-Z3 -> 存2`
- `L4-Z4 -> 存3`
- `L2-Z4 -> 存4北`
- `Z4-L12 -> 存4南`
- `L12-L13 -> 渡8`
- `L13-L14 -> 渡9`
- `L14-L15 -> 渡10`
- `L15-L16 -> 联7`
- `L16-L19 -> 渡11`
- `L16-L17 -> 渡12`
- `L17-L18 -> 渡13`
- `L19-卸轮尽头 -> 轮`
- `L15-抛丸尽头 -> 抛`
- `L9-油漆尽头 -> 油`
- `L7-机库尽头 -> 机库`
- `Z3-L13 -> 预修`

### Multi-Segment Branches

下列聚合分支需要按简称表拆段：

- `L2-L12 -> 存5北 + 存5南`
- `Z1-L8 -> 机北 + 机棚`
- `L7-调梁尽头 -> 调北 + 调棚`
- `L9-洗罐尽头 -> 洗北 + 洗南`
- `L19-修1尽头 -> 修1库外 + 修1库内`
- `L17-修2尽头 -> 修2库外 + 修2库内`
- `L18-修3尽头 -> 修3库外 + 修3库内`
- `L18-修4尽头 -> 修4库外 + 修4库内`

物理距离分配原则：

- 直接有简称边界的数据，按简称边界切分
- 警冲标、平交道、库门等过渡距离并入相邻简称段，使各段求和严格等于聚合总长
- 有效长度仅作为拆段边界参考，不覆盖运行物理距离

## SVG Rendering

实验版 SVG 使用自动图布局，但渲染对象从“股道块”改为“共享端点的连续线网”，整体形状是横向细长的直线 `ribbon`，不再贴近原始图片轮廓。

渲染原则：

- 真实连接关系必须保留
- 相邻股道在共享节点处直接相连，不再画成独立小段再额外加连接线
- `联6` 所在节点组固定在最右
- `修1-修4` 所在节点组固定在最左
- 其它节点根据真实拓扑自动布局
- 横轴表达真实拓扑顺序
- 纵轴不表达业务含义，只负责把支线靠斥力排开
- 线条优先使用直线，先不引入曲线或折线拐点规则
- 对少量文档已明确的串联走廊，保留顺序约束，例如 `L8 -> 机棚 -> 机北 -> Z1` 与 `调棚 -> 调北 -> L7`
- 使用简化排斥策略避免明显重叠，不引入大量个别线路特例

渲染内容：

- 每条简称轨道是一条 `nodeA -> nodeB` 的连续直线路径
- 标签显示 `track_code`
- 聚合物理距离放在图例中
- 多段聚合分支在图例中显示 `display_name -> segment list -> aggregate distance`

本阶段不做交互、不接入 `app.py`，只输出静态 SVG，目标是用于人工核对线路拆段和连续线网布局方向是否正确。

## Auto-Layout Strategy

自动布局只保留少量硬约束，其余都交给图布局：

- 以真实连接节点为图节点，以简称股道为图边
- `联6` 相关节点为右侧固定锚点
- `修1库外/内` 至 `修4库外/内` 相关节点为左侧固定锚点
- 横向主顺序由拓扑深度决定，但允许少量“有序走廊”覆盖局部串联顺序
- 画布宽度按层级数自动拉宽；单层或单边跨度对超长线路使用 `base + scale * log1p(length_m)` 压缩，而不是线性放大
- 同层节点只在纵向用斥力和少量邻接吸引展开
- `x` 方向不参与斥力漂移，保持清晰的从左到右拓扑顺序
- 每条股道的标签挂在对应直线中点附近

最终目标不是复刻现场图，而是自动得到一张方向正确、横向细长、线网连续、少规则维护的线路示意图。

## Testing

测试分两类：

- 主数据测试：校验拆段 JSON 中关键分支的简称顺序、段数和距离求和
- SVG 测试：校验导出的 SVG 保持左右锚点顺序、关键走廊顺序正确、整体宽高比明显横向拉长、主链 x 单调递增、相邻股道共享节点连续、输出直线路径、并且超长线路对画布宽度的影响已被 `log1p` 压缩

## Integration Later

实验通过后，再决定是否把该 JSON 挂入 `MasterData` 正式模型，或者把自动布局 SVG 生成逻辑接入 `app.py`。
