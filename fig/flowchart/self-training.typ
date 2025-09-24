#import "@preview/fletcher:0.5.8" as fletcher: diagram, edge, node
#import fletcher.shapes: diamond
#set page(width: auto, height: auto, margin: (x: 2pt, y: 2pt))

#diagram(
  node-stroke: 1pt,
  node((0, 0), [在 $(X_s, Y_s)$ 上\ 训练初始分类器 $C_0$]),
  edge("-|>"),
  node((1, 0), [用 $C_0$ 预测 $Y_t$]),
  edge("-|>"),
  node((2, 0), [筛出高置信度样本\ 合入 $X_s$ 得到 $X_s '$]),
  edge("-|>"),
  node((2, 1), [在 $(X_s ', Y_s ')$ 上\ 训练新分类器 $C$]),
  edge("-|>"),
  node((1, 1), [用 $C$ 预测 $Y_t '$]),
  edge("-|>"),
  node((0, 1), [得到最终结果 $Y_t '$], corner-radius: 4pt),
)
