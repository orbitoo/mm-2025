#import "@preview/fletcher:0.5.8" as fletcher: diagram, edge, node
#import fletcher.shapes: diamond
#set page(width: auto, height: auto, margin: (x: 2pt, y: 2pt))

#diagram(
  node-stroke: 1pt,
  node((0, 0), [源域数据 $X_s$], corner-radius: 4pt),
  edge("r,d", "-|>", corner-radius: 4pt),
  node((0, 2), [目标域数据 $X_t$], corner-radius: 4pt),
  edge("r,u", "-|>", corner-radius: 4pt),
  node((1, 1), [统一特征提取]),
  edge("-|>"),
  node((2, 1), [Z-score\ 标准化]),
  edge("u,r", "-|>", [初始\ KNN\ 模型], label-side: left, corner-radius: 4pt, label-pos: 30%),
  node((3, 0), [伪标签 $Y_t$]),
  edge("-|>"),
  node((4, 0), [计算边缘分布 $M_0$\ 计算条件分布 $M_c$\ 构建 MMD 矩阵]),
  edge("-|>"),
  node((4, 1), [求解变换矩阵 $A$\ 训练新 KNN 模型\ 更新伪标签 $Y_t$]),
  edge("-|>"),
  node((3, 1), [终止条件], shape: diamond),
  edge((3, 0), "-|>", [否], label-side: left),
  edge("d,r", "-|>", [是], corner-radius: 4pt),
  node((4, 2), [输出：$Y_t, A$]),
)
