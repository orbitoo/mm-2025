#import "@preview/fletcher:0.5.8" as fletcher: diagram, edge, node
#import fletcher.shapes: diamond, hexagon
#set page(width: auto, height: auto, margin: (x: 2pt, y: 2pt))

#diagram(
  node-stroke: 1pt,
  // node((0, 0), [源域一维信号], corner-radius: 4pt, name: <x>),
  // node((0, 1), [目标一维域信号], corner-radius: 4pt, name: <y>),
  // node((1, 0.5), [二维时频图], name: <out>),
  // let verts = (
  //   ((), "-|", (<y.east>, 50%, <out.west>)),
  //   ((), "|-", <out>),
  //   <out>,
  // ),
  // edge(<x>, ..verts, "-|>"),
  // edge(<y>, ..verts, "-|>"),
  // edge("-|>"),
  node((2, 0.5), [特征提取器 $G_f$], name: <gf>, shape: hexagon, extrude: (-3, 0), inset: 9pt),
  edge((), ((), "|-", <mmd>), <mmd>, "-|>"),
  node((4, 3), [MMD 损失 $L_("MMD")$], name: <mmd>),
  node((4, 0), [标签预测器 $G_y$], name: <gy>, shape: hexagon, extrude: (-3, 0), inset: 9pt),
  node((4, 1), [梯度反转层 GRL\ 前向: $f(x) = x$\ 反向: $f(x)=-alpha x$], name: <grl>, extrude: (-3, 0), inset: 9pt),
  node((5, 1), [领域判别器 $G_d$], name: <gd>, shape: hexagon, extrude: (-3, 0), inset: 9pt),
  let verts = (
    ((), "-|", (<gf.east>, 50%, <gy.west>)),
    ((), "|-", <gf>),
    <gf>,
  ),
  edge(<gy>, ..verts, "<|-"),
  edge(<grl>, ..verts, "<|-"),
  edge(<grl>, <gd>, "-|>"),
  edge("r", "-|>"),
  node((5, 0), [标签损失 $L_y$], name: <ly>),
  edge("u,l,l,l,d", <gf>, "-|>", [反向传播梯度], label-pos: 35%),
  edge(<gy>, "<|-"),
  node((5, 2), [领域损失 $L_d$], name: <ld>),
  edge("l,u", "-|>", [反向传播梯度], label-side: left, label-pos: 25%),
  edge(<gd>, "<|-"),
  node((6, 1), [对抗损失 $L_"adv"$], name: <ladv>),
  node((7, 1), [Adam 优化器\ $L = L_y + alpha L_"adv"+beta L_"MMD"$], extrude: (-3, 0), inset: 9pt),
  edge((), ((), "|-", <mmd>), <mmd>, "<|-"),
  edge((), ((), "|-", <ly>), <ly>, "<|-"),
  edge("l", "<|-"),
)
