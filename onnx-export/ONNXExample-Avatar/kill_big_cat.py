import onnx
from onnx import helper, ModelProto
import argparse



def split_wide_concats(model: ModelProto, fanin=4) -> ModelProto:
    g = model.graph
    new_nodes = []
    for n in g.node:
        if n.op_type != "Concat" or len(n.input) <= fanin:
            new_nodes.append(n); continue
        axis = next((a.i for a in n.attribute if a.name=='axis'), 1)
        inputs = list(n.input)
        tmp_idx = 0
        while len(inputs) > fanin:
            grouped = []
            for i in range(0, len(inputs), fanin):
                group = inputs[i:i+fanin]
                if len(group) == 1:
                    grouped.append(group[0]); continue
                tmp = f"{n.name or 'Concat'}_tmp_{tmp_idx}"; tmp_idx += 1
                new_nodes.append(helper.make_node("Concat", group, [tmp],
                                                  name=f"{n.name or 'Concat'}_stage_{tmp_idx}",
                                                  axis=axis))
                grouped.append(tmp)
            inputs = grouped
        new_nodes.append(helper.make_node("Concat", inputs, list(n.output), name=n.name, axis=axis))
    g.ClearField("node")
    g.node.extend(new_nodes)
    return model



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()


    m = onnx.load(args.input)
    m = split_wide_concats(m, fanin=7)
    onnx.save(m, args.output)
