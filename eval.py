from modules import *
from papercandy import *
from datasets import tfc2016


def evaluate(group_name: str, my_network: nn.Module = CNN()) -> str:
    my_config = new_config("test.cfg")
    my_config.set("group_name", group_name)
    CONFIG().CURRENT = my_config
    num_batches = 10000
    ds = tfc2016.TFC2016Image("data/img_test", 28).shuffle()
    my_dataloader = tfc2016.ClassificationDataloader(ds, num_works=my_config.get("num_works", required_type=int, default_val=1),
                                                  batch_size=1)
    nc = NetworkC(my_network)
    nc.load(f"models/{my_config.get('group_name', must_exist=True)}_model.pth")
    tester = Tester(my_dataloader)
    tester.set_network(nc)
    res = tester.test(num_batches)
    num_res = len(res)
    correct_counter = 0
    for i in range(num_res):
        if res[i].input_data.target.cpu()[0] == res[i].output.cpu().max(1, keepdim=True)[1][0].item():
            correct_counter += 1
    with open("results/acc.txt", "a") as fh:
        fh.write(f"{group_name}: {correct_counter * 100 / num_res}%\n")
    return f"Accuracy: {correct_counter * 100 / num_res}%."


if __name__ == '__main__':
    for j in range(1, 31):
        print(evaluate(f"G{j}"))
    for j in range(31, 61):
        print(evaluate(f"G{j}", LeNet5()))
